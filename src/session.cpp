#include "internal.h"
#include "sampling.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace mula {
namespace {
template<class T> std::vector<T> copy(const T * p,size_t n) {
    if(!p)throw std::invalid_argument("Missing conditioning buffer");return std::vector<T>(p,p+n);
}
void finite(const std::vector<float> & v) {
    if(!std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}))throw std::invalid_argument("Nonfinite conditioning");
}
std::vector<float> branch_input(const std::vector<float> & x,int branch) {
    return branch?std::vector<float>(x.size(),-1.0f):x;
}
using Dumps=std::map<std::string,std::vector<float>>;
void collect(Graph & g,Dumps & values) {
    for(const auto & item:g.outputs) {
        const auto data=g.read(item.first);auto & out=values[item.first];out.insert(out.end(),data.begin(),data.end());
    }
}
std::vector<float> embed(Session & s,int branch) {
    const bool first=s.frame_index==0;const int n=first?s.condition.prompt:1;
    Graph g(s.model);ggml_tensor * x=nullptr;
    for(int q=0;q<9;++q) {
        std::vector<int32_t> token(size_t(n),0);std::vector<float> mask(size_t(n),0);
        for(int t=0;t<n;++t) {
            token[size_t(t)]=first?s.condition.tokens[size_t(t*9+q)]:(q<8?s.previous[size_t(q)]:0);
            mask[size_t(t)]=float(first?s.condition.mask[size_t(t*9+q)]:(q<8));
            if(q<8)token[size_t(t)]+=q*8197;
        }
        ggml_tensor * term;
        if(q==8&&branch)term=ggml_get_rows(g.ctx,s.model.get("unconditional_text_embedding.weight"),g.ids(std::vector<int32_t>(size_t(n),0)));
        else term=ggml_get_rows(g.ctx,s.model.get(q==8?"text_embeddings.weight":"audio_embeddings.weight"),g.ids(token));
        term=ggml_mul(g.ctx,term,g.input(mask,1,n));x=x?ggml_add(g.ctx,x,term):term;
    }
    g.keep("embedding",x);
    if(first) {
        if(branch) {
            g.keep("qwen",s.model.get("unconditional_text_embedding.weight"));
            g.keep("muq",s.model.get("unconditional_text_embedding.weight"));
        } else {
            g.keep("qwen",g.linear("qwen_linear",g.input(s.condition.qwen,1024,1),true));
            g.keep("muq",g.linear("muq_linear",g.input(s.condition.muq,512,1),true));
        }
    }
    g.execute();auto values=g.read("embedding");
    if(first) {
        auto q=g.read("qwen"),m=g.read("muq");
        std::copy(q.begin(),q.end(),values.begin()+s.condition.qwen_index*3072);
        std::copy(m.begin(),m.end(),values.begin()+s.condition.muq_index*3072);
    }
    return values;
}
std::vector<float> run_backbone(Session & s,int branch,Dumps * dumps,std::vector<float> & first_logits) {
    const auto embedding=embed(s,branch);const int n=s.frame_index==0?s.condition.prompt:1;
    Graph g(s.model);auto * x=g.input(embedding,3072,n);
    auto * style=ggml_silu(g.ctx,g.input(branch_input(s.condition.qwen,branch),1024,1));
    auto * piano=g.input(branch_input(s.condition.piano,branch),256,s.condition.symbolic);
    auto * drum=g.input(branch_input(s.condition.drum,branch),128,s.condition.symbolic);
    auto * chord=g.input(branch_input(s.condition.chord,branch),12,s.condition.symbolic);
    auto * one=g.input({1.0f},1,1);
    ggml_tensor * previous_context=nullptr;
    for(int layer=0;layer<28;++layer) {
        const auto l=std::to_string(layer),p="peft_layers."+l;
        if(layer>=20) {
            auto * affine=g.linear("tag_mlp_layers."+l+".mlp",style,true);
            auto * scale=ggml_view_1d(g.ctx,affine,3072,0);
            auto * shift=ggml_view_1d(g.ctx,affine,3072,3072*4);
            if(dumps&&layer==27) {
                scale=g.keep("stage.tag_mlp_layers.27.0",scale);shift=g.keep("stage.tag_mlp_layers.27.1",shift);
            }
            x=ggml_add(g.ctx,ggml_mul(g.ctx,x,ggml_add(g.ctx,scale,one)),shift);
        }
        x=g.transformer("backbone.layers."+l,x,24,8,s.position,true,{},&s.backbone,layer,branch);
        if(dumps&&(layer==0||layer==13||layer==27))x=g.keep("stage.backbone.layers."+l,x);
        if(layer>=20) {
            const auto a=p+".adaptor";
            auto * cp=g.linear(a+".pianoroll_proj",piano,true);
            auto * cd=g.linear(a+".drum_pianoroll_proj",drum,true);
            auto * cc=g.linear(a+".chord_proj",chord,true);
            auto * context=g.linear(a+".fusion_proj",ggml_concat(g.ctx,ggml_concat(g.ctx,cp,cd,0),cc,0),true);
            for(int i=0;i<2;++i)context=g.transformer(a+".decoder.layers."+std::to_string(i),context,8,8,0,false,s.condition.context_mask);
            context=g.norm(a+".decoder.norm",context);
            context=g.linear(a+".out_proj",context,true);
            if(previous_context)context=ggml_add(g.ctx,previous_context,context);
            previous_context=context;
            const auto c=p+".cross_attn";
            auto * normalized=g.norm(c+".norm",x);
            auto * attended=g.attention(c+".attn",normalized,context,8,8,s.position,false,s.condition.context_mask);
            auto * gate_x=g.linear(c+".x_gate_proj",normalized,true);
            auto * gate_context=g.linear(c+".context_gate_proj",context,true);
            auto * gate=g.attention(c+".gate",gate_x,gate_context,1,1,s.position,false,s.condition.context_mask);
            gate=ggml_tanh(g.ctx,g.linear(c+".gate_proj",gate,true));
            x=ggml_add(g.ctx,x,ggml_mul(g.ctx,attended,gate));
            if(dumps&&layer==27) {
                x=g.keep("stage.peft_layers.27.0",x);previous_context=g.keep("stage.peft_layers.27.1",previous_context);
            }
        }
    }
    x=g.norm("backbone.norm",x);if(dumps)x=g.keep("stage.backbone.norm",x);
    auto * last=ggml_view_2d(g.ctx,x,3072,1,x->nb[1],size_t(n-1)*x->nb[1]);
    g.keep("last_hidden",last);g.keep("raw_logits",g.linear("codebook0_head",last));
    g.execute();first_logits=g.read("raw_logits");if(dumps)collect(g,*dumps);
    return g.read("last_hidden");
}
}
Session::Session(const mulacover_config_v1 & config):capacity(config.cache_capacity){model.load(config);}
void Session::begin(const mulacover_condition_v1 & c) {
    if(c.size<sizeof(c)||c.prompt_frames<1||c.prompt_frames>capacity||c.symbolic_frames<1||c.symbolic_frames>5000||
       c.qwen_index<0||c.qwen_index>=c.prompt_frames||c.muq_index<0||c.muq_index>=c.prompt_frames||
       c.qwen_index==c.muq_index||!std::isfinite(c.guidance)||c.guidance<=0)
        throw std::invalid_argument("Invalid prepared conditioning dimensions or guidance");
    Condition next;
    next.prompt=int(c.prompt_frames);next.symbolic=int(c.symbolic_frames);next.qwen_index=int(c.qwen_index);next.muq_index=int(c.muq_index);next.guidance=c.guidance;
    next.tokens=copy(c.tokens,size_t(c.prompt_frames)*9);next.mask=copy(c.tokens_mask,size_t(c.prompt_frames)*9);
    next.context_mask=copy(c.context_mask,size_t(c.symbolic_frames));
    next.qwen=copy(c.qwen,1024);next.muq=copy(c.muq,512);
    next.piano=copy(c.pianoroll,size_t(c.symbolic_frames)*256);next.drum=copy(c.drum_pianoroll,size_t(c.symbolic_frames)*128);next.chord=copy(c.chord,size_t(c.symbolic_frames)*12);
    for(size_t i=0;i<next.tokens.size();++i)if(next.tokens[i]<0||next.tokens[i]>=(i%9==8?128256:8197)||next.mask[i]>1)
        throw std::invalid_argument("Invalid prompt token or mask");
    if(!std::all_of(next.context_mask.begin(),next.context_mask.end(),[](uint8_t x){return x<=1;})||
       std::all_of(next.context_mask.begin(),next.context_mask.end(),[](uint8_t x){return x!=0;}))
        throw std::invalid_argument("Invalid or fully padded symbolic mask");
    for(auto * values:{&next.qwen,&next.muq,&next.piano,&next.drum,&next.chord})finite(*values);
    const int batch=c.guidance>1?2:1;
    if(backbone.branches!=batch||!backbone.buffer||!decoder.buffer) {
        // Preserve an existing session if a replacement cache cannot allocate.
        Cache next_backbone,next_decoder;
        next_backbone.allocate(model,28,128,8,capacity,batch);next_decoder.allocate(model,3,384,4,8,batch);
        backbone.swap(next_backbone);decoder.swap(next_decoder);
    }
    condition=std::move(next);branches=batch;reset();
}
void Session::reset(){backbone.clear();decoder.clear();position=0;frame_index=0;stopped=false;previous.fill(0);}
void Session::frame(const int32_t * codes,const char * dump,mulacover_frame_v1 & result,
        const mulacover_sampling_config_v1 * sampling,int32_t * stop) {
    if(!branches||(!codes&&!sampling)||stopped)throw std::invalid_argument("Begin/reset a nonterminal session before requesting frames");
    if(codes)for(int q=0;q<8;++q)if(codes[q]<0||codes[q]>=8192)throw std::invalid_argument("Teacher RVQ code outside [0,8191]");
    const int n=frame_index==0?condition.prompt:1;
    if(position+n>capacity)throw std::invalid_argument("Backbone cache capacity exhausted");
    const bool dumping=dump&&*dump;Dumps collected;
    std::vector<float> raw(size_t(8*branches*8197)),guided(8*8197);decoder.clear();
    std::array<int32_t,8> selected{};if(codes)std::copy_n(codes,8,selected.begin());
    std::array<std::vector<float>,2> hidden;
    for(int branch=0;branch<branches;++branch) {
        Dumps branch_stages;
        std::vector<float> first;
        hidden[size_t(branch)]=run_backbone(*this,branch,dumping?&branch_stages:nullptr,first);
        std::copy(first.begin(),first.end(),raw.begin()+branch*8197);
        for(auto & stage:branch_stages)if(dumping&&stage.first!="raw_logits"&&stage.first!="last_hidden") {
            auto & data=collected[stage.first];data.insert(data.end(),stage.second.begin(),stage.second.end());
        }
    }
    auto guide_and_select=[&](int q) {
        for(int v=0;v<8197;++v) {
            const float cond=raw[size_t(q*branches*8197+v)];
            if(branches==2) {const float uncond=raw[size_t((q*branches+1)*8197+v)];guided[size_t(q*8197+v)]=uncond+(cond-uncond)*condition.guidance;}
            else guided[size_t(q*8197+v)]=cond;
        }
        if(sampling)selected[size_t(q)]=sample_logits(guided.data()+q*8197,*sampling,frame_index,q);
    };
    guide_and_select(0);
    for(int q=1;q<8;++q) {
        for(int branch=0;branch<branches;++branch) {
            Graph g(model);
            auto * audio=ggml_get_rows(g.ctx,model.get("audio_embeddings.weight"),g.ids({selected[size_t(q-1)]+(q-1)*8197}));
            auto * input=q==1?ggml_concat(g.ctx,g.input(hidden[size_t(branch)],3072,1),audio,1):audio;
            auto * x=g.linear("projection",input);
            if(dumping)x=g.keep("local."+std::to_string(q)+".projection",x);
            for(int layer=0;layer<3;++layer)x=g.transformer("decoder.layers."+std::to_string(layer),x,8,4,q==1?0:q,true,{},&decoder,layer,branch);
            x=g.norm("decoder.norm",x);if(dumping)x=g.keep("local."+std::to_string(q)+".decoder",x);
            auto * last=ggml_view_2d(g.ctx,x,3072,1,x->nb[1],size_t(x->ne[1]-1)*x->nb[1]);
            g.keep("logits",g.linear("audio_head."+std::to_string(q-1),last));g.execute();
            auto values=g.read("logits");std::copy(values.begin(),values.end(),raw.begin()+(q*branches+branch)*8197);
            if(dumping)for(auto & item:g.outputs)if(item.first!="logits") {
                auto data=g.read(item.first);auto & dest=collected[item.first];dest.insert(dest.end(),data.begin(),data.end());
            }
        }
        guide_and_select(q);
    }
    if(!std::all_of(guided.begin(),guided.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("Nonfinite guided logits");
    auto raw_owned=std::make_unique<float[]>(raw.size()),guided_owned=std::make_unique<float[]>(guided.size());
    std::copy(raw.begin(),raw.end(),raw_owned.get());std::copy(guided.begin(),guided.end(),guided_owned.get());
    if(dumping) {
        std::filesystem::create_directories(dump);const auto prefix=(std::filesystem::path(dump)/("frame"+std::to_string(frame_index))).string();
        for(auto & item:collected)write_f32(prefix+"."+item.first+".f32",item.second);
        for(int q=0;q<8;++q) {
            write_f32(prefix+".raw_logits."+std::to_string(q)+".f32",std::vector<float>(raw.begin()+q*branches*8197,raw.begin()+(q+1)*branches*8197));
            write_f32(prefix+".logits."+std::to_string(q)+".f32",std::vector<float>(guided.begin()+q*8197,guided.begin()+(q+1)*8197));
        }
        backbone.dump(prefix,"backbone",position+n,24);decoder.dump(prefix,"decoder",8,8);
    }
    result.raw_logits=raw_owned.release();result.logits=guided_owned.release();result.branches=branches;
    std::copy(selected.begin(),selected.end(),result.codes);previous=selected;
    position+=n;result.cache_position=position;++frame_index;
    if(sampling){*stop=stop_reason(selected.data());stopped=*stop!=MULACOVER_CONTINUE_V1;}
}
}

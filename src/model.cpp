#include "internal.h"
#include "weight-policy.h"
#include "backend-select.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace mula {
Model::~Model() {
    if(profiling){size_t graph_storage=0;for(auto & p:workspaces)graph_storage+=p.second->buffer_bytes;
        size_t metadata=0;for(auto & p:workspaces)metadata+=ggml_get_mem_size(p.second->ctx);
        std::fprintf(stderr,"[mulacover-optimization-memory] device_score_peak_bytes=%llu workspace_buffer_bytes=%llu workspace_metadata_bytes=%llu workspace_count=%llu\n",(unsigned long long)device_score_bytes,(unsigned long long)graph_storage,(unsigned long long)metadata,(unsigned long long)workspaces.size());}
    if(profiling)std::fprintf(stderr,"[mulacover-score-cache] hits=%llu misses=%llu\n",(unsigned long long)score_cache_hits,(unsigned long long)score_cache_misses);
    if(profiling)for(const auto & p:timings){const auto & t=p.second;std::fprintf(stderr,"[mulacover-profile] {\"role\":\"%s\",\"graphs\":%llu,\"nodes\":%llu,\"build_ms\":%.3f,\"allocate_ms\":%.3f,\"upload_ms\":%.3f,\"compute_ms\":%.3f,\"read_ms\":%.3f}\n",p.first.c_str(),(unsigned long long)t.graphs,(unsigned long long)t.nodes,t.build_ms,t.allocate_ms,t.upload_ms,t.compute_ms,t.read_ms);}
    workspaces.clear(); // Allocators and arenas must die before their backend.
    if(buffer) ggml_backend_buffer_free(buffer);
    if(ctx) ggml_free(ctx);
    if(gguf) gguf_free(gguf);
    if(backend) ggml_backend_free(backend);
}
ggml_tensor * Model::get(const std::string & name) const {
    auto * t=ggml_get_tensor(ctx,name.c_str());
    if(!t) throw std::runtime_error("Missing tensor: "+name);
    return t;
}
void Model::validate() const {
    std::string precision="f32";
    const int precision_key=gguf_find_key(gguf,"mulacover.weight_precision");
    if(precision_key>=0) {
        if(gguf_get_kv_type(gguf,precision_key)!=GGUF_TYPE_STRING)throw std::runtime_error("Invalid weight precision metadata");
        precision=gguf_get_val_str(gguf,precision_key);
        std::transform(precision.begin(),precision.end(),precision.begin(),[](unsigned char c){return char(std::tolower(c));});
        weight_mix(precision);
    }
    if(precision!="f32") {
        const int policy=gguf_find_key(gguf,"mulacover.quantization_policy");
        if(policy<0||gguf_get_kv_type(gguf,policy)!=GGUF_TYPE_STRING||std::string(gguf_get_val_str(gguf,policy))!=weight_policy)
            throw std::runtime_error("Unsupported quantization policy");
    }
    std::set<std::string> names;
    auto shape=[&](const std::string & n,std::initializer_list<int64_t> dims) {
        auto * t=get(n); int i=0;
        if(t->type!=weight_type(n,ggml_n_dims(t),t->ne[0],precision))
            throw std::runtime_error("Weight type does not match declared precision policy: "+n);
        for(auto d:dims) if(t->ne[i++]!=d) throw std::runtime_error("Wrong shape: "+n);
        for(;i<4;++i) if(t->ne[i]!=1) throw std::runtime_error("Wrong rank: "+n);
        names.insert(n);
    };
    auto linear=[&](const std::string & n,int in,int out,bool bias=false) {
        shape(n+".weight",{in,out}); if(bias) shape(n+".bias",{out});
    };
    auto tr=[&](const std::string & p,int layers,int d,int heads,int kv,int ff) {
        for(int i=0;i<layers;++i) {
            auto b=p+".layers."+std::to_string(i);
            linear(b+".attn.q_proj",d,d); linear(b+".attn.k_proj",d,kv*(d/heads));
            linear(b+".attn.v_proj",d,kv*(d/heads)); linear(b+".attn.output_proj",d,d);
            linear(b+".mlp.w1",d,ff); linear(b+".mlp.w2",ff,d); linear(b+".mlp.w3",d,ff);
            shape(b+".sa_norm.scale",{d}); shape(b+".mlp_norm.scale",{d});
        }
        shape(p+".norm.scale",{d});
    };
    tr("backbone",28,3072,24,8,8192); tr("decoder",3,3072,8,4,8192);
    shape("text_embeddings.weight",{3072,128256}); shape("audio_embeddings.weight",{3072,8197*8});
    shape("unconditional_text_embedding.weight",{3072,1});
    linear("projection",3072,3072); linear("codebook0_head",3072,8197);
    for(int q=0;q<7;++q) linear("audio_head."+std::to_string(q),3072,8197);
    linear("qwen_linear",1024,3072,true); linear("muq_linear",512,3072,true);
    for(int i=20;i<28;++i) {
        auto p="peft_layers."+std::to_string(i);
        auto a=p+".adaptor";
        linear(a+".pianoroll_proj",256,24,true); linear(a+".drum_pianoroll_proj",128,24,true);
        linear(a+".chord_proj",12,24,true); linear(a+".fusion_proj",72,512,true);
        tr(a+".decoder",2,512,8,8,2048); linear(a+".out_proj",512,3072,true);
        auto c=p+".cross_attn";
        for(auto pair:{std::make_pair("attn",3072),std::make_pair("gate",12)})
            for(auto proj:{"q_proj","k_proj","v_proj","output_proj"}) linear(c+"."+pair.first+"."+proj,pair.second,pair.second);
        linear(c+".x_gate_proj",3072,12,true); linear(c+".context_gate_proj",3072,12,true);
        linear(c+".gate_proj",12,1,true); shape(c+".norm.scale",{3072});
        linear("tag_mlp_layers."+std::to_string(i)+".mlp",1024,6144,true);
    }
    for(int d:{12,64,128,384}) shape("rope.cache."+std::to_string(d),{d,8192});
    if(names.size()!=size_t(gguf_get_n_tensors(gguf))) throw std::runtime_error("Unexpected runtime tensors");
}
void Model::load(const mulacover_config_v1 & config) {
    gguf_init_params p{true,&ctx}; gguf=gguf_init_from_file(config.model_path,p);
    if(!gguf||!ctx) throw std::runtime_error("Cannot read model GGUF");
    int arch=gguf_find_key(gguf,"general.architecture"), version=gguf_find_key(gguf,"mulacover.format_version");
    if(arch<0||gguf_get_kv_type(gguf,arch)!=GGUF_TYPE_STRING||std::string(gguf_get_val_str(gguf,arch))!="mulacover"||
       version<0||gguf_get_kv_type(gguf,version)!=GGUF_TYPE_UINT32||gguf_get_val_u32(gguf,version)!=1)
        throw std::runtime_error("Unsupported MuLaCover GGUF architecture/version");
    validate();
    static std::once_flag once; std::call_once(once,[]{ggml_backend_load_all();});
    const std::string requested(config.backend);
    if(!known_backend(requested.c_str())) throw std::invalid_argument("Backend must be cpu, cuda, vulkan or metal");
    backend=open_backend(requested,"generator");
    if(!backend) throw std::runtime_error("Requested backend unavailable");
    if(requested=="cpu") {
        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto f=reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_set_n_threads"));
        if(f) f(backend,config.threads);
    }
    buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer) throw std::runtime_error("Cannot allocate generator weights");
    ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream file(config.model_path,std::ios::binary|std::ios::ate);
    const auto length=file.tellg();
    for(int64_t i=0;i<gguf_get_n_tensors(gguf);++i) {
        auto * t=get(gguf_get_tensor_name(gguf,i));
        uint64_t offset=gguf_get_data_offset(gguf)+gguf_get_tensor_offset(gguf,i); size_t bytes=ggml_nbytes(t);
        if(length<0||offset>uint64_t(length)||bytes>uint64_t(length)-offset) throw std::runtime_error("Truncated model payload");
        std::vector<uint8_t> data(bytes); file.seekg(std::streamoff(offset));
        if(!file.read(reinterpret_cast<char *>(data.data()),std::streamsize(bytes))) throw std::runtime_error("Cannot read model payload");
        if(!ggml_validate_row_data(t->type,data.data(),bytes)) throw std::runtime_error("Invalid/nonfinite model weights");
        ggml_backend_tensor_set(t,data.data(),0,bytes);
    }
}
Cache::~Cache(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
DeviceScoreCache::~DeviceScoreCache(){clear();}
void DeviceScoreCache::clear(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);buffer=nullptr;ctx=nullptr;layers={};last_context=nullptr;ready=false;}
void DeviceScoreCache::allocate(Model & model,int symbolic) {
    clear();ctx=ggml_init({1024*1024,nullptr,true});
    if(!ctx)throw std::runtime_error("Cannot allocate device score metadata");
    for(auto & l:layers) {
        l.affine=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,6144,1);
        auto kv=[&](int dim,int heads){return AttentionKV{ggml_new_tensor_3d(ctx,GGML_TYPE_F32,dim,symbolic,heads),ggml_new_tensor_3d(ctx,GGML_TYPE_F32,dim,symbolic,heads)};};
        l.attention=kv(384,8);l.gate=kv(12,1);
    }
    last_context=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,3072,symbolic);
    buffer=ggml_backend_alloc_ctx_tensors(ctx,model.backend);
    if(!buffer)throw std::runtime_error("Cannot allocate device score buffers");
}
void Cache::allocate(Model & model,int layers,int dim,int heads,int cap,int batch) {
    if(buffer) {ggml_backend_buffer_free(buffer);buffer=nullptr;}
    if(ctx) {ggml_free(ctx);ctx=nullptr;}
    keys.clear();values.clear();capacity=cap;branches=batch;
    ctx=ggml_init({1024*1024,nullptr,true}); if(!ctx) throw std::runtime_error("Cannot allocate KV metadata");
    for(int i=0;i<layers;++i) {
        keys.push_back(ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,cap,heads,batch));
        values.push_back(ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,cap,heads,batch));
    }
    buffer=ggml_backend_alloc_ctx_tensors(ctx,model.backend);
    if(!buffer) throw std::runtime_error("Cannot allocate KV buffers"); clear();
}
void Cache::clear(){if(buffer)ggml_backend_buffer_clear(buffer,0);}
void Cache::swap(Cache & other) noexcept {
    std::swap(ctx,other.ctx);std::swap(buffer,other.buffer);keys.swap(other.keys);values.swap(other.values);
    std::swap(capacity,other.capacity);std::swap(branches,other.branches);
}
void write_f32(const std::string & path,const std::vector<float> & data) {
    std::ofstream f(path,std::ios::binary);
    if(!f.write(reinterpret_cast<const char *>(data.data()),std::streamsize(data.size()*4))) throw std::runtime_error("Cannot write tensor dump: "+path);
}
void Cache::dump(const std::string & path,const std::string & group,int valid,int query_heads) const {
    for(size_t l=0;l<keys.size();++l) for(int kind=0;kind<2;++kind) {
        auto * t=kind?values[l]:keys[l]; const size_t d=size_t(t->ne[0]),h=size_t(t->ne[2]);
        std::vector<float> data(size_t(branches)*h*size_t(valid)*d);
        std::vector<float> tail(size_t(capacity-valid)*d);
        for(int b=0;b<branches;++b)for(size_t head=0;head<h;++head) {
            size_t offset=size_t(b)*t->nb[3]+head*t->nb[2];
            ggml_backend_tensor_get(t,data.data()+(size_t(b)*h+head)*size_t(valid)*d,offset,size_t(valid)*d*4);
            if(!tail.empty()) {
                ggml_backend_tensor_get(t,tail.data(),offset+size_t(valid)*d*4,tail.size()*4);
                if(!std::all_of(tail.begin(),tail.end(),[](float x){return x==0;})) throw std::runtime_error("KV cache tail modified beyond active positions");
            }
        }
        // Torchtune expands grouped K/V heads before writing its cache. Native
        // attention uses GGML's GQA broadcast; expose the same logical layout
        // in debug dumps without storing redundant physical cache entries.
        if(query_heads%int(h))throw std::runtime_error("Invalid GQA head ratio");
        std::vector<float> expanded(size_t(branches)*size_t(query_heads)*size_t(valid)*d);
        for(int b=0;b<branches;++b)for(int head=0;head<query_heads;++head)
            std::copy_n(data.data()+(size_t(b)*h+size_t(head/(query_heads/int(h))))*size_t(valid)*d,
                size_t(valid)*d,expanded.data()+(size_t(b)*size_t(query_heads)+size_t(head))*size_t(valid)*d);
        write_f32(path+".cache."+group+"."+std::to_string(l)+(kind?".v.f32":".k.f32"),expanded);
    }
}
}

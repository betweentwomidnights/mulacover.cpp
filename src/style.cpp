// Qwen3 backbone follows ACE-Step's qwen3-enc.h graph structure. This strict
// F32 path retains separate projections/manual attention and frozen HF RoPE.
#include "conditioning-internal.h"
#include "backend-select.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>

namespace mula::frontend {
static void load(Model & m,const char * path,const char * backend,int threads){
    if(!path||!*path)throw std::invalid_argument("Nonempty tags require a style model");
    m.gguf=gguf_init_from_file(path,{true,&m.ctx});if(!m.gguf||!m.ctx)throw std::runtime_error("Cannot read style GGUF");
    auto key=[&](const char * n,gguf_type t){int k=gguf_find_key(m.gguf,n);if(k<0||gguf_get_kv_type(m.gguf,k)!=t)throw std::runtime_error(std::string("Invalid style metadata: ")+n);return k;};
    if(std::string(gguf_get_val_str(m.gguf,key("general.architecture",GGUF_TYPE_STRING)))!="mulacover-style"||gguf_get_val_u32(m.gguf,key("mulacover.style_version",GGUF_TYPE_UINT32))!=1||std::string(gguf_get_val_str(m.gguf,key("mulacover.style_revision",GGUF_TYPE_STRING)))!="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")throw std::runtime_error("Unsupported style GGUF architecture/version/revision");
    for(auto item:{std::make_pair("hidden_size",1024),{"intermediate_size",3072},{"num_attention_heads",16},{"num_key_value_heads",8},{"head_dim",128},{"num_hidden_layers",28},{"vocab_size",151669}}){auto n=std::string("style.")+item.first;if(gguf_get_val_u32(m.gguf,key(n.c_str(),GGUF_TYPE_UINT32))!=uint32_t(item.second))throw std::runtime_error("Unsupported Qwen shape config");}
    if(gguf_get_val_f32(m.gguf,key("style.rms_norm_eps",GGUF_TYPE_FLOAT32))!=1e-6f||gguf_get_val_f32(m.gguf,key("style.rope_theta",GGUF_TYPE_FLOAT32))!=1000000.f)throw std::runtime_error("Unsupported Qwen norm/RoPE config");
    std::set<std::string> names;auto shape=[&](const std::string & name,int64_t a,int64_t b=1){auto * t=m.get(name);if(t->type!=GGML_TYPE_F32||t->ne[0]!=a||t->ne[1]!=b||t->ne[2]!=1||t->ne[3]!=1)throw std::runtime_error("Expected strict F32 style tensor: "+name);names.insert(name);};
    shape("embed_tokens.weight",1024,151669);shape("norm.weight",1024);shape("rope.cos",128,8192);shape("rope.sin",128,8192);
    for(int i=0;i<28;++i){auto p="layers."+std::to_string(i);shape(p+".input_layernorm.weight",1024);shape(p+".post_attention_layernorm.weight",1024);for(auto part:{"q","k","v"})shape(p+".self_attn."+part+"_proj.weight",1024,std::string(part)=="q"?2048:1024);shape(p+".self_attn.o_proj.weight",2048,1024);shape(p+".self_attn.q_norm.weight",128);shape(p+".self_attn.k_norm.weight",128);shape(p+".mlp.gate_proj.weight",1024,3072);shape(p+".mlp.up_proj.weight",1024,3072);shape(p+".mlp.down_proj.weight",3072,1024);}
    if(names.size()!=size_t(gguf_get_n_tensors(m.gguf)))throw std::runtime_error("Unexpected style tensor inventory");
    static std::once_flag once;std::call_once(once,[]{ggml_backend_load_all();});
    m.backend=mula::open_backend(backend,"style");
    if(!m.backend)throw std::runtime_error("Requested style backend unavailable");
    if(std::strcmp(backend,"cpu")==0){auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(m.backend));auto set=reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_set_n_threads"));if(set)set(m.backend,threads);}
    m.buffer=ggml_backend_alloc_ctx_tensors(m.ctx,m.backend);if(!m.buffer)throw std::runtime_error("Cannot allocate style weights");ggml_backend_buffer_set_usage(m.buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream file(path,std::ios::binary|std::ios::ate);auto length=file.tellg();
    for(int64_t i=0;i<gguf_get_n_tensors(m.gguf);++i){auto * t=m.get(gguf_get_tensor_name(m.gguf,i));const uint64_t offset=gguf_get_data_offset(m.gguf)+gguf_get_tensor_offset(m.gguf,i);const size_t bytes=ggml_nbytes(t);if(length<0||offset>uint64_t(length)||bytes>uint64_t(length)-offset)throw std::runtime_error("Truncated style weights");std::vector<uint8_t> data(bytes);file.seekg(std::streamoff(offset));if(!file.read(reinterpret_cast<char *>(data.data()),std::streamsize(bytes))||!ggml_validate_row_data(t->type,data.data(),bytes))throw std::runtime_error("Invalid/nonfinite style payload");ggml_backend_tensor_set(t,data.data(),0,bytes);}
}
static ggml_tensor * norm(Graph & g,const std::string & prefix,ggml_tensor * x){return ggml_mul(g.ctx,ggml_rms_norm(g.ctx,x,1e-6f),g.model.get(prefix+".weight"));}
static ggml_tensor * rope(Graph & g,ggml_tensor * x,int64_t heads,int64_t frames){
    auto * a=ggml_view_3d(g.ctx,x,64,heads,frames,x->nb[1],x->nb[2],0);auto * b=ggml_view_3d(g.ctx,x,64,heads,frames,x->nb[1],x->nb[2],64*4);
    auto trig=[&](const char * n){auto * root=g.model.get(n);return ggml_view_3d(g.ctx,root,64,1,frames,128*4,128*4,0);};
    auto * c=trig("rope.cos");auto * s=trig("rope.sin");
    auto * first=ggml_sub(g.ctx,ggml_mul(g.ctx,a,c),ggml_mul(g.ctx,b,s));auto * second=ggml_add(g.ctx,ggml_mul(g.ctx,b,c),ggml_mul(g.ctx,a,s));
    return ggml_concat(g.ctx,first,second,0);
}
std::vector<float> style(const char * path,const char * backend,int threads,const std::vector<int32_t> & ids,const char * directory,
                        const mulacover_conditioning_request_v1 & request,uint64_t & weights,uint64_t & graph){
    checkpoint(request);Model model;load(model,path,backend,threads);weights=ggml_backend_buffer_get_size(model.buffer);checkpoint(request);
    const int64_t n=int64_t(ids.size());const bool dumping=directory&&*directory;
    auto save=[&](const std::string & name,const std::vector<float> & v){if(dumping){std::filesystem::create_directories(directory);write_f32((std::filesystem::path(directory)/(name+".f32")).string(),v);}};
    std::vector<float> hidden;
    {Graph g(model);g.keep("embedding",ggml_get_rows(g.ctx,model.get("embed_tokens.weight"),g.ids(ids)));g.execute();hidden=g.read("embedding");save("style-layer-0",hidden);}
    progress(request,0,28);
    for(int layer=0;layer<28;++layer){
        checkpoint(request);Graph g(model);auto * x=g.input(hidden,1024,n);auto p="layers."+std::to_string(layer);auto a=p+".self_attn";
        auto * z=norm(g,p+".input_layernorm",x);
        auto * q=ggml_reshape_3d(g.ctx,g.linear(a+".q_proj",z),128,16,n);auto * k=ggml_reshape_3d(g.ctx,g.linear(a+".k_proj",z),128,8,n);auto * v=ggml_reshape_3d(g.ctx,g.linear(a+".v_proj",z),128,8,n);
        q=rope(g,norm(g,a+".q_norm",q),16,n);k=rope(g,norm(g,a+".k_norm",k),8,n);
        q=ggml_permute(g.ctx,q,0,2,1,3);k=ggml_cont(g.ctx,ggml_permute(g.ctx,k,0,2,1,3));v=ggml_cont(g.ctx,ggml_permute(g.ctx,v,0,2,1,3));
        std::vector<float> mask(size_t(n*n),0);for(int64_t t=0;t<n;++t)for(int64_t j=t+1;j<n;++j)mask[size_t(t*n+j)]=-INFINITY;
        auto * scores=ggml_scale(g.ctx,g.mm(k,q),1.0f/std::sqrt(128.0f));scores=ggml_soft_max(g.ctx,ggml_add(g.ctx,scores,g.input(mask,n,n)));
        auto * y=g.mm(ggml_cont(g.ctx,ggml_transpose(g.ctx,v)),scores);y=ggml_cont(g.ctx,ggml_permute(g.ctx,y,0,2,1,3));x=ggml_add(g.ctx,x,g.linear(a+".o_proj",ggml_reshape_2d(g.ctx,y,2048,n)));
        z=norm(g,p+".post_attention_layernorm",x);auto * gate=ggml_silu(g.ctx,g.linear(p+".mlp.gate_proj",z));auto * up=g.linear(p+".mlp.up_proj",z);x=ggml_add(g.ctx,x,g.linear(p+".mlp.down_proj",ggml_mul(g.ctx,gate,up)));
        g.keep("hidden",x);g.execute();hidden=g.read("hidden");checkpoint(request);if(layer==0||layer==13)save("style-layer-"+std::to_string(layer+1),hidden);progress(request,layer+1,28);
    }
    {Graph g(model);g.keep("hidden",norm(g,"norm",g.input(hidden,1024,n)));g.execute();hidden=g.read("hidden");}
    checkpoint(request);save("style-hidden",hidden);save("style-layer-28",hidden);graph=model.graph_bytes;
    // HF normalize uses a float32 vector norm; accumulation order can differ
    // slightly across backends and is audited independently of integer tokens.
    float sum=0;for(size_t i=hidden.size()-1024;i<hidden.size();++i)sum+=hidden[i]*hidden[i];const float scale=std::max(std::sqrt(sum),1e-12f);
    std::vector<float> output(1024);for(size_t i=0;i<1024;++i)output[i]=hidden[hidden.size()-1024+i]/scale;return output;
}
}

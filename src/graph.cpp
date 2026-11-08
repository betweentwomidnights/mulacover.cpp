#include "internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mula {
Graph::Graph(Model & m):model(m) {
    ctx=ggml_init({128*1024*1024,nullptr,true});
    if(!ctx) throw std::runtime_error("Cannot allocate graph metadata");
    graph=ggml_new_graph_custom(ctx,32768,false);
}
Graph::~Graph(){if(allocator)ggml_gallocr_free(allocator);if(ctx)ggml_free(ctx);}
ggml_tensor * Graph::input(const std::vector<float> & data,int64_t width,int64_t frames) {
    if(data.size()!=size_t(width)*size_t(frames)) throw std::runtime_error("Internal input shape mismatch");
    auto * t=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,frames);ggml_set_input(t);
    Binding b{t,{}};b.bytes.resize(data.size()*4);std::memcpy(b.bytes.data(),data.data(),b.bytes.size());bindings.push_back(std::move(b));return t;
}
ggml_tensor * Graph::ids(const std::vector<int32_t> & data) {
    auto * t=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,int64_t(data.size()));ggml_set_input(t);
    Binding b{t,{}};b.bytes.resize(data.size()*4);std::memcpy(b.bytes.data(),data.data(),b.bytes.size());bindings.push_back(std::move(b));return t;
}
ggml_tensor * Graph::mm(ggml_tensor * a,ggml_tensor * b) {
    auto * t=ggml_mul_mat(ctx,a,b);
    const auto source_precision=a->type==GGML_TYPE_F32?GGML_PREC_F32:
        ggml_is_quantized(a->type)?GGML_PREC_Q8:GGML_PREC_DEFAULT;
    if(!ggml_prec_set_acc(t,GGML_PREC_F32)||
       (source_precision!=GGML_PREC_DEFAULT&&!ggml_prec_set_src(t,source_precision,1)))
        throw std::runtime_error("Cannot request generator matmul precision");
    return t;
}
ggml_tensor * Graph::linear(const std::string & p,ggml_tensor * x,bool bias) {
    auto * y=mm(model.get(p+".weight"),x);return bias?ggml_add(ctx,y,model.get(p+".bias")):y;
}
ggml_tensor * Graph::norm(const std::string & p,ggml_tensor * x) {
    return ggml_mul(ctx,ggml_rms_norm(ctx,x,1e-5f),model.get(p+".scale"));
}
ggml_tensor * Graph::keep(const std::string & name,ggml_tensor * x) {
    if(x->view_src||x->buffer||x->data)x=ggml_cont(ctx,x);
    ggml_set_name(x,name.c_str());ggml_set_output(x);outputs[name]=x;return x;
}
ggml_tensor * Graph::rope(ggml_tensor * x,ggml_tensor * positions,int dim) {
    const int64_t heads=x->ne[1],time=x->ne[2],pairs=dim/2;
    // Exact upstream cos/sin values; standard GGML views, products and sums.
    // Keep the two products separate, as in Torchtune's FP32 implementation.
    auto * split=ggml_reshape_4d(ctx,x,2,pairs,heads,time);
    split=ggml_cont(ctx,ggml_permute(ctx,split,3,0,1,2)); // [pairs,heads,time,2]
    auto * x1=ggml_view_3d(ctx,split,pairs,heads,time,split->nb[1],split->nb[2],0);
    auto * x2=ggml_view_3d(ctx,split,pairs,heads,time,split->nb[1],split->nb[2],split->nb[3]);
    auto * trig=ggml_get_rows(ctx,model.get("rope.cache."+std::to_string(dim)),positions);
    trig=ggml_reshape_3d(ctx,trig,2,pairs,time);
    trig=ggml_cont(ctx,ggml_permute(ctx,trig,2,0,1,3)); // [pairs,time,2]
    auto * cosine=ggml_view_3d(ctx,trig,pairs,1,time,size_t(pairs)*4,size_t(pairs)*4,0);
    auto * sine=ggml_view_3d(ctx,trig,pairs,1,time,size_t(pairs)*4,size_t(pairs)*4,trig->nb[2]);
    auto * even=ggml_sub(ctx,ggml_mul(ctx,x1,cosine),ggml_mul(ctx,x2,sine));
    auto * odd=ggml_add(ctx,ggml_mul(ctx,x2,cosine),ggml_mul(ctx,x1,sine));
    auto * joined=ggml_concat(ctx,even,odd,3);
    joined=ggml_cont(ctx,ggml_permute(ctx,joined,1,2,3,0));
    return ggml_reshape_3d(ctx,joined,dim,heads,time);
}
ggml_tensor * Graph::attention(const std::string & p,ggml_tensor * x,ggml_tensor * context,int heads,int kv_heads,
                               int pos,bool causal,const std::vector<uint8_t> & padding,Cache * cache,int layer,int branch,
                               const std::vector<float> & score_bias) {
    const int64_t dim=x->ne[0]/heads, queries=x->ne[1], keys=context->ne[1];
    std::vector<int32_t> qpos(size_t(queries),0),kpos(size_t(keys),0);
    for(int64_t t=0;t<queries;++t)qpos[size_t(t)]=pos+int32_t(t);
    for(int64_t t=0;t<keys;++t)kpos[size_t(t)]=cache?pos+int32_t(t):int32_t(t);
    auto * q=rope(ggml_reshape_3d(ctx,linear(p+".q_proj",x),dim,heads,queries),ids(qpos),int(dim));
    auto * k=rope(ggml_reshape_3d(ctx,linear(p+".k_proj",context),dim,kv_heads,keys),ids(kpos),int(dim));
    auto * v=ggml_reshape_3d(ctx,linear(p+".v_proj",context),dim,kv_heads,keys);
    q=ggml_permute(ctx,q,0,2,1,3); // [dim,queries,heads]
    k=ggml_cont(ctx,ggml_permute(ctx,k,0,2,1,3));
    v=ggml_cont(ctx,ggml_permute(ctx,v,0,2,1,3));
    int64_t attended_keys=keys;
    if(cache) {
        auto write=[&](ggml_tensor * root,ggml_tensor * value) {
            auto * dst=ggml_view_3d(ctx,root,dim,keys,kv_heads,root->nb[1],root->nb[2],size_t(branch)*root->nb[3]+size_t(pos)*root->nb[1]);
            ggml_build_forward_expand(graph,ggml_cpy(ctx,value,dst));
            return ggml_view_3d(ctx,root,dim,cache->capacity,kv_heads,root->nb[1],root->nb[2],size_t(branch)*root->nb[3]);
        };
        k=write(cache->keys.at(size_t(layer)),k);v=write(cache->values.at(size_t(layer)),v);
        attended_keys=cache->capacity;
    }
    std::vector<float> mask(size_t(attended_keys*queries),0.0f);
    if(!score_bias.empty()) {
        if(score_bias.size()!=mask.size())throw std::invalid_argument("Wrong score-clock bias dimensions");
        mask=score_bias;
    }
    for(int64_t t=0;t<queries;++t)for(int64_t key=0;key<attended_keys;++key) {
        bool hidden=(causal&&key>pos+t)||(!padding.empty()&&padding.at(size_t(key))!=0);
        if(hidden)mask[size_t(t*attended_keys+key)]=-INFINITY;
    }
    auto * scores=mm(k,q);
    scores=ggml_soft_max_ext(ctx,scores,input(mask,attended_keys,queries),1.0f/std::sqrt(float(dim)),0);
    auto * vt=ggml_cont(ctx,ggml_transpose(ctx,v));
    auto * y=mm(vt,scores);
    y=ggml_cont(ctx,ggml_permute(ctx,y,0,2,1,3));
    return linear(p+".output_proj",ggml_reshape_2d(ctx,y,dim*heads,queries));
}
ggml_tensor * Graph::transformer(const std::string & p,ggml_tensor * x,int heads,int kv,int pos,bool causal,
                                 const std::vector<uint8_t> & padding,Cache * cache,int layer,int branch) {
    auto * n=norm(p+".sa_norm",x);
    x=ggml_add(ctx,x,attention(p+".attn",n,n,heads,kv,pos,causal,padding,cache,layer,branch));
    n=norm(p+".mlp_norm",x);
    auto * gate=ggml_silu(ctx,linear(p+".mlp.w1",n));
    auto * up=linear(p+".mlp.w3",n);
    return ggml_add(ctx,x,linear(p+".mlp.w2",ggml_mul(ctx,gate,up)));
}
void Graph::execute() {
    for(auto & pair:outputs)ggml_build_forward_expand(graph,pair.second);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto * n=ggml_graph_node(graph,i);
        if(!ggml_backend_supports_op(model.backend,n))throw std::runtime_error("Unsupported native operation: "+std::string(ggml_op_name(n->op)));
    }
    allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if(!allocator||!ggml_gallocr_alloc_graph(allocator,graph))throw std::runtime_error("Cannot allocate generator graph");
    model.graph_bytes=std::max(model.graph_bytes,ggml_gallocr_get_buffer_size(allocator,0));
    for(const auto & b:bindings)ggml_backend_tensor_set(b.tensor,b.bytes.data(),0,b.bytes.size());
    if(ggml_backend_graph_compute(model.backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Generator graph execution failed");
}
std::vector<float> Graph::read(const std::string & name) const {
    auto * t=outputs.at(name);std::vector<float> data(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t,data.data(),0,data.size()*4);
    if(!std::all_of(data.begin(),data.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("Nonfinite native tensor: "+name);
    return data;
}
}

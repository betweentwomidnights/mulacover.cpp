#pragma once
#include "mulacover.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mula {
struct Model {
    ggml_context * ctx=nullptr;
    gguf_context * gguf=nullptr;
    ggml_backend_t backend=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    size_t graph_bytes=0;
    Model()=default;
    Model(const Model &)=delete;
    Model & operator=(const Model &)=delete;
    ~Model();
    void load(const mulacover_config_v1 &);
    ggml_tensor * get(const std::string &) const;
    void validate() const;
};
struct Cache {
    ggml_context * ctx=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    std::vector<ggml_tensor *> keys,values;
    int capacity=0,branches=0;
    Cache()=default;
    Cache(const Cache &)=delete;
    Cache & operator=(const Cache &)=delete;
    ~Cache();
    void allocate(Model &,int layers,int head_dim,int heads,int capacity,int branches);
    void clear();
    void swap(Cache &) noexcept;
    void dump(const std::string & path,const std::string & group,int valid,int query_heads) const;
};
struct Binding { ggml_tensor * tensor; std::vector<uint8_t> bytes; };
struct Graph {
    Model & model;
    ggml_context * ctx=nullptr;
    ggml_cgraph * graph=nullptr;
    ggml_gallocr_t allocator=nullptr;
    std::vector<Binding> bindings;
    std::map<std::string,ggml_tensor *> outputs;
    explicit Graph(Model &);
    Graph(const Graph &)=delete;
    Graph & operator=(const Graph &)=delete;
    ~Graph();
    ggml_tensor * input(const std::vector<float> &,int64_t width,int64_t frames);
    ggml_tensor * ids(const std::vector<int32_t> &);
    ggml_tensor * mm(ggml_tensor *,ggml_tensor *);
    ggml_tensor * linear(const std::string &,ggml_tensor *,bool bias=false);
    ggml_tensor * norm(const std::string &,ggml_tensor *);
    ggml_tensor * keep(const std::string &,ggml_tensor *);
    ggml_tensor * rope(ggml_tensor *,ggml_tensor * positions,int head_dim);
    ggml_tensor * attention(const std::string &,ggml_tensor * x,ggml_tensor * context,int heads,int kv_heads,
                           int pos,bool causal,const std::vector<uint8_t> & padding,Cache * cache=nullptr,int layer=0,int branch=0);
    ggml_tensor * transformer(const std::string &,ggml_tensor *,int heads,int kv_heads,int pos,bool causal,
                             const std::vector<uint8_t> & padding,Cache * cache=nullptr,int layer=0,int branch=0);
    void execute();
    std::vector<float> read(const std::string &) const;
};
struct Condition {
    std::vector<int32_t> tokens;
    std::vector<uint8_t> mask,context_mask;
    std::vector<float> qwen,muq,piano,drum,chord;
    int prompt=0,symbolic=0,qwen_index=0,muq_index=0;
    float guidance=1;
};
struct Session {
    Model model;
    Cache backbone,decoder;
    Condition condition;
    int capacity=8192,branches=0,position=0,frame_index=0;
    std::array<int32_t,8> previous{};
    explicit Session(const mulacover_config_v1 &);
    void begin(const mulacover_condition_v1 &);
    void reset();
    void frame(const int32_t *,const char *,mulacover_frame_v1 &);
};
void write_f32(const std::string &,const std::vector<float> &);
}
struct mulacover_context { std::unique_ptr<mula::Session> session; };

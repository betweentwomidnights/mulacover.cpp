#pragma once
#include "ggml.h"
#include <stdexcept>
#include <string>

namespace mula {
inline constexpr const char * weight_policy = "mulacover-kquant-v1";
struct WeightMix { ggml_type body, critical; };
inline WeightMix weight_mix(const std::string & mix) {
    if(mix=="f32")return {GGML_TYPE_F32,GGML_TYPE_F32};
    if(mix=="f16")return {GGML_TYPE_F16,GGML_TYPE_F16};
    if(mix=="q8_0")return {GGML_TYPE_Q8_0,GGML_TYPE_Q8_0};
    if(mix=="q6_k")return {GGML_TYPE_Q6_K,GGML_TYPE_Q6_K};
    if(mix=="q5_k_m")return {GGML_TYPE_Q5_K,GGML_TYPE_Q6_K};
    if(mix=="q4_k_m")return {GGML_TYPE_Q4_K,GGML_TYPE_Q6_K};
    throw std::runtime_error("Unsupported MuLaCover weight precision: "+mix);
}
inline ggml_type weight_type(const std::string & name,int dims,int64_t row,const std::string & mix) {
    const auto types=weight_mix(mix);
    // SA3's policy: V/down/embed at the critical type, other block-aligned 2D
    // weights at the body type. MuLaCover's eight output heads are also critical.
    // Norms, biases, exact RoPE tables and non-block-aligned rows stay F32.
    if(dims!=2||name.size()<7||name.compare(name.size()-7,7,".weight")!=0)return GGML_TYPE_F32;
    const bool critical=name.find(".v_proj.weight")!=std::string::npos||
        name.find(".mlp.w2.weight")!=std::string::npos||
        name=="text_embeddings.weight"||name=="audio_embeddings.weight"||
        name=="codebook0_head.weight"||name.rfind("audio_head.",0)==0;
    const auto target=critical?types.critical:types.body;
    return row%ggml_blck_size(target)==0?target:GGML_TYPE_F32;
}
}

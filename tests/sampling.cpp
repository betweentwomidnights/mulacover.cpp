#include "sampling.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

int main() {
    mulacover_sampling_config_v1 c{};c.size=sizeof(c);c.top_k=2;c.temperature=1;c.seed=1701;
    std::vector<float> logits(8197,-1000),noise(8*8197,1);logits[4]=1;logits[8]=0;logits[9]=0;
    auto p=mula::sampling_probabilities(logits.data(),1,2);
    if(p[4]<=p[8]||p[8]!=p[9]||p[10]!=0||std::fabs(std::accumulate(p.begin(),p.end(),0.0)-1)>1e-6)return 1;
    noise[8]=0.01f;c.exponential_noise=noise.data();c.noise_count=int64_t(noise.size());mula::validate_sampling(c);
    if(mula::sample_logits(logits.data(),c,0,0)!=8)return 1;
    c.top_k=1;if(mula::sample_logits(logits.data(),c,0,0)!=4)return 1;
    c.top_k=0;try{mula::validate_sampling(c);return 1;}catch(const std::invalid_argument &){}
    c.top_k=8197;noise[23]=0;try{mula::validate_sampling(c);return 1;}catch(const std::invalid_argument &){}
    c.exponential_noise=nullptr;c.noise_count=0;c.temperature=std::numeric_limits<float>::infinity();
    try{mula::validate_sampling(c);return 1;}catch(const std::invalid_argument &){}
    c.temperature=1;mula::validate_sampling(c);
    // Independent integer SplitMix64 reference vectors for a uniform categorical distribution.
    std::vector<float> flat(8197,1);
    if(mula::sample_logits(flat.data(),c,0,0)!=2636||mula::sample_logits(flat.data(),c,1,0)!=5605||
       mula::sample_logits(flat.data(),c,0,1)!=5410)return 1;
    // Torch F32 race has an exact tie here; double division incorrectly picks 8.
    std::vector<float> pair(8197,-1000);pair[4]=1;pair[8]=0;
    noise.assign(8*8197,1);noise[4]=0.043003447353839874f;noise[8]=0.015820084139704704f;
    c.top_k=2;c.exponential_noise=noise.data();c.noise_count=int64_t(noise.size());
    if(mula::sample_logits(pair.data(),c,0,0)!=4)return 1;
    logits[1]=std::numeric_limits<float>::quiet_NaN();try{mula::sampling_probabilities(logits.data(),1,250);return 1;}catch(const std::runtime_error &){}
    int32_t codes[8]={0};if(mula::stop_reason(codes))return 1;
    codes[0]=8192;if(mula::stop_reason(codes)!=MULACOVER_PADDING_V1)return 1;
    codes[7]=8193;if(mula::stop_reason(codes)!=MULACOVER_EOS_V1)return 1;
    return 0;
}

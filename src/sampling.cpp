#include "sampling.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mula {
void validate_sampling(const mulacover_sampling_config_v1 & c) {
    if(c.size<sizeof(c)||!std::isfinite(c.temperature)||c.temperature<=0||c.top_k<1||c.top_k>8197||
       (c.exponential_noise?c.noise_count!=8*8197:c.noise_count!=0))
        throw std::invalid_argument("Invalid sampling configuration");
    if(c.exponential_noise)for(int64_t i=0;i<c.noise_count;++i)
        if(!std::isfinite(c.exponential_noise[i])||c.exponential_noise[i]<=0)
            throw std::invalid_argument("Exponential noise must be finite and positive");
}
std::vector<float> sampling_probabilities(const float * logits,float temperature,int top_k) {
    std::vector<float> scaled(8197);
    const float inverse_temperature=1.0f/temperature;
    for(int i=0;i<8197;++i)scaled[size_t(i)]=logits[i]*inverse_temperature;
    std::vector<float> sorted=scaled;
    if(!std::all_of(sorted.begin(),sorted.end(),[](float x){return std::isfinite(x);}))
        throw std::runtime_error("Nonfinite sampling logits");
    std::nth_element(sorted.begin(),sorted.begin()+top_k-1,sorted.end(),std::greater<float>());
    const float threshold=sorted[size_t(top_k-1)],maximum=*std::max_element(sorted.begin(),sorted.end());
    std::vector<float> probabilities(8197),log_probabilities(8197,-INFINITY);double sum=0;
    for(int i=0;i<8197;++i)if(scaled[size_t(i)]>=threshold) {
        probabilities[size_t(i)]=std::exp(scaled[size_t(i)]-maximum);sum+=probabilities[size_t(i)];
    }
    const float log_sum=std::log(float(sum));
    float log_max=-INFINITY;
    for(int i=0;i<8197;++i)if(scaled[size_t(i)]>=threshold) {
        log_probabilities[size_t(i)]=(scaled[size_t(i)]-maximum)-log_sum;
        log_max=std::max(log_max,log_probabilities[size_t(i)]);
    }
    // Preserve upstream log_softmax followed by softmax, including F32 rounding.
    sum=0;
    for(int i=0;i<8197;++i){probabilities[size_t(i)]=std::exp(log_probabilities[size_t(i)]-log_max);sum+=probabilities[size_t(i)];}
    const float denominator=float(sum);
    for(auto & p:probabilities)p/=denominator;
    return probabilities;
}
static double exponential(uint64_t seed,uint64_t index) {
    uint64_t z=seed+(index+1)*UINT64_C(0x9e3779b97f4a7c15);
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);z^=z>>31;
    // The open interval also covers the largest representable integer safely.
    const double u=(double(z>>12)+0.5)*0x1.0p-52;
    return -std::log(u);
}
int32_t sample_logits(const float * logits,const mulacover_sampling_config_v1 & c,int frame,int head) {
    const auto probabilities=sampling_probabilities(logits,c.temperature,c.top_k);
    double best=-1;int32_t selected=0;
    for(int i=0;i<8197;++i) {
        const uint64_t index=(uint64_t(frame)*8+uint64_t(head))*8197+uint64_t(i);
        const double noise=c.exponential_noise?c.exponential_noise[head*8197+i]:exponential(c.seed,index);
        const double score=probabilities[size_t(i)]/noise;
        if(score>best){best=score;selected=i;}
    }
    return selected;
}
int32_t stop_reason(const int32_t * codes) {
    for(int q=0;q<8;++q)if(codes[q]>=8193)return MULACOVER_EOS_V1;
    for(int q=0;q<8;++q)if(codes[q]==8192)return MULACOVER_PADDING_V1;
    return MULACOVER_CONTINUE_V1;
}
}

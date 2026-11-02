#pragma once
#include "mulacover_sampling.h"
#include <vector>

namespace mula {
void validate_sampling(const mulacover_sampling_config_v1 &);
std::vector<float> sampling_probabilities(const float *,float temperature,int top_k);
int32_t sample_logits(const float *,const mulacover_sampling_config_v1 &,int frame,int head);
int32_t stop_reason(const int32_t * codes);
}

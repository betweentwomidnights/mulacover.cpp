#include "mulacover.h"
#include "mulacover_sampling.h"
#include "mulacover_tempo.h"
#include "mulacover_continuation.h"
#include "mulacover_job.h"
#include "mulacover_conditioning.h"
#include <string.h>
int main(void) {
    const mulacover_conditioning_api_v1 * frontend=mulacover_get_conditioning_api(1);
    if(!frontend||frontend->size<MULACOVER_CONDITIONING_API_V1_SIZE||frontend->version!=1||mulacover_get_conditioning_api(0))return 1;
    const mulacover_job_api_v1 * job=mulacover_get_job_api(1);
    if(!job||job->size<MULACOVER_JOB_API_V1_SIZE||job->version!=1||mulacover_get_job_api(0)||mulacover_get_job_api(2))return 1;
    const mulacover_api_v1 * api=mulacover_get_api(1);
    if(!api||api->size<MULACOVER_API_V1_SIZE||api->version!=1||mulacover_get_api(0)||mulacover_get_api(2))return 1;
    const mulacover_sampling_api_v1 * sampling=mulacover_get_sampling_api(1);
    if(!sampling||sampling->size<MULACOVER_SAMPLING_API_V1_SIZE||sampling->core!=api||mulacover_get_sampling_api(0)||mulacover_get_sampling_api(2))return 1;
    const mulacover_tempo_api_v1 * tempo=mulacover_get_tempo_api(1);
    if(!tempo||tempo->size<MULACOVER_TEMPO_API_V1_SIZE||tempo->core!=api||mulacover_get_tempo_api(0)||mulacover_get_tempo_api(2))return 1;
    const mulacover_continuation_api_v1 * continuation=mulacover_get_continuation_api(1);
    if(!continuation||continuation->size<MULACOVER_CONTINUATION_API_V1_SIZE||continuation->core!=api||mulacover_get_continuation_api(0)||mulacover_get_continuation_api(2))return 1;
    struct {mulacover_tempo_config_v1 config;unsigned char tail[16];} clock;
    memset(&clock,0xa5,sizeof(clock));clock.config.size=sizeof(clock);tempo->config_init(&clock.config);
    if(clock.tail[0]!=0xa5||clock.tail[15]!=0xa5||clock.config.bpm!=0||clock.config.width_beats!=1||clock.config.offset_beats!=0)return 1;
    struct {mulacover_sampling_config_v1 config;unsigned char tail[16];} sample;
    memset(&sample,0xa5,sizeof(sample));sample.config.size=sizeof(sample);sampling->config_init(&sample.config);
    if(sample.tail[0]!=0xa5||sample.config.temperature!=1||sample.config.top_k!=250||sample.config.exponential_noise)return 1;
    struct {mulacover_config_v1 config;unsigned char tail[16];} large;
    memset(&large,0xa5,sizeof(large));large.config.size=sizeof(large);api->config_init(&large.config);
    if(large.tail[0]!=0xa5||large.tail[15]!=0xa5||large.config.cache_capacity!=8192||strcmp(large.config.backend,"cpu"))return 1;
    mulacover_frame_v1 frame={0};frame.size=sizeof(frame);api->frame_init(&frame);api->frame_free(&frame);api->frame_free(&frame);
    mulacover_error_v1 e={0};e.size=sizeof(e);mulacover_context * c=0;
    if(api->create(&large.config,&c,&e)!=MULACOVER_INVALID_ARGUMENT_V1||c||!e.message[0])return 1;
    if(api->teacher_frame(0,0,0,&frame,&e)!=MULACOVER_INVALID_ARGUMENT_V1)return 1;
    api->destroy(0);return 0;
}

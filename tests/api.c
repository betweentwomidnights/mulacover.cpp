#include "mulacover.h"
#include <string.h>
int main(void) {
    const mulacover_api_v1 * api=mulacover_get_api(1);
    if(!api||api->size<MULACOVER_API_V1_SIZE||api->version!=1||mulacover_get_api(0)||mulacover_get_api(2))return 1;
    struct {mulacover_config_v1 config;unsigned char tail[16];} large;
    memset(&large,0xa5,sizeof(large));large.config.size=sizeof(large);api->config_init(&large.config);
    if(large.tail[0]!=0xa5||large.tail[15]!=0xa5||large.config.cache_capacity!=8192||strcmp(large.config.backend,"cpu"))return 1;
    mulacover_frame_v1 frame={0};frame.size=sizeof(frame);api->frame_init(&frame);api->frame_free(&frame);api->frame_free(&frame);
    mulacover_error_v1 e={0};e.size=sizeof(e);mulacover_context * c=0;
    if(api->create(&large.config,&c,&e)!=MULACOVER_INVALID_ARGUMENT_V1||c||!e.message[0])return 1;
    if(api->teacher_frame(0,0,0,&frame,&e)!=MULACOVER_INVALID_ARGUMENT_V1)return 1;
    api->destroy(0);return 0;
}

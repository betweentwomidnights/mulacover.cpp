#include "mulacover_continuation.h"
#include "continuation-time.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
static void check(bool b){if(!b){std::cerr<<"Native continuation check failed\n";std::exit(1);}}
int main() {
    const auto * api=mulacover_get_continuation_api(1);
    check(api&&api->size>=MULACOVER_CONTINUATION_API_V1_SIZE&&api->core==mulacover_get_api(1));
    check(!mulacover_get_continuation_api(0)&&!mulacover_get_continuation_api(2));
    struct {mulacover_continuation_config_v1 config;unsigned char tail[16];} storage;
    std::memset(&storage,0xa5,sizeof(storage));auto & c=storage.config;c.size=sizeof(storage);api->config_init(&c);
    check(c.new_frames==100&&c.prompt_frames==1&&c.cache_capacity==0&&storage.tail[0]==0xa5&&storage.tail[15]==0xa5);
    mulacover_error_v1 e{};e.size=sizeof(e);
    mulacover_continuation_plan_v1 p{};p.size=sizeof(p);api->plan_init(&p);
    c.source_frames=372;c.source_start_frame=50;c.guide_frames=100;c.new_frames=375;c.prompt_frames=25;
    check(!api->plan(&c,&p,&e)&&p.source_start_frame==50&&p.guide_frames==100&&p.total_frames==475&&p.cache_capacity==512);
    auto saved=p;c.cache_capacity=256;check(api->plan(&c,&p,&e)==1&&std::memcmp(&saved,&p,sizeof(p))==0);
    c.cache_capacity=0;c.guide_frames=0;check(!api->plan(&c,&p,&e)&&p.guide_frames==322&&p.cache_capacity==768);
    c.source_start_frame=372;check(api->plan(&c,&p,&e)==1);
    c.source_frames=std::numeric_limits<int64_t>::max();c.source_start_frame=c.source_frames-1;c.new_frames=1;c.prompt_frames=1;
    check(!api->plan(&c,&p,&e)&&p.guide_frames==1&&p.total_frames==2);
    c.source_start_frame=0;check(api->plan(&c,&p,&e)==1);
    c.source_frames=0;c.new_frames=8192;check(!api->plan(&c,&p,&e)&&p.cache_capacity==8192);
    c.prompt_frames=2;check(api->plan(&c,&p,&e)==1);check(api->plan(nullptr,&p,&e)==1);
    check(mula::duration_frames("7.92")==99&&mula::duration_frames("2.32")==29);
    check(mula::duration_frames("0.079999999999999999")==0&&mula::duration_frames(".080000000000000000")==1);
    check(mula::duration_frames("+8e0")==100&&mula::duration_frames("80e-1")==100&&mula::duration_frames("1e-100")==0);
    check(mula::duration_frames("737869762948382064.56")==std::numeric_limits<int64_t>::max());
    for(const char * bad:{"", ".", "-1", "NaN", "inf", "1e", "1,5", "1.2.3", "1e100", "737869762948382064.64"}) {
        bool caught=false;try{mula::duration_frames(bad);}catch(const std::invalid_argument &){caught=true;}check(caught);
    }
    const int64_t count=3*3840;std::vector<float> source(size_t(count)*2);
    for(int64_t i=0;i<count;++i){source[size_t(i)]=float(i);source[size_t(count+i)]=-float(i);}
    mulacover_continuation_audio_v1 a{};a.size=sizeof(a);api->audio_init(&a);
    check(!api->copy_audio(source.data(),count,48000,1,&a,&e)&&a.num_samples==2*3840&&a.sample_rate==48000);
    check(!std::memcmp(a.samples,source.data()+3840,size_t(a.num_samples)*4));
    check(!std::memcmp(a.samples+a.num_samples,source.data()+count+3840,size_t(a.num_samples)*4));
    const auto * occupied=a.samples;check(api->copy_audio(source.data(),count,48000,1,&a,&e)==1&&a.samples==occupied);
    source.clear();source.shrink_to_fit();check(a.samples[0]==3840&&a.samples[a.num_samples]==-3840);
    api->audio_free(&a);api->audio_free(&a);check(!a.samples&&!a.num_samples&&!a.sample_rate);
    source.assign(size_t(count)*2,0);source.back()=std::numeric_limits<float>::quiet_NaN();
    check(api->copy_audio(source.data(),count,48000,1,&a,&e)==1&&!a.samples&&!a.num_samples);
    source.back()=0;check(api->copy_audio(source.data(),count,44100,1,&a,&e)==1);
    check(api->copy_audio(source.data(),count,48000,3,&a,&e)==1);
    check(api->copy_audio(source.data(),count-1,48000,1,&a,&e)==1);
    return 0;
}

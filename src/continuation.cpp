#include "mulacover_continuation.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace {
template<class T> void init(T * p) {
    if(!p||p->size<sizeof(*p))return;const auto size=p->size;std::memset(p,0,sizeof(*p));p->size=size;
}
template<class F> int guard(mulacover_error_v1 * e,F && f) {
    int code=0;const char * message="";char detail[512]{};
    try{f();}
    catch(const std::invalid_argument & ex){code=1;std::snprintf(detail,sizeof(detail),"%s",ex.what());message=detail;}
    catch(const std::exception & ex){code=2;std::snprintf(detail,sizeof(detail),"%s",ex.what());message=detail;}
    catch(...){code=2;message="Unknown continuation error";}
    if(e&&e->size>=sizeof(*e)){e->status=code;std::snprintf(e->message,sizeof(e->message),"%s",message);}return code;
}
}
extern "C" {
void MULACOVER_CALL mulacover_continuation_config_init_v1(mulacover_continuation_config_v1 * c) {
    if(!c||c->size<sizeof(*c))return;init(c);c->new_frames=100;c->prompt_frames=1;
}
void MULACOVER_CALL mulacover_continuation_plan_init_v1(mulacover_continuation_plan_v1 * p){init(p);}
void MULACOVER_CALL mulacover_continuation_audio_init_v1(mulacover_continuation_audio_v1 * a){init(a);}
void MULACOVER_CALL mulacover_continuation_audio_free_v1(mulacover_continuation_audio_v1 * a) {
    if(!a||a->size<sizeof(*a))return;delete[]a->samples;init(a);
}
int MULACOVER_CALL mulacover_continuation_plan_build_v1(const mulacover_continuation_config_v1 * c,
        mulacover_continuation_plan_v1 * result,mulacover_error_v1 * e) {
    return guard(e,[&]{
        if(!c||c->size<sizeof(*c)||!result||result->size<sizeof(*result)||c->source_frames<0||
           c->source_start_frame<0||c->source_start_frame>c->source_frames||c->guide_frames<0||
           c->new_frames<1||c->new_frames>8192||c->prompt_frames<1||c->prompt_frames>8192||
           c->cache_capacity<0||c->cache_capacity>8192)
            throw std::invalid_argument("Invalid continuation plan configuration");
        const int64_t available=c->source_frames-c->source_start_frame;
        const int64_t guide=c->guide_frames?c->guide_frames:available;
        if(guide>available||(c->source_frames>0&&guide==0)||guide>8192-c->new_frames)
            throw std::invalid_argument("Guide selection is empty, exceeds encoded audio or exceeds model context");
        const int64_t total=guide+c->new_frames,required=c->prompt_frames+total-1;
        if(required>8192)throw std::invalid_argument("Guide and continuation exceed the 8192-position model context");
        const int capacity=c->cache_capacity?c->cache_capacity:int((required+255)/256*256);
        if(capacity<required)throw std::invalid_argument("Cache capacity is insufficient for the guide and continuation");
        const auto size=result->size;
        *result={size,c->source_start_frame,guide,c->new_frames,total,capacity};
    });
}
int MULACOVER_CALL mulacover_continuation_copy_audio_v1(const float * samples,int64_t count,int32_t rate,
        int64_t guide_frames,mulacover_continuation_audio_v1 * result,mulacover_error_v1 * e) {
    return guard(e,[&]{
        if(!samples||!result||result->size<sizeof(*result)||result->samples||result->num_samples||result->sample_rate||
           rate!=48000||count<3840||count%3840||count/3840>8192||guide_frames<0||guide_frames>=count/3840)
            throw std::invalid_argument("Invalid or occupied continuation audio arguments");
        const int64_t skip=guide_frames*3840,n=count-skip;
        auto next=std::make_unique<float[]>(size_t(n)*2);
        for(int channel=0;channel<2;++channel) {
            const float * first=samples+channel*count+skip;
            if(!std::all_of(first,first+n,[](float x){return std::isfinite(x);}))throw std::invalid_argument("Nonfinite continuation samples");
            std::copy(first,first+n,next.get()+channel*n);
        }
        result->samples=next.release();result->num_samples=n;result->sample_rate=48000;
    });
}
const mulacover_continuation_api_v1 * MULACOVER_CALL mulacover_get_continuation_api(uint32_t version) {
    static const mulacover_continuation_api_v1 api={uint32_t(MULACOVER_CONTINUATION_API_V1_SIZE),1,mulacover_get_api(1),
        mulacover_continuation_config_init_v1,mulacover_continuation_plan_init_v1,mulacover_continuation_plan_build_v1,
        mulacover_continuation_audio_init_v1,mulacover_continuation_audio_free_v1,mulacover_continuation_copy_audio_v1};
    return version==1?&api:nullptr;
}
}

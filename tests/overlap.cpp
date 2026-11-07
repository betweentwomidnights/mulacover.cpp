#include "mulacover_overlap.h"
#include <cstdio>
#include <cstring>

#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"overlap check failed at %d: %s\n",__LINE__,#x);return 1;}}while(0)
static int32_t MULACOVER_CALL cancel(void * u){++*static_cast<int *>(u);return 1;}
int main(){
    const auto * api=mulacover_get_overlap_api(1);CHECK(api&&api->size>=MULACOVER_OVERLAP_API_V1_SIZE&&api->version==1);
    CHECK(!mulacover_get_overlap_api(0)&&!mulacover_get_overlap_api(2));
    struct {mulacover_overlap_config_v1 c;uint64_t tail;} ext{};ext.c.size=sizeof(ext);ext.tail=123;api->config_init(&ext.c);CHECK(ext.tail==123&&ext.c.size==sizeof(ext)&&ext.c.overlap_frames==0);
    mulacover_overlap_config_v1 c{};c.size=sizeof(c);api->config_init(&c);
    mulacover_job_request_v1 r{};r.size=sizeof(r);mulacover_job_request_init_v1(&r);
    r.input_audio.num_samples=48000*40;r.cursor_sample=960000;r.natural_max_seconds=40;r.conditioning.prompt_frames=25;
    mulacover_overlap_plan_v1 p{};p.size=sizeof(p);api->plan_init(&p);mulacover_error_v1 e{};e.size=sizeof(e);
    CHECK(!api->plan(&r,&c,&p,&e)&&p.model_guide_frames==250&&p.overlap_frames==0&&p.generated_frames_ceiling==500);
    const auto plain=p.source;c.overlap_frames=8;
    CHECK(!api->plan(&r,&c,&p,&e)&&p.model_guide_frames==242&&p.overlap_frames==8&&p.generated_frames_ceiling==508);
    CHECK(!std::memcmp(&plain,&p.source,sizeof(plain))); // cursor, cache and NEW ceiling do not move
    c.overlap_frames=8192;CHECK(!api->plan(&r,&c,&p,&e)&&p.overlap_frames==249&&p.model_guide_frames==1);
    r.guide_samples=3840;CHECK(!api->plan(&r,&c,&p,&e)&&p.overlap_frames==0&&p.model_guide_frames==1);
    const auto before=p;c.overlap_frames=-1;CHECK(api->plan(&r,&c,&p,&e)==1&&!std::memcmp(&before,&p,sizeof(p)));
    c.overlap_frames=8193;CHECK(api->plan(&r,&c,&p,&e)==1&&!std::memcmp(&before,&p,sizeof(p)));
    c.overlap_frames=8;r.guide_samples=0;
    struct {mulacover_overlap_result_v1 r;uint64_t tail;} larger{};larger.r.size=sizeof(larger);larger.tail=456;api->result_init(&larger.r);
    CHECK(larger.tail==456&&larger.r.size==sizeof(larger)&&larger.r.job.size==sizeof(larger.r.job)&&larger.r.staging.source.size==sizeof(larger.r.staging.source));
    const auto empty=larger.r;CHECK(api->run(nullptr,&c,&larger.r,&e)==1&&!std::memcmp(&empty,&larger.r,sizeof(empty)));
    float sample=0,qwen[1024]{},muq[512]{};int32_t tokens[225]{};uint8_t mask[225]{};int polls=0;
    r.input_audio.samples=&sample;r.encoder_model=r.generator_model=r.flow_model=r.sq_model="must-not-open.gguf";
    r.conditioning.tokens=tokens;r.conditioning.tokens_mask=mask;r.conditioning.qwen=qwen;r.conditioning.muq=muq;
    r.should_cancel=cancel;r.callback_user=&polls;
    CHECK(api->run(&r,&c,&larger.r,&e)==MULACOVER_JOB_CANCELLED_V1&&polls==1&&!std::memcmp(&empty,&larger.r,sizeof(empty)));
    larger.r.job.codes=new int32_t[8];auto occupied=larger.r;
    CHECK(api->run(&r,&c,&larger.r,&e)==1&&!std::memcmp(&occupied,&larger.r,sizeof(occupied)));
    api->result_free(&larger.r);api->result_free(&larger.r);CHECK(larger.tail==456&&!std::memcmp(&empty,&larger.r,sizeof(empty)));
    std::puts("overlap negotiation, source-clock planning, clamp, cancellation and ownership passed");return 0;
}

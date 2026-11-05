#include "mulacover_job.h"
#include "../tools/wave-io.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"job check failed at line %d: %s\n",__LINE__,#x);return 1;}}while(0)
struct Observer {std::string cancel_stage;int64_t after=0;bool cancel=false;double last=0;bool ordered=true;int polls=0;};
static void MULACOVER_CALL progress(void * u,const mulacover_job_progress_v1 * p){auto & o=*static_cast<Observer *>(u);o.ordered&=p->fraction>=o.last&&p->fraction<=1;o.last=p->fraction;if(o.cancel_stage==p->stage_name&&p->completed>=o.after)o.cancel=true;}
static int32_t MULACOVER_CALL cancel(void * u){auto & o=*static_cast<Observer *>(u);++o.polls;return o.cancel?1:0;}
int main(int argc,char ** argv){
    auto * api=mulacover_get_job_api(1);CHECK(api&&api->version==1&&api->size>=MULACOVER_JOB_API_V1_SIZE&&!mulacover_get_job_api(0)&&!mulacover_get_job_api(2));
    struct {mulacover_job_request_v1 r;uint64_t tail;} ext{};ext.r.size=sizeof(ext);ext.tail=123;api->request_init(&ext.r);CHECK(ext.tail==123&&ext.r.size==sizeof(ext)&&ext.r.rng_cuda_blocks==288&&ext.r.cursor_sample==-1);
    mulacover_job_request_v1 r{};r.size=sizeof(r);api->request_init(&r);r.input_audio.num_samples=44100*70;r.input_audio.sample_rate=44100;r.input_audio.channels=2;r.input_audio.channel_stride=1;r.input_audio.sample_stride=2;r.cursor_sample=1312417;r.context_samples=1312416;r.guide_samples=44100*8;r.natural_max_seconds=90;r.conditioning.prompt_frames=25;
    mulacover_job_plan_v1 p{};p.size=sizeof(p);mulacover_error_v1 e{};e.size=sizeof(e);CHECK(!api->plan(&r,&p,&e));CHECK(p.cursor_sample==1312417&&p.context_start_sample==1&&p.context_samples==1312416&&p.context_frames==372&&p.guide_start_frame==272&&p.guide_frames==100&&p.new_frames==1125&&p.cache_capacity==1280);
    r.guide_samples=0;CHECK(!api->plan(&r,&p,&e)&&p.guide_frames==372&&p.cache_capacity==1536);
    auto before=p;r.natural_max_seconds=.079;CHECK(api->plan(&r,&p,&e)==1&&!std::memcmp(&before,&p,sizeof(p)));r.natural_max_seconds=.24;CHECK(!api->plan(&r,&p,&e)&&p.new_frames==3);
    for(int which=0;which<8;++which){auto bad=r;if(which==0)bad.cursor_sample=bad.input_audio.num_samples+1;if(which==1)bad.guide_samples=1;if(which==2)bad.context_samples=1;if(which==3)bad.natural_max_seconds=std::numeric_limits<double>::infinity();if(which==4)bad.cache_capacity=2;if(which==5)bad.input_audio.sample_stride=INT64_MAX;if(which==6)bad.input_audio.sample_rate=12345;if(which==7)bad.natural_max_seconds=655.36;CHECK(api->plan(&bad,&p,&e)==1);}
    mulacover_job_result_v1 out{};out.size=sizeof(out);api->result_init(&out);auto empty=out;CHECK(api->run(nullptr,&out,&e)==1&&!std::memcmp(&empty,&out,sizeof(out)));
    float sample=0;int32_t tokens[225]{};uint8_t mask[225]{};float qwen[1024]{},muq[512]{};r.input_audio.samples=&sample;r.encoder_model=r.generator_model=r.flow_model=r.sq_model="must-not-open.gguf";r.conditioning.tokens=tokens;r.conditioning.tokens_mask=mask;r.conditioning.qwen=qwen;r.conditioning.muq=muq;Observer observer;observer.cancel=true;r.on_progress=progress;r.should_cancel=cancel;r.callback_user=&observer;
    CHECK(api->run(&r,&out,&e)==MULACOVER_JOB_CANCELLED_V1&&!std::memcmp(&empty,&out,sizeof(out))&&observer.polls==1);
    r.rng_cuda_blocks=0;CHECK(api->run(&r,&out,&e)==1&&!std::memcmp(&empty,&out,sizeof(out)));r.rng_cuda_blocks=288;
    api->result_free(&out);api->result_free(&out);
    out.guide_frames=1;auto occupied=out;CHECK(api->run(&r,&out,&e)==1&&!std::memcmp(&occupied,&out,sizeof(out)));api->result_init(&out);
    if(argc==1)return 0;
    CHECK(argc==7); // source, preset, encoder, generator, flow, SQ (CUDA)
    auto pcm=wave_io::read_wave(argv[1]);api->request_init(&r);r.encoder_model=argv[3];r.generator_model=argv[4];r.flow_model=argv[5];r.sq_model=argv[6];r.input_audio={sizeof(r.input_audio),pcm.samples.data(),int64_t(pcm.samples.size()/pcm.channels),pcm.rate,pcm.channels,1,pcm.channels};
    // Deliberately poison every future sample: a successful job proves the PCM
    // view was narrowed before finite validation/resampling, not just token trim.
    r.cursor_sample=pcm.rate*2/25;r.natural_max_seconds=.16;r.segment_codes=93;r.decoder_steps=1;
    for(size_t i=size_t(r.cursor_sample)*pcm.channels;i<pcm.samples.size();++i)pcm.samples[i]=std::numeric_limits<float>::quiet_NaN();
    std::filesystem::path preset(argv[2]);auto t=wave_io::read<int32_t>(preset/"tokens.i32",225);auto m=wave_io::read<uint8_t>(preset/"mask.u8",225);auto q=wave_io::read<float>(preset/"qwen.f32",1024);auto mu=wave_io::read<float>(preset/"muq.f32",512);
    r.conditioning.tokens=t.data();r.conditioning.tokens_mask=m.data();r.conditioning.qwen=q.data();r.conditioning.muq=mu.data();r.conditioning.prompt_frames=25;r.conditioning.symbolic_frames=540;r.conditioning.qwen_index=14;r.conditioning.muq_index=17;r.on_progress=progress;r.should_cancel=cancel;r.callback_user=&observer;
    for(const char * stage:{"encoding","sampling","flow","waveform"}){
        observer={};observer.cancel_stage=stage;observer.after=std::string(stage)=="encoding"?0:1;
        CHECK(api->run(&r,&out,&e)==MULACOVER_JOB_CANCELLED_V1&&!std::memcmp(&empty,&out,sizeof(out))&&observer.ordered);
        std::printf("cancelled %s without publishing output\n",stage);
    }
    std::vector<int32_t> reference_codes;std::vector<float> reference_audio;
    std::vector<float> piano(540*256,-1),drum(540*128,-1),chord(540*12,-1);std::vector<uint8_t> padding(540,0);
    for(int repeat=0;repeat<2;++repeat){observer={};
        // Supplied dropped-score tensors and the built-in none mode must be
        // equivalent, while retaining both public conditioning paths.
        if(repeat){r.score_mode=MULACOVER_JOB_SCORE_PROVIDED_V1;r.conditioning.pianoroll=piano.data();r.conditioning.drum_pianoroll=drum.data();r.conditioning.chord=chord.data();r.conditioning.context_mask=padding.data();}
        CHECK(!api->run(&r,&out,&e)&&observer.ordered&&observer.last==1&&out.guide_frames==1&&out.new_frames==2&&out.num_samples==7680&&!out.full_samples&&out.sampler_final_offset==64);
        if(!repeat){reference_codes.assign(out.codes,out.codes+24);reference_audio.assign(out.samples,out.samples+15360);}else{CHECK(!std::memcmp(reference_codes.data(),out.codes,96)&&!std::memcmp(reference_audio.data(),out.samples,15360*4));}
        api->result_free(&out);CHECK(!std::memcmp(&empty,&out,sizeof(out)));
    }
    std::puts("past-only poisoned-future, repeated-job replay and cancellation checks passed");return 0;
}

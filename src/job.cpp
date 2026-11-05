#include "mulacover_job.h"
#include "sampling.h"
#include "heartcodec_work.h"
#include "heartcodec_rng.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Cancelled {};
template<class T> void init(T * p){if(!p||p->size<sizeof(*p))return;const auto size=p->size;std::memset(p,0,sizeof(*p));p->size=size;}
int status(mulacover_error_v1 * e,int code,const char * message){if(e&&e->size>=sizeof(*e)){e->status=code;std::snprintf(e->message,sizeof(e->message),"%s",message);}return code;}
template<class F> int guard(mulacover_error_v1 * e,F && f){try{f();return status(e,0,"");}catch(const Cancelled &){return status(e,MULACOVER_JOB_CANCELLED_V1,"Continuation job cancelled");}catch(const std::invalid_argument & x){return status(e,1,x.what());}catch(const std::exception & x){return status(e,2,x.what());}catch(...){return status(e,2,"Unknown continuation job failure");}}
bool supported_rate(int rate){for(int r:{16000,22050,24000,32000,44100,48000,88200,96000,192000})if(rate==r)return true;return false;}
void require_path(const char * path){if(!path||!*path)throw std::invalid_argument("All four model paths are required");}
template<class T> void finite(const T * data,int64_t count){if(!data||!std::all_of(data,data+count,[](T x){return std::isfinite(x);}))throw std::invalid_argument("Missing/nonfinite prepared conditioning");}
void plan(const mulacover_job_request_v1 & r,mulacover_job_plan_v1 & out) {
    const auto & a=r.input_audio;
    if(r.size<sizeof(r)||a.size<sizeof(a)||a.num_samples<1||!supported_rate(a.sample_rate)||(a.channels!=1&&a.channels!=2)||a.channel_stride<1||a.sample_stride<1||
       r.cursor_sample < -1||r.cursor_sample>a.num_samples||r.context_samples<0||r.guide_samples<0||
       !std::isfinite(r.natural_max_seconds)||r.natural_max_seconds<=0||r.natural_max_seconds>655.36||r.conditioning.size<sizeof(r.conditioning))
        throw std::invalid_argument("Invalid continuation audio geometry, duration or condition struct");
    const int64_t cursor=r.cursor_sample<0?a.num_samples:r.cursor_sample;
    const int64_t unit=int64_t(a.sample_rate)*2/25;
    const int64_t available=r.context_samples?std::min(cursor,r.context_samples):cursor;
    const int64_t count=available/unit*unit;
    if(count<unit||count>INT32_MAX)throw std::invalid_argument("Past context must contain a complete 80ms frame and fit encoder input limits");
    const int64_t context_frames=count/unit;
    const int64_t guide=r.guide_samples?std::min(count,r.guide_samples)/unit:context_frames;
    if(guide<1)throw std::invalid_argument("Guide must contain a complete 80ms frame");
    const int64_t maximum=int64_t(std::floor(r.natural_max_seconds*12.5));
    mulacover_continuation_config_v1 c{};c.size=sizeof(c);mulacover_continuation_config_init_v1(&c);
    c.source_frames=context_frames;c.source_start_frame=context_frames-guide;c.guide_frames=guide;c.new_frames=maximum;
    c.prompt_frames=r.conditioning.prompt_frames;c.cache_capacity=r.cache_capacity;
    mulacover_continuation_plan_v1 p{};p.size=sizeof(p);mulacover_error_v1 e{};e.size=sizeof(e);
    if(mulacover_continuation_plan_build_v1(&c,&p,&e))throw std::invalid_argument(e.message);
    const int64_t limit=int64_t(std::numeric_limits<ptrdiff_t>::max()/sizeof(float));
    if(cursor-1>limit/a.sample_stride||(a.channels-1)>limit/a.channel_stride||
       (a.channels-1)*a.channel_stride>limit-(cursor-1)*a.sample_stride)throw std::invalid_argument("PCM address range overflows");
    out={out.size,cursor,cursor-count,count,context_frames,context_frames-guide,guide,maximum,p.cache_capacity};
}
struct Observer {
    const mulacover_job_request_v1 & request;
    bool cancelled=false;
    double last=0;
    bool poll(){if(!cancelled&&request.should_cancel)cancelled=request.should_cancel(request.callback_user)!=0;return cancelled;}
    void check(){if(poll())throw Cancelled{};}
    void progress(int stage,const char * name,int64_t completed,int64_t total,double fraction){
        check();fraction=std::max(last,std::min(1.0,fraction));last=fraction;
        if(request.on_progress){mulacover_job_progress_v1 p{sizeof(p),stage,name,completed,total,fraction};request.on_progress(request.callback_user,&p);}check();
    }
    static int32_t HEARTCODEC_CALL cancel(void * p){return static_cast<Observer *>(p)->poll()?1:0;}
    static void HEARTCODEC_CALL codec_progress(void * user,const heartcodec_work_progress_v1 * p){
        auto & o=*static_cast<Observer *>(user);const double fraction=p->total?double(p->completed)/double(p->total):0;
        // Do not throw across the codec callback. Its own post-callback poll
        // observes cancellation and returns the negotiated cancellation status.
        const bool encoding=p->stage==HEARTCODEC_WORK_ENCODING_V1;
        const double value=encoding?.1*fraction:p->stage==HEARTCODEC_WORK_FLOW_V1?.9+.07*fraction:.97+.03*fraction;
        o.last=std::max(o.last,std::min(1.0,value));
        if(o.request.on_progress){mulacover_job_progress_v1 tick{sizeof(tick),encoding?MULACOVER_JOB_ENCODING_V1:MULACOVER_JOB_DECODING_V1,
            encoding?"encoding":p->stage==HEARTCODEC_WORK_FLOW_V1?"flow":"waveform",p->completed,p->total,o.last};o.request.on_progress(o.request.callback_user,&tick);}
    }
};
void codec_status(int code,const heartcodec_error_v1 & e){if(code==HEARTCODEC_WORK_CANCELLED_V1)throw Cancelled{};if(code)throw std::runtime_error(e.message);}
struct Entropy {heartcodec_rng_result_v1 value{};Entropy(){value.size=sizeof(value);heartcodec_rng_result_init_v1(&value);}~Entropy(){heartcodec_rng_result_free_v1(&value);}void clear(){heartcodec_rng_result_free_v1(&value);}};
struct Encoded {heartcodec_encoded_tokens_v1 value{};Encoded(){value.size=sizeof(value);heartcodec_encoded_tokens_init_v1(&value);}~Encoded(){heartcodec_encoded_tokens_free_v1(&value);}};
struct Stereo {heartcodec_stereo_v1 value{};Stereo(){value.size=sizeof(value);heartcodec_stereo_init_v1(&value);}~Stereo(){heartcodec_stereo_free_v1(&value);}};
struct Frame {mulacover_frame_v1 value{};Frame(){value.size=sizeof(value);mulacover_frame_init_v1(&value);}~Frame(){mulacover_frame_free_v1(&value);}void clear(){mulacover_frame_free_v1(&value);}};
double elapsed(std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
}
extern "C" {
void MULACOVER_CALL mulacover_job_request_init_v1(mulacover_job_request_v1 * r){
    if(!r||r->size<sizeof(*r))return;init(r);r->backend="cuda";r->cpu_threads=4;r->cursor_sample=-1;r->natural_max_seconds=30;
    r->input_audio={sizeof(r->input_audio),nullptr,0,48000,1,1,1};r->conditioning.size=sizeof(r->conditioning);r->conditioning.prompt_frames=2;r->conditioning.symbolic_frames=1;r->conditioning.guidance=1.5f;r->conditioning.muq_index=1;
    r->sampling.size=sizeof(r->sampling);mulacover_sampling_config_init_v1(&r->sampling);r->rng_mode=MULACOVER_JOB_RNG_TORCH_CUDA_V1;r->rng_cuda_blocks=288;r->decoder_seed=1701;r->decoder_steps=10;r->segment_codes=372;
}
void MULACOVER_CALL mulacover_job_plan_init_v1(mulacover_job_plan_v1 * p){init(p);}
void MULACOVER_CALL mulacover_job_result_init_v1(mulacover_job_result_v1 * r){init(r);}
void MULACOVER_CALL mulacover_job_result_free_v1(mulacover_job_result_v1 * r){if(!r||r->size<sizeof(*r))return;delete[]r->samples;delete[]r->full_samples;delete[]r->codes;init(r);}
int MULACOVER_CALL mulacover_job_plan_build_v1(const mulacover_job_request_v1 * r,mulacover_job_plan_v1 * p,mulacover_error_v1 * e){
    return guard(e,[&]{if(!r||!p||p->size<sizeof(*p))throw std::invalid_argument("Invalid job plan arguments");mulacover_job_plan_v1 next{};next.size=p->size;plan(*r,next);*p=next;});
}
int MULACOVER_CALL mulacover_job_run_v1(const mulacover_job_request_v1 * request,mulacover_job_result_v1 * output,mulacover_error_v1 * error){
    return guard(error,[&]{
        if(!request||!output||output->size<sizeof(*output)||output->samples||output->full_samples||output->codes||output->num_samples||output->full_num_samples||output->sample_rate||output->guide_frames||output->new_frames)
            throw std::invalid_argument("Invalid job request or occupied result");
        const auto & r=*request;mulacover_job_plan_v1 p{};p.size=sizeof(p);plan(r,p);
        require_path(r.encoder_model);require_path(r.generator_model);require_path(r.flow_model);require_path(r.sq_model);
        if(!r.input_audio.samples||!r.backend||(std::strcmp(r.backend,"cpu")&&std::strcmp(r.backend,"cuda"))||r.cpu_threads<1||r.cpu_threads>1024||
           (r.score_mode!=MULACOVER_JOB_SCORE_NONE_V1&&r.score_mode!=MULACOVER_JOB_SCORE_PROVIDED_V1)||(r.rng_mode!=0&&r.rng_mode!=1)||r.rng_device<0||r.rng_cuda_blocks>1048576||(r.rng_mode==1&&!r.rng_cuda_blocks)||
           r.sampler_offset%4||(r.rng_mode==0&&r.sampler_offset)||r.decoder_steps<1||r.decoder_steps>1000||r.segment_codes<93||r.segment_codes%93||r.segment_codes>4092||
           (r.include_full_audio!=0&&r.include_full_audio!=1)||r.sampling.exponential_noise||r.sampling.noise_count)
            throw std::invalid_argument("Invalid continuation job configuration");
        mula::validate_sampling(r.sampling);
        auto condition=r.conditioning;
        if(!condition.tokens||!condition.tokens_mask||condition.symbolic_frames<1||condition.symbolic_frames>5000||condition.qwen_index<0||condition.qwen_index>=condition.prompt_frames||
           condition.muq_index<0||condition.muq_index>=condition.prompt_frames||condition.qwen_index==condition.muq_index||!std::isfinite(condition.guidance)||condition.guidance<=0)
            throw std::invalid_argument("Invalid prepared preset dimensions");
        finite(condition.qwen,1024);finite(condition.muq,512);
        for(int64_t i=0;i<condition.prompt_frames*9;++i)if(condition.tokens[i]<0||condition.tokens[i]>=(i%9==8?128256:8197)||condition.tokens_mask[i]>1)throw std::invalid_argument("Invalid prepared prompt tokens/masks");
        std::vector<float> piano,drum,chord;std::vector<uint8_t> mask;
        if(r.score_mode==MULACOVER_JOB_SCORE_NONE_V1){piano.assign(size_t(condition.symbolic_frames)*256,-1);drum.assign(size_t(condition.symbolic_frames)*128,-1);chord.assign(size_t(condition.symbolic_frames)*12,-1);mask.assign(size_t(condition.symbolic_frames),0);
            condition.pianoroll=piano.data();condition.drum_pianoroll=drum.data();condition.chord=chord.data();condition.context_mask=mask.data();}
        else {finite(condition.pianoroll,condition.symbolic_frames*256);finite(condition.drum_pianoroll,condition.symbolic_frames*128);finite(condition.chord,condition.symbolic_frames*12);
            if(!condition.context_mask||!std::all_of(condition.context_mask,condition.context_mask+condition.symbolic_frames,[](uint8_t x){return x<=1;})||std::all_of(condition.context_mask,condition.context_mask+condition.symbolic_frames,[](uint8_t x){return x!=0;}))throw std::invalid_argument("Invalid symbolic padding mask");}
        Observer observer{r};observer.progress(MULACOVER_JOB_LOADING_ENCODER_V1,"loading encoder",0,0,0);
        const auto * work=heartcodec_get_work_api(1);const auto * rng=heartcodec_get_rng_api(1);
        if(!work||work->size<HEARTCODEC_WORK_API_V1_SIZE||!rng||rng->size<HEARTCODEC_RNG_API_V1_SIZE)throw std::runtime_error("Required HeartCodec work/RNG capability unavailable");
        heartcodec_error_v1 ce{};ce.size=sizeof(ce);Entropy entropy;
        uint64_t offset=r.sampler_offset;
        auto next_noise=[&]{heartcodec_rng_config_v1 c{};c.size=sizeof(c);rng->config_init(&c);c.distribution=HEARTCODEC_RNG_EXPONENTIAL_V1;c.seed=r.sampling.seed;c.offset=offset;c.count=8197;c.draws=8;c.max_blocks=r.rng_cuda_blocks;c.device=r.rng_device;
            codec_status(rng->generate(&c,&entropy.value,&ce),ce);};
        if(r.rng_mode==MULACOVER_JOB_RNG_TORCH_CUDA_V1){next_noise();observer.check();}
        heartcodec_work_control_v1 control{};control.size=sizeof(control);work->control_init(&control);control.should_cancel=Observer::cancel;control.on_progress=Observer::codec_progress;control.user=&observer;
        heartcodec_config_v1 hc{};hc.size=sizeof(hc);heartcodec_config_init_v1(&hc);hc.model_path=r.encoder_model;hc.backend=r.backend;hc.cpu_threads=r.cpu_threads;
        heartcodec_encoder_context * ec=nullptr;codec_status(heartcodec_encoder_context_create_v1(&hc,&ec,&ce),ce);
        std::unique_ptr<heartcodec_encoder_context,decltype(&heartcodec_encoder_context_destroy_v1)> encoder(ec,heartcodec_encoder_context_destroy_v1);
        const auto encode_start=std::chrono::steady_clock::now();observer.progress(MULACOVER_JOB_ENCODING_V1,"encoding",0,p.context_frames,0);
        const auto & a=r.input_audio;heartcodec_waveform_v1 view{sizeof(view),a.samples+p.context_start_sample*a.sample_stride,p.context_samples,a.sample_rate,a.channels,a.channel_stride,a.sample_stride};
        Encoded encoded;codec_status(work->encode(ec,&view,nullptr,&control,&encoded.value,&ce),ce);
        if(encoded.value.frames!=p.context_frames)throw std::runtime_error("Encoder framing did not match the job plan");
        heartcodec_memory_v1 em{};em.size=sizeof(em);codec_status(heartcodec_encoder_get_memory_v1(ec,&em,&ce),ce);encoder.reset();
        const double encode_seconds=elapsed(encode_start);
        observer.progress(MULACOVER_JOB_LOADING_GENERATOR_V1,"loading generator",0,0,.1);
        mulacover_config_v1 gc{};gc.size=sizeof(gc);mulacover_config_init_v1(&gc);gc.model_path=r.generator_model;gc.backend=r.backend;gc.threads=r.cpu_threads;gc.cache_capacity=p.cache_capacity;
        mulacover_context * raw_generator=nullptr;mulacover_error_v1 ge{};ge.size=sizeof(ge);
        if(mulacover_create_v1(&gc,&raw_generator,&ge))throw std::runtime_error(ge.message);
        std::unique_ptr<mulacover_context,decltype(&mulacover_destroy_v1)> generator(raw_generator,mulacover_destroy_v1);
        observer.check();if(mulacover_session_begin_v1(generator.get(),&condition,&ge))throw std::runtime_error(ge.message);
        const auto generate_start=std::chrono::steady_clock::now();Frame frame;std::vector<int32_t> codes;codes.reserve(size_t(p.guide_frames+p.new_frames)*8);
        auto sampling=r.sampling;int32_t stop=0;const int64_t total=p.guide_frames+p.new_frames;
        for(int64_t f=0;f<total;++f){
            const bool priming=f<p.guide_frames;observer.progress(priming?MULACOVER_JOB_PRIMING_V1:MULACOVER_JOB_SAMPLING_V1,priming?"priming":"sampling",priming?f:f-p.guide_frames,priming?p.guide_frames:p.new_frames,.1+.8*double(f)/double(total));
            if(priming){if(mulacover_teacher_frame_v1(generator.get(),encoded.value.data+(p.guide_start_frame+f)*8,nullptr,&frame.value,&ge))throw std::runtime_error(ge.message);}
            else {
                if(r.rng_mode==MULACOVER_JOB_RNG_TORCH_CUDA_V1){if(!entropy.value.data)next_noise();sampling.exponential_noise=entropy.value.data;sampling.noise_count=8*8197;}
                if(mulacover_sample_frame_v1(generator.get(),&sampling,nullptr,&frame.value,&stop,&ge))throw std::runtime_error(ge.message);
                if(r.rng_mode==MULACOVER_JOB_RNG_TORCH_CUDA_V1){offset=entropy.value.next_offset;entropy.clear();sampling.exponential_noise=nullptr;sampling.noise_count=0;}
            }
            observer.check();if(stop==MULACOVER_PADDING_V1)throw std::runtime_error("Generator emitted padding");
            if(stop==MULACOVER_EOS_V1){frame.clear();break;}
            codes.insert(codes.end(),frame.value.codes,frame.value.codes+8);frame.clear();
        }
        if(int64_t(codes.size()/8)<=p.guide_frames)throw std::runtime_error("EOS before any new continuation audio");
        mulacover_memory_v1 gm{};gm.size=sizeof(gm);if(mulacover_get_memory_v1(generator.get(),&gm,&ge))throw std::runtime_error(ge.message);
        generator.reset();heartcodec_encoded_tokens_free_v1(&encoded.value);entropy.clear();const double generate_seconds=elapsed(generate_start);
        observer.progress(MULACOVER_JOB_LOADING_DECODER_V1,"loading decoder",0,0,.9);
        hc.model_path=r.flow_model;heartcodec_context * dc=nullptr;codec_status(heartcodec_context_create_v1(&hc,&dc,&ce),ce);
        std::unique_ptr<heartcodec_context,decltype(&heartcodec_context_destroy_v1)> decoder(dc,heartcodec_context_destroy_v1);
        codec_status(heartcodec_component_set_v1(dc,HEARTCODEC_SQ_V1,r.sq_model,&ce),ce);observer.check();
        heartcodec_flow_config_v1 flow{};flow.size=sizeof(flow);heartcodec_flow_config_init_v1(&flow);flow.seed=r.decoder_seed;flow.steps=r.decoder_steps;flow.segment_codes=r.segment_codes;
        uint64_t decode_offset=0;int64_t rng_windows=0;
        if(r.rng_mode==MULACOVER_JOB_RNG_TORCH_CUDA_V1){
            const int64_t hop=r.segment_codes/93*80,overlap=r.segment_codes-hop;const int64_t padded=(std::max(r.segment_codes,int64_t(codes.size()/8))-overlap+hop-1)/hop*hop+overlap;
            heartcodec_rng_config_v1 c{};c.size=sizeof(c);rng->config_init(&c);c.seed=r.decoder_seed;c.count=r.segment_codes*2*256;c.draws=(padded-r.segment_codes)/hop+1;c.max_blocks=r.rng_cuda_blocks;c.device=r.rng_device;
            codec_status(rng->generate(&c,&entropy.value,&ce),ce);flow.noise=entropy.value.data;flow.noise_segments=entropy.value.draws;decode_offset=entropy.value.next_offset;rng_windows=entropy.value.draws;
        }
        const auto decode_start=std::chrono::steady_clock::now();Stereo audio;heartcodec_tokens_v1 tv{sizeof(tv),codes.data(),int64_t(codes.size()/8),1,8};
        observer.progress(MULACOVER_JOB_DECODING_V1,"decoding",0,0,.9);codec_status(work->tokens_decode(dc,&tv,&flow,&control,&audio.value,&ce),ce);decoder.reset();entropy.clear();
        if(audio.value.sample_rate!=48000||audio.value.num_samples!=int64_t(codes.size()/8)*3840)throw std::runtime_error("Unexpected decoded clock");
        const int64_t skip=p.guide_frames*3840,n=audio.value.num_samples-skip;auto suffix=std::make_unique<float[]>(size_t(n)*2);
        for(int ch=0;ch<2;++ch)std::copy_n(audio.value.samples+ch*audio.value.num_samples+skip,size_t(n),suffix.get()+ch*n);
        std::unique_ptr<float[]> full;if(r.include_full_audio){full=std::make_unique<float[]>(size_t(audio.value.num_samples)*2);std::copy_n(audio.value.samples,size_t(audio.value.num_samples)*2,full.get());}
        auto owned_codes=std::make_unique<int32_t[]>(codes.size());std::copy(codes.begin(),codes.end(),owned_codes.get());
        const double decode_seconds=elapsed(decode_start);observer.progress(MULACOVER_JOB_DONE_V1,"done",1,1,1);
        mulacover_job_result_v1 result{};result.size=output->size;result.samples=suffix.release();result.num_samples=n;result.sample_rate=48000;result.full_samples=full.release();result.full_num_samples=r.include_full_audio?audio.value.num_samples:0;
        result.codes=owned_codes.release();result.guide_frames=p.guide_frames;result.new_frames=int64_t(codes.size()/8)-p.guide_frames;result.stop_reason=stop;result.plan=p;
        result.sampler_final_offset=offset;result.decoder_final_offset=decode_offset;result.decoder_rng_windows=rng_windows;result.encode_seconds=encode_seconds;result.generate_seconds=generate_seconds;result.decode_seconds=decode_seconds;
        result.encoder_weight_bytes=em.weight_buffer_bytes;result.encoder_graph_bytes=em.graph_buffer_bytes;result.generator_weight_bytes=gm.weight_bytes;result.generator_cache_bytes=gm.cache_bytes;result.generator_graph_bytes=gm.graph_bytes;
        *output=result;
    });
}
const mulacover_job_api_v1 * MULACOVER_CALL mulacover_get_job_api(uint32_t version){
    static const mulacover_job_api_v1 api={MULACOVER_JOB_API_V1_SIZE,1,mulacover_job_request_init_v1,mulacover_job_plan_init_v1,mulacover_job_result_init_v1,mulacover_job_result_free_v1,mulacover_job_plan_build_v1,mulacover_job_run_v1};return version==1?&api:nullptr;
}
}

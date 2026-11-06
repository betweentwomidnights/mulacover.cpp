#include "service.h"
#include "multipart.h"
#include "transcription.h"
#include "mulacover_conditioning.h"
#include "mulacover_runtime.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>

#ifndef MULACOVER_VERSION
#define MULACOVER_VERSION "unknown"
#endif
#ifndef MULACOVER_SOURCE_COMMIT
#define MULACOVER_SOURCE_COMMIT "unknown"
#endif
namespace mulacover::server {
namespace {
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
struct Cancelled {};
std::string id() {
    std::random_device random; std::ostringstream out;
    out<<std::hex;
    for(int i=0;i<4;++i){out.width(8);out.fill('0');out<<uint32_t(random());}
    return out.str();
}
int64_t integer(const json::Value & v,const char * key,int64_t fallback,int64_t low,int64_t high) {
    const auto * p=v.find(key);
    if(!p)return fallback;
    if(p->type!=json::Type::number||!json::is_integer(*p))throw std::invalid_argument(std::string(key)+" must be an integer");
    size_t used=0;int64_t n;
    try{n=std::stoll(p->text,&used);}catch(...){throw std::invalid_argument(std::string(key)+" is outside the integer range");}
    if(used!=p->text.size()||n<low||n>high)throw std::invalid_argument(std::string(key)+" is outside the allowed range");
    return n;
}
double real(const json::Value & v,const char * key,double fallback,double low,double high) {
    double n=json::number(v,key,fallback);
    if(n<low||n>high)throw std::invalid_argument(std::string(key)+" is outside the allowed range");
    return n;
}
uint64_t seed(const json::Value & v,const char * key,uint64_t fallback) {
    const auto * p=v.find(key);
    if(!p)return fallback;
    if((p->type!=json::Type::number&&p->type!=json::Type::string)||p->text.empty()||
       p->text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument(std::string(key)+" must be an unsigned decimal integer (use a string for 64-bit seeds)");
    return json::u64(v,key,fallback);
}
mulacover_job_request_v1 request(const Config & c,const Input & in) {
    mulacover_job_request_v1 r{};r.size=sizeof(r);mulacover_job_request_init_v1(&r);
    r.encoder_model=c.encoder.c_str();r.generator_model=c.generator.c_str();r.flow_model=c.flow.c_str();r.sq_model=c.sq.c_str();
    r.backend=c.backend.c_str();r.cpu_threads=c.threads;
    const auto & a=in.audio;const auto & v=in.options;
    r.input_audio={sizeof(r.input_audio),a.samples.data(),int64_t(a.samples.size()/a.channels),a.rate,a.channels,1,a.channels};
    r.cursor_sample=integer(v,"cursor_sample",-1,-1,r.input_audio.num_samples);
    r.context_samples=integer(v,"context_samples",0,0,INT64_MAX);
    r.guide_samples=integer(v,"guide_samples",0,0,INT64_MAX);
    if(v.find("audio_context_seconds")){
        if(v.find("cursor_sample")||v.find("context_samples")||v.find("guide_samples"))throw std::invalid_argument("Use audio_context_seconds or explicit cursor/context/guide, not both");
        const double seconds=real(v,"audio_context_seconds",29.76,.08,86400);
        const int64_t frame=int64_t(a.rate)*2/25;if(frame<=0)throw std::invalid_argument("Invalid source sample rate");
        const int64_t frames=std::min(r.input_audio.num_samples/frame,int64_t(std::floor(seconds*12.5)));
        r.cursor_sample=r.context_samples=r.guide_samples=frames*frame;
    }
    r.natural_max_seconds=real(v,"natural_max_seconds",30,.08,655.36);
    r.cache_capacity=int(integer(v,"cache_capacity",0,0,c.max_cache_capacity));
    r.conditioning.guidance=float(real(v,"guidance",1.5,.001,20));
    r.sampling.seed=seed(v,"seed",1701);
    r.sampling.top_k=int(integer(v,"top_k",250,1,8197));
    r.sampling.temperature=float(real(v,"temperature",1,.001,10));
    r.decoder_seed=seed(v,"decoder_seed",1701);
    r.decoder_steps=int(integer(v,"decoder_steps",10,1,1000));
    r.segment_codes=integer(v,"segment_codes",372,93,4092);
    if(r.segment_codes%93)throw std::invalid_argument("segment_codes must be a multiple of 93");
    const auto rng=json::string(v,"rng",c.backend=="cuda"?"torch-cuda":"native");
    if(rng!="torch-cuda"&&rng!="native")throw std::invalid_argument("rng must be torch-cuda or native");
    if(rng=="torch-cuda"&&c.backend!="cuda")throw std::invalid_argument("Torch CUDA RNG requires the CUDA backend");
    r.rng_mode=rng=="torch-cuda"?1:0;
    r.rng_cuda_blocks=uint32_t(integer(v,"rng_cuda_blocks",288,1,1048576));
    const auto score=json::string(v,"score_mode","none");
    if(score!="none"&&score!="provided")throw std::invalid_argument("score_mode must be none or provided");
    r.score_mode=score=="provided"?1:0;
    const bool transcribe=json::boolean(v,"transcribe",false);
    if(transcribe){
        if(!in.melody.empty()||!in.chords.empty()||!in.drums.empty())throw std::invalid_argument("Automatic transcription cannot use MIDI uploads");
        if(!c.transcription_port)throw std::invalid_argument("SheetSage2 service is not configured");
        r.score_mode=1;
    }
    if(r.score_mode&&in.melody.empty()&&!transcribe)throw std::invalid_argument("Score-guided continuation needs melody_midi");
    if(!r.score_mode&&(!in.melody.empty()||!in.chords.empty()||!in.drums.empty()))throw std::invalid_argument("MIDI uploads require score_mode provided");
    r.include_full_audio=json::boolean(v,"include_full_audio",false)?1:0;
    return r;
}
mulacover_job_plan_v1 plan(const Config & c,const mulacover_job_request_v1 & r) {
    mulacover_job_plan_v1 p{};p.size=sizeof(p);mulacover_error_v1 e{};e.size=sizeof(e);
    if(mulacover_job_plan_build_v1(&r,&p,&e))throw std::invalid_argument(e.message);
    if(p.cache_capacity>c.max_cache_capacity)throw std::invalid_argument("Context plus new audio exceeds the server cache budget; shorten the guide or duration");
    return p;
}
Input parse_input(const HttpRequest & req,const Config & config) {
    const auto ct=req.headers.find("content-type");
    const auto boundary=multipart_boundary(ct==req.headers.end()?"":ct->second);
    if(!boundary)throw std::invalid_argument("Expected multipart/form-data with audio and options parts");
    Input in;in.options=json::object();bool audio=false;std::set<std::string> seen;
    for(auto & part:parse_multipart(req.body,*boundary)) {
        if(!seen.insert(part.name).second)throw std::invalid_argument("Duplicate multipart part");
        if(part.name=="audio"){
            try{in.audio=wave_io::read_wave_bytes(part.data);}catch(const std::runtime_error & e){throw std::invalid_argument(e.what());}
            audio=true;
        }
        else if(part.name=="options"){
            if(part.data.size()>65536)throw std::invalid_argument("Options exceed 64 KiB");
            try{in.options=json::parse(part.data);}catch(const std::runtime_error & e){throw std::invalid_argument(e.what());}
        }else if(part.name=="melody_midi")in.melody=std::move(part.data);
        else if(part.name=="chord_midi")in.chords=std::move(part.data);
        else if(part.name=="drum_midi")in.drums=std::move(part.data);
        else throw std::invalid_argument("Unknown multipart part: "+part.name);
    }
    if(!audio)throw std::invalid_argument("Missing audio WAV upload");
    if(in.options.type!=json::Type::object)throw std::invalid_argument("options must be a JSON object");
    const std::set<std::string> keys={"lyrics","tags","audio_context_seconds","cursor_sample","context_samples","guide_samples","natural_max_seconds","cache_capacity","guidance","seed","top_k","temperature","decoder_seed","decoder_steps","segment_codes","rng","rng_cuda_blocks","score_mode","symbolic_frames","include_full_audio","transcribe","transcription_lane","transcription_scope"};
    for(const auto & p:in.options.object)if(!keys.count(p.first))throw std::invalid_argument("Unknown option: "+p.first);
    for(const char * key:{"lyrics","tags"}){
        auto text=json::string(in.options,key,"");
        if(text.size()>16384||text.find('\0')!=std::string::npos)throw std::invalid_argument(std::string(key)+" exceeds 16 KiB or contains NUL");
    }
    integer(in.options,"symbolic_frames",0,!json::boolean(in.options,"transcribe",false)&&json::string(in.options,"score_mode","none")=="none"?1:0,5000);
    const auto lane=json::string(in.options,"transcription_lane","combined");if(lane!="combined"&&lane!="vocal"&&lane!="instrumental")throw std::invalid_argument("Invalid transcription lane");
    const auto scope=json::string(in.options,"transcription_scope","context");if(scope!="context"&&scope!="source")throw std::invalid_argument("Invalid transcription scope");
    for(const auto * midi:{&in.melody,&in.chords,&in.drums}){
        if(midi->size()>4*1024*1024)throw std::invalid_argument("MIDI exceeds 4 MiB");
        if(!midi->empty()&&(midi->size()<14||midi->compare(0,4,"MThd")))throw std::invalid_argument("Expected standard MIDI upload");
    }
    plan_audio_input(config,in); // validate before accepting work, no models/RNG
    return in;
}
std::string wav(const float * samples,int64_t count,int rate) {
    if(count<0||uint64_t(count)>(UINT32_MAX-36)/8)throw std::runtime_error("Invalid output WAV length");
    std::string out;out.reserve(44+size_t(count)*8);
    auto u32=[&](uint32_t n){for(int i=0;i<4;++i)out.push_back(char(n>>(8*i)));};
    auto u16=[&](uint16_t n){out.push_back(char(n));out.push_back(char(n>>8));};
    out+="RIFF";u32(uint32_t(36+count*8));out+="WAVEfmt ";u32(16);u16(3);u16(2);u32(rate);u32(rate*8);u16(8);u16(32);out+="data";u32(uint32_t(count*8));
    for(int64_t i=0;i<count;++i)for(int channel=0;channel<2;++channel){uint32_t bits;std::memcpy(&bits,samples+i+channel*count,4);u32(bits);}
    return out;
}
struct Scratch {
    fs::path root;
    Scratch(){root=fs::temp_directory_path()/ ("mulacover-"+id());if(!fs::create_directory(root))throw std::runtime_error("Cannot create MIDI staging directory");}
    ~Scratch(){std::error_code ec;fs::remove_all(root,ec);}
    std::string midi(const char * name,const std::string & bytes){
        if(bytes.empty())return {};
        const auto path=root/name;std::ofstream f(path,std::ios::binary);
        if(!f.write(bytes.data(),std::streamsize(bytes.size())))throw std::runtime_error("Cannot stage MIDI upload");
        return path.string();
    }
};
struct Callbacks {
    const Progress & progress;const Cancel & cancel;
    bool failed=false;
    void emit(double fraction,const char * stage,int64_t n,int64_t total)noexcept{
        try{progress(fraction,stage,n,total);}catch(...){failed=true;}
    }
    static int32_t MULACOVER_CALL poll(void * u)noexcept{
        auto & c=*static_cast<Callbacks *>(u);try{return c.failed||c.cancel()?1:0;}catch(...){c.failed=true;return 1;}
    }
    static void MULACOVER_CALL style(void * u,int64_t n,int64_t total)noexcept{static_cast<Callbacks *>(u)->emit(total?.1*double(n)/total:0,"preparing style",n,total);}
    static void MULACOVER_CALL job(void * u,const mulacover_job_progress_v1 * p)noexcept{static_cast<Callbacks *>(u)->emit(.1+.9*p->fraction,p->stage_name,p->completed,p->total);}
};
}
void Config::resolve_models() {
    auto model=[&](const std::string & dir,const char * name){return fs::absolute(fs::u8path(dir)/name).lexically_normal().u8string();};
    tokenizers=model(model_dir,"mulacover-conditioning-tokenizers.gguf");style=model(model_dir,"mulacover-style-f32.gguf");generator=model(model_dir,"mulacover-q4_k_m.gguf");
    encoder=model(codec_model_dir,"heartcodec-encoder-f32-v2.gguf");flow=model(codec_model_dir,"heartcodec-flow-f32.gguf");sq=model(codec_model_dir,"heartcodec-sq-codec-f32.gguf");
}
mulacover_job_plan_v1 plan_audio_input(const Config & c,const Input & in){return plan(c,request(c,in));}
std::string props(const Config & c) {
    auto v=json::object();auto & o=v.object;
    o["service"]=json::from_string("mulacover");o["version"]=json::from_string(MULACOVER_VERSION);o["commit"]=json::from_string(MULACOVER_SOURCE_COMMIT);
    const auto * runtime=mulacover_get_runtime_api(1);
    if(!runtime||runtime->size<MULACOVER_RUNTIME_API_V1_SIZE)throw std::runtime_error("Runtime metadata capability unavailable");
    o["library_version"]=json::from_string(runtime->library_version());
    o["ggml_version"]=json::from_string(runtime->ggml_version());
    o["ggml_commit"]=json::from_string(runtime->ggml_commit());o["backend"]=json::from_string(c.backend);
    o["source_dirty"]=json::from_bool(MULACOVER_SOURCE_DIRTY!=0);
    o["http_api_version"]=json::from_integer(1);o["sample_rate"]=json::from_integer(48000);o["channels"]=json::from_integer(2);
    o["max_cache_capacity"]=json::from_integer(c.max_cache_capacity);o["natural_max_seconds"]=json::from_double(30);
    o["early_eos"]=json::from_bool(true);o["transcription"]=json::from_bool(c.transcription_port>0);o["bpm_lock"]=json::from_bool(false);
    o["transcription_port"]=json::from_integer(c.transcription_port);
    o["job_abi"]=json::from_integer(1);o["conditioning_abi"]=json::from_integer(1);
    auto modes=json::array();modes.array={json::from_string("none"),json::from_string("provided")};o["score_modes"]=std::move(modes);
    auto models=json::object();bool ready=true;
    for(const auto & item:std::map<std::string,std::string>{{"generator",c.generator},{"tokenizers",c.tokenizers},{"style",c.style},{"encoder",c.encoder},{"flow",c.flow},{"sq",c.sq}}){
        std::error_code ec;const bool present=fs::is_regular_file(fs::u8path(item.second),ec);ready=ready&&present;
        auto model=json::object();model.object["file"]=json::from_string(fs::u8path(item.second).filename().u8string());model.object["present"]=json::from_bool(present);models.object[item.first]=std::move(model);
    }
    o["models"]=std::move(models);o["models_present"]=json::from_bool(ready);
    return json::stringify(v);
}
Outcome run_native(const Config & config,const Input & in,const Progress & progress,const Cancel & cancel) {
    if(cancel())throw Cancelled{};
    auto job=request(config,in);auto preliminary=plan(config,job);
    const bool automatic=json::boolean(in.options,"transcribe",false);TranscribedScore transcribed;
    if(automatic){
        const bool full_score=json::string(in.options,"transcription_scope","context")=="source";
        transcribed=transcribe_score(config,in.audio,full_score?0:preliminary.context_start_sample,full_score?int64_t(in.audio.samples.size()/in.audio.channels):preliminary.context_samples,
            json::string(in.options,"transcription_lane","combined"),[&](double p,const std::string & stage,int64_t n,int64_t total){progress(.25*p,stage,n,total);},cancel);
        transcribed.metadata.object["scope"]=json::from_string(full_score?"source":"context");
    }
    Progress pipeline_progress=[&](double p,const std::string & stage,int64_t n,int64_t total){progress((automatic?.25:0)+(automatic?.75:1)*p,stage,n,total);};
    const auto * conditioning=mulacover_get_conditioning_api(1);const auto * jobs=mulacover_get_job_api(1);
    if(!conditioning||conditioning->size<MULACOVER_CONDITIONING_API_V1_SIZE||!jobs||jobs->size<MULACOVER_JOB_API_V1_SIZE)throw std::runtime_error("Required native capability unavailable");
    struct Owned {
        const mulacover_conditioning_api_v1 * c;const mulacover_job_api_v1 * j;
        mulacover_conditioning_context * context=nullptr;
        mulacover_prepared_condition_v1 prepared{};mulacover_job_result_v1 result{};
        Owned(const mulacover_conditioning_api_v1 * a,const mulacover_job_api_v1 * b):c(a),j(b){prepared.size=sizeof(prepared);c->result_init(&prepared);result.size=sizeof(result);j->result_init(&result);}
        ~Owned(){j->result_free(&result);c->result_free(&prepared);if(context)c->destroy(context);}
    } owned(conditioning,jobs);
    mulacover_error_v1 e{};e.size=sizeof(e);
    auto check=[&](int code){if(code==3)throw Cancelled{};if(code)throw std::runtime_error(e.message);};
    pipeline_progress(0,"loading tokenizers",0,0);check(conditioning->create(config.tokenizers.c_str(),&owned.context,&e));
    if(cancel())throw Cancelled{};
    Scratch scratch;const auto melody=scratch.midi("melody.mid",automatic?transcribed.melody:in.melody),chords=scratch.midi("chords.mid",automatic?transcribed.chords:in.chords),drums=scratch.midi("drums.mid",in.drums);
    const auto lyrics=json::string(in.options,"lyrics","[Instrumental]"),tags=json::string(in.options,"tags","");
    mulacover_conditioning_request_v1 r{};r.size=sizeof(r);conditioning->request_init(&r);
    r.lyrics=lyrics.c_str();r.tags=tags.c_str();r.style_model=config.style.c_str();r.backend=config.backend.c_str();r.threads=config.threads;
    r.score_mode=job.score_mode;r.symbolic_frames=integer(in.options,"symbolic_frames",job.score_mode?0:540,0,5000);
    r.melody_midi=melody.empty()?nullptr:melody.c_str();r.chord_midi=chords.empty()?nullptr:chords.c_str();r.drum_midi=drums.empty()?nullptr:drums.c_str();
    r.guidance=job.conditioning.guidance;r.max_audio_frames=preliminary.guide_frames+preliminary.new_frames;
    Callbacks callbacks{pipeline_progress,cancel};r.should_cancel=Callbacks::poll;r.callback_user=&callbacks;r.on_style_progress=Callbacks::style;
    check(conditioning->prepare(owned.context,&r,&owned.prepared,&e));
    conditioning->destroy(owned.context);owned.context=nullptr;
    job.conditioning=owned.prepared.condition;plan(config,job); // actual text prompt budget
    job.on_progress=Callbacks::job;job.should_cancel=Callbacks::poll;job.callback_user=&callbacks;
    check(jobs->run(&job,&owned.result,&e));if(cancel())throw Cancelled{};
    auto & result=owned.result;Outcome out;out.wav=wav(result.samples,result.num_samples,result.sample_rate);
    if(result.full_samples)out.full_wav=wav(result.full_samples,result.full_num_samples,result.sample_rate);
    auto & meta=out.meta.object;
    if(automatic)meta["transcription"]=std::move(transcribed.metadata);
    meta["new_frames"]=json::from_integer(result.new_frames);meta["guide_frames"]=json::from_integer(result.guide_frames);
    meta["duration"]=json::from_double(double(result.num_samples)/result.sample_rate);meta["sample_rate"]=json::from_integer(result.sample_rate);meta["channels"]=json::from_integer(2);
    meta["eos"]=json::from_bool(result.stop_reason==MULACOVER_EOS_V1);meta["cache_capacity"]=json::from_integer(result.plan.cache_capacity);
    meta["cursor_sample"]=json::from_integer(result.plan.cursor_sample);meta["context_start_sample"]=json::from_integer(result.plan.context_start_sample);
    meta["seed"]=json::from_string(std::to_string(job.sampling.seed));meta["sampler_final_offset"]=json::from_string(std::to_string(result.sampler_final_offset));
    meta["decoder_final_offset"]=json::from_string(std::to_string(result.decoder_final_offset));
    meta["encode_seconds"]=json::from_double(result.encode_seconds);meta["generate_seconds"]=json::from_double(result.generate_seconds);meta["decode_seconds"]=json::from_double(result.decode_seconds);
    return out;
}
struct Service::Job {
    std::string id, status="queued",stage="queued",error;
    std::atomic<bool> cancelled{false};
    double progress=0;int64_t completed=0,total=0;
    Clock::time_point finished{};
    Input input;Outcome output;
    bool terminal()const{return status=="completed"||status=="cancelled"||status=="failed";}
};
Service::Service(Config c,Runner runner):config_(std::move(c)),runner_(std::move(runner)),worker_([this]{work();}){}
Service::~Service(){shutdown();}
void Service::shutdown(){
    {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;for(auto & j:jobs_)j.second->cancelled=true;}
    wake_.notify_all();if(worker_.joinable())worker_.join();
}
void Service::prune(){
    for(auto i=jobs_.begin();i!=jobs_.end();)if(i->second->terminal()&&Clock::now()-i->second->finished>std::chrono::minutes(30))i=jobs_.erase(i);else ++i;
    while(jobs_.size()>4){auto oldest=jobs_.end();for(auto i=jobs_.begin();i!=jobs_.end();++i)if(i->second->terminal()&&(oldest==jobs_.end()||i->second->finished<oldest->second->finished))oldest=i;if(oldest==jobs_.end())break;jobs_.erase(oldest);}
}
void Service::work(){
    for(;;){
        std::shared_ptr<Job> job;
        {std::unique_lock<std::mutex> lock(mutex_);wake_.wait(lock,[&]{return stopping_||pending_;});if(stopping_&&!pending_)return;job=std::move(pending_);job->status="running";}
        std::string error;bool cancelled=false;Outcome result;
        try{
            if(job->cancelled)throw Cancelled{};
            result=runner_(config_,job->input,[&](double p,const std::string & stage,int64_t n,int64_t total){std::lock_guard<std::mutex> lock(mutex_);job->progress=std::max(job->progress,std::clamp(p,0.0,1.0));job->stage=stage;job->completed=n;job->total=total;},[&]{return job->cancelled.load();});
            cancelled=job->cancelled;
        }catch(const Cancelled &){cancelled=true;}catch(const std::exception & ex){error=ex.what();}catch(...){error="Unknown native worker failure";}
        {std::lock_guard<std::mutex> lock(mutex_);job->input=Input{};
         if(cancelled||job->cancelled){job->status="cancelled";job->stage="cancelled";}
         else if(!error.empty()){job->status="failed";job->stage="failed";job->error=error;}
         else{job->status="completed";job->stage="done";job->progress=1;job->output=std::move(result);}
         job->finished=Clock::now();busy_=false;}
    }
}
HttpResponse Service::handle(const HttpRequest & req){
    HttpResponse out;
    try{
        const auto origin=req.headers.find("origin");
        if(origin!=req.headers.end()&&origin->second!="http://127.0.0.1:"+std::to_string(config_.port)&&origin->second!="http://localhost:"+std::to_string(config_.port)&&origin->second!=config_.allow_origin)
            out=error_response(403,"Origin is not allowed","forbidden");
        else out=route(req);
        if(origin!=req.headers.end()&&origin->second==config_.allow_origin&&!config_.allow_origin.empty()){
            out.headers["Access-Control-Allow-Origin"]=config_.allow_origin;out.headers["Vary"]="Origin";
            out.headers["Access-Control-Allow-Methods"]="GET, POST, OPTIONS";out.headers["Access-Control-Allow-Headers"]="Content-Type";
        }
    }catch(const std::invalid_argument & e){out=error_response(400,e.what(),"invalid_request");}
     catch(const std::exception & e){out=error_response(500,e.what(),"server_error");}
    out.headers["Cache-Control"]="no-store";out.headers["X-Content-Type-Options"]="nosniff";return out;
}
HttpResponse Service::route(const HttpRequest & req){
    if(req.method=="OPTIONS")return {200,"text/plain","",{}};
    const bool get=req.method=="GET",post=req.method=="POST";
    auto method=[](){return error_response(405,"Method not allowed","method_not_allowed");};
    if(req.path=="/props")return get?json_response(props(config_)):method();
    if(req.path=="/health"){
        if(!get)return method();auto v=json::parse(props(config_));std::lock_guard<std::mutex> lock(mutex_);v.object["status"]=json::from_string("ok");v.object["busy"]=json::from_bool(busy_);return json_response(json::stringify(v));
    }
    if(req.path=="/continue"){
        if(!post)return method();
        // Admission is serialized before parsing large uploads. No second job
        // or model context can be accepted while the worker is active.
        std::lock_guard<std::mutex> lock(mutex_);prune();
        if(stopping_)return error_response(503,"Server is stopping","unavailable");
        if(busy_)return error_response(409,"A job is already running","busy");
        auto job=std::make_shared<Job>();job->input=parse_input(req,config_);do{job->id=id();}while(jobs_.count(job->id));
        jobs_[job->id]=job;pending_=job;busy_=true;prune();wake_.notify_one();
        return json_response("{\"success\":true,\"session_id\":"+json::quote(job->id)+"}",202);
    }
    if(req.path=="/unload"){
        if(!post)return method();std::lock_guard<std::mutex> lock(mutex_);
        if(busy_)return error_response(409,"Cancel the running job before unloading","busy");
        jobs_.clear();return json_response("{\"success\":true}"); // models already released per job
    }
    const auto slash=req.path.find('/',1);
    if(slash==std::string::npos)return error_response(404,"Unknown route","not_found");
    const auto action=req.path.substr(0,slash),key=req.path.substr(slash+1);
    if(action!="/poll_status"&&action!="/cancel"&&action!="/audio"&&action!="/full_audio"&&action!="/release")return error_response(404,"Unknown route","not_found");
    if((action=="/cancel"||action=="/release")?!post:!get)return method();
    std::lock_guard<std::mutex> lock(mutex_);prune();auto it=jobs_.find(key);
    if(it==jobs_.end())return error_response(404,"Unknown or expired session","not_found");auto & j=*it->second;
    if(action=="/cancel"){if(!j.terminal())j.cancelled=true;return json_response("{\"success\":true}");}
    if(action=="/release"){if(!j.terminal())return error_response(409,"Cancel running work before releasing it","busy");jobs_.erase(it);return json_response("{\"success\":true}");}
    if(action=="/audio"||action=="/full_audio"){
        if(j.status!="completed")return error_response(409,"Audio is not ready","not_ready");
        const auto & bytes=action=="/audio"?j.output.wav:j.output.full_wav;
        if(bytes.empty())return error_response(404,"Full audio was not requested","not_found");
        return {200,"audio/wav",bytes,{{"Content-Disposition",std::string("inline; filename=\"")+(action=="/audio"?"continuation.wav":"guide-and-continuation.wav")+"\""}}};
    }
    auto v=json::object();auto & o=v.object;o["success"]=json::from_bool(j.status!="failed");o["session_id"]=json::from_string(j.id);o["status"]=json::from_string(j.status);
    o["generation_in_progress"]=json::from_bool(!j.terminal());o["progress"]=json::from_double(j.progress);o["stage"]=json::from_string(j.stage);o["step"]=json::from_integer(j.completed);o["total_steps"]=json::from_integer(j.total);
    if(!j.error.empty())o["error"]=json::from_string(j.error);
    if(j.status=="completed"){o["audio_url"]=json::from_string("/audio/"+j.id);o["meta"]=j.output.meta;if(!j.output.full_wav.empty())o["full_audio_url"]=json::from_string("/full_audio/"+j.id);}
    return json_response(json::stringify(v));
}
}

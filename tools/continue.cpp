#include "mulacover_job.h"
#include "wave-io.h"
#include "conditioning-args.h"
#include <chrono>
#include <csignal>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace {
volatile std::sig_atomic_t interrupted=0;
void interrupt(int){interrupted=1;}
struct Progress {std::string stage;int64_t last=-1;};
int32_t MULACOVER_CALL cancel(void *){return interrupted?1:0;}
void MULACOVER_CALL progress(void * user,const mulacover_job_progress_v1 * p){
    auto & state=*static_cast<Progress *>(user);
    if(state.stage!=p->stage_name||p->completed==p->total||p->completed-state.last>=32){
        state.stage=p->stage_name;state.last=p->completed;
        std::cout<<p->stage_name<<" "<<p->completed<<"/"<<p->total<<" ("<<int(p->fraction*100)<<"%)"<<std::endl;
    }
}
template<class T> void write(const std::filesystem::path & path,const T * data,size_t count){
    std::ofstream file(path,std::ios::binary);
    if(!file.write(reinterpret_cast<const char *>(data),std::streamsize(count*sizeof(T))))throw std::runtime_error("Cannot write output");
}
}
int main(int argc,char ** argv){
    mulacover_job_result_v1 result{};result.size=sizeof(result);mulacover_job_result_init_v1(&result);
    struct Owner{mulacover_job_result_v1 * r;~Owner(){mulacover_job_result_free_v1(r);}} owner{&result};
    try{
        if(argc==1)throw std::runtime_error("Usage: mulacover-continue --audio INPUT.wav --output FRESH_DIR --encoder-model GGUF --model GGUF --flow-model GGUF --sq-model GGUF; conditioning: --tokenizers-model GGUF --style-model GGUF --lyrics-file UTF8 --tags-file UTF8, or --input PRESET_DIR --prompt-frames N --qwen-index N --muq-index N; controls: --symbolic-frames N --cursor-sample N --context-samples N --guide-samples N --natural-max-seconds 30 --score-mode none|provided --melody-midi MIDI --chord-midi MIDI --drum-midi MIDI --include-full-audio 0|1");
        std::map<std::string,std::string> args;
        const std::set<std::string> known={"--audio","--input","--output","--encoder-model","--model","--flow-model","--sq-model","--prompt-frames","--symbolic-frames","--qwen-index","--muq-index","--cursor-sample","--context-samples","--guide-samples","--natural-max-seconds","--capacity","--backend","--threads","--score-mode","--guidance","--temperature","--top-k","--seed","--decoder-seed","--decoder-steps","--segment-codes","--rng","--rng-cuda-blocks","--rng-device","--sampler-offset","--include-full-audio","--tokenizers-model","--style-model","--lyrics-file","--tags-file","--melody-midi","--chord-midi","--drum-midi"};
        for(int i=1;i<argc;i+=2)if(i+1>=argc||!known.count(argv[i])||!args.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("Unknown/missing/duplicate option");
        auto get=[&](const char * key,const char * fallback){return args.count(key)?args.at(key):std::string(fallback);};
        auto seed=[&](const char * key,const char * fallback){auto text=get(key,fallback);if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("Invalid unsigned seed/offset");return std::stoull(text);};
        const bool native=args.count("--tokenizers-model")!=0;
        const std::filesystem::path input(get("--input","")),output(args.at("--output"));
        if(std::filesystem::exists(output))throw std::runtime_error("Use a fresh output directory");
        auto pcm=wave_io::read_wave(args.at("--audio"));
        const auto * api=mulacover_get_job_api(1);if(!api||api->size<MULACOVER_JOB_API_V1_SIZE)throw std::runtime_error("Native job capability unavailable");
        mulacover_job_request_v1 request{};request.size=sizeof(request);api->request_init(&request);
        request.encoder_model=args.at("--encoder-model").c_str();request.generator_model=args.at("--model").c_str();request.flow_model=args.at("--flow-model").c_str();request.sq_model=args.at("--sq-model").c_str();
        const auto backend=get("--backend","cuda");request.backend=backend.c_str();request.cpu_threads=std::stoi(get("--threads","4"));
        request.input_audio={sizeof(request.input_audio),pcm.samples.data(),int64_t(pcm.samples.size()/pcm.channels),pcm.rate,pcm.channels,1,pcm.channels};
        request.cursor_sample=std::stoll(get("--cursor-sample","-1"));request.context_samples=std::stoll(get("--context-samples","0"));request.guide_samples=std::stoll(get("--guide-samples","0"));
        request.natural_max_seconds=std::stod(get("--natural-max-seconds","30"));request.cache_capacity=std::stoi(get("--capacity","0"));
        Progress state;request.on_progress=progress;request.should_cancel=cancel;request.callback_user=&state;std::signal(SIGINT,interrupt);
        auto & condition=request.conditioning;
        const auto score=get("--score-mode","none");if(score!="none"&&score!="provided")throw std::runtime_error("Invalid score mode");
        std::vector<int32_t> tokens;std::vector<uint8_t> mask;std::vector<float> qwen,muq,piano,drum,chord;std::vector<uint8_t> context_mask;
        conditioning_tool::Prepared prepared;
        const auto conditioning_started=std::chrono::steady_clock::now();
        request.score_mode=score=="none"?MULACOVER_JOB_SCORE_NONE_V1:MULACOVER_JOB_SCORE_PROVIDED_V1;
        if(native){
            for(const char * key:{"--input","--prompt-frames","--qwen-index","--muq-index"})if(args.count(key))throw std::runtime_error("Native conditioning cannot use cached prompt/preset arguments");
            mulacover_job_plan_v1 preliminary{};preliminary.size=sizeof(preliminary);mulacover_error_v1 e{};e.size=sizeof(e);
            if(api->plan(&request,&preliminary,&e))throw std::runtime_error(e.message);
            const int status=prepared.prepare(args,preliminary.guide_frames+preliminary.new_frames,cancel,&state,e);if(status){std::cerr<<e.message<<std::endl;return status;}condition=prepared.value.condition;
        }else{
            for(const char * key:{"--style-model","--lyrics-file","--tags-file","--melody-midi","--chord-midi","--drum-midi"})if(args.count(key))throw std::runtime_error("Native conditioning arguments require --tokenizers-model");
            if(!args.count("--input"))throw std::runtime_error("Supply --input or --tokenizers-model");
            condition.prompt_frames=std::stoll(args.at("--prompt-frames"));condition.symbolic_frames=std::stoll(args.at("--symbolic-frames"));
            if(condition.prompt_frames<2||condition.prompt_frames>8192||condition.symbolic_frames<1||condition.symbolic_frames>5000)throw std::runtime_error("Invalid prepared dimensions");
            tokens=wave_io::read<int32_t>(input/"tokens.i32",size_t(condition.prompt_frames)*9);mask=wave_io::read<uint8_t>(input/"mask.u8",size_t(condition.prompt_frames)*9);qwen=wave_io::read<float>(input/"qwen.f32",1024);muq=wave_io::read<float>(input/"muq.f32",512);
            condition.tokens=tokens.data();condition.tokens_mask=mask.data();condition.qwen=qwen.data();condition.muq=muq.data();condition.qwen_index=std::stoll(args.at("--qwen-index"));condition.muq_index=std::stoll(args.at("--muq-index"));condition.guidance=std::stof(get("--guidance","1.5"));
            if(request.score_mode){piano=wave_io::read<float>(input/"piano.f32",size_t(condition.symbolic_frames)*256);drum=wave_io::read<float>(input/"drum.f32",size_t(condition.symbolic_frames)*128);chord=wave_io::read<float>(input/"chord.f32",size_t(condition.symbolic_frames)*12);context_mask=wave_io::read<uint8_t>(input/"context-mask.u8",size_t(condition.symbolic_frames));condition.pianoroll=piano.data();condition.drum_pianoroll=drum.data();condition.chord=chord.data();condition.context_mask=context_mask.data();}
        }
        const double conditioning_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-conditioning_started).count();
        request.sampling.seed=seed("--seed","1701");request.sampling.temperature=std::stof(get("--temperature","1"));request.sampling.top_k=std::stoi(get("--top-k","250"));
        const auto rng=get("--rng","torch-cuda");if(rng!="torch-cuda"&&rng!="native")throw std::runtime_error("Invalid RNG mode");request.rng_mode=rng=="torch-cuda"?1:0;
        const auto blocks=seed("--rng-cuda-blocks","288");if(blocks>1048576)throw std::runtime_error("Invalid RNG block cap");request.rng_cuda_blocks=uint32_t(blocks);request.rng_device=std::stoi(get("--rng-device","0"));request.sampler_offset=seed("--sampler-offset","0");
        request.decoder_seed=seed("--decoder-seed","1701");request.decoder_steps=std::stoi(get("--decoder-steps","10"));request.segment_codes=std::stoll(get("--segment-codes","372"));request.include_full_audio=std::stoi(get("--include-full-audio","0"));
        mulacover_error_v1 error{};error.size=sizeof(error);mulacover_job_plan_v1 plan{};plan.size=sizeof(plan);
        if(api->plan(&request,&plan,&error))throw std::runtime_error(error.message);
        std::cout<<"Context samples ["<<plan.context_start_sample<<","<<plan.cursor_sample<<"), guide "<<plan.guide_frames<<" frames, new ceiling "<<plan.new_frames<<", cache "<<plan.cache_capacity<<std::endl;
        const auto start=std::chrono::steady_clock::now();const int status=api->run(&request,&result,&error);
        if(status){std::cerr<<error.message<<std::endl;return status;}
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::filesystem::create_directories(output);wave_io::wav(output/"continuation.wav",result);
        if(native)prepared.save(output/"conditioning");
        if(result.full_samples){auto full=result;full.samples=result.full_samples;full.num_samples=result.full_num_samples;wave_io::wav(output/"audio.wav",full);}
        write(output/"codes.i32",result.codes,size_t(result.guide_frames+result.new_frames)*8);write(output/"guide.i32",result.codes,size_t(result.guide_frames)*8);
        std::ofstream report(output/"run.json");report<<std::setprecision(12)
          <<"{\"native_pcm_job\":true,\"native_conditioning\":"<<(native?"true":"false")<<",\"backend\":\""<<backend<<"\",\"score_mode\":\""<<score<<"\",\"rng\":\""<<rng<<"\",\"rng_cuda_blocks\":"<<request.rng_cuda_blocks
          <<",\"seed\":"<<request.sampling.seed<<",\"decoder_seed\":"<<request.decoder_seed<<",\"cursor_sample\":"<<plan.cursor_sample<<",\"context_start_sample\":"<<plan.context_start_sample<<",\"context_samples\":"<<plan.context_samples<<",\"context_frames\":"<<plan.context_frames<<",\"guide_start_frame\":"<<plan.guide_start_frame<<",\"guide_frames\":"<<result.guide_frames<<",\"new_frames\":"<<result.new_frames<<",\"requested_new_frames\":"<<plan.new_frames<<",\"cache_capacity\":"<<plan.cache_capacity<<",\"eos\":"<<(result.stop_reason==MULACOVER_EOS_V1?"true":"false")
          <<",\"sampler_final_offset\":"<<result.sampler_final_offset<<",\"decoder_final_offset\":"<<result.decoder_final_offset<<",\"decoder_rng_windows\":"<<result.decoder_rng_windows<<",\"samples_per_channel\":"<<result.num_samples
          <<",\"conditioning_seconds\":"<<conditioning_seconds<<",\"pipeline_seconds\":"<<conditioning_seconds+seconds<<",\"style_weight_bytes\":"<<prepared.value.style_weight_bytes<<",\"style_graph_bytes\":"<<prepared.value.style_graph_bytes
          <<",\"encode_seconds\":"<<result.encode_seconds<<",\"generate_seconds\":"<<result.generate_seconds<<",\"decode_seconds\":"<<result.decode_seconds<<",\"total_seconds\":"<<seconds
          <<",\"encoder_weight_bytes\":"<<result.encoder_weight_bytes<<",\"encoder_graph_bytes\":"<<result.encoder_graph_bytes<<",\"generator_weight_bytes\":"<<result.generator_weight_bytes<<",\"generator_cache_bytes\":"<<result.generator_cache_bytes<<",\"generator_graph_bytes\":"<<result.generator_graph_bytes<<"}\n";
        if(!report)throw std::runtime_error("Cannot write job report");
        std::cout<<"Generated "<<result.new_frames<<" new frames ("<<result.new_frames*.08<<"s), "<<(result.stop_reason==MULACOVER_EOS_V1?"EOS":"ceiling")<<", in "<<seconds<<"s"<<std::endl;return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<std::endl;return 1;}
}

#include "mulacover_sampling.h"
#include "heartcodec.h"
#include "heartcodec_rng.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

template<class T>static std::vector<T> read(const std::filesystem::path & path,size_t n) {
    if(std::filesystem::file_size(path)!=n*sizeof(T))throw std::runtime_error("Wrong input length: "+path.string());
    std::vector<T> v(n);std::ifstream f(path,std::ios::binary);
    if(!f.read(reinterpret_cast<char *>(v.data()),std::streamsize(n*sizeof(T))))throw std::runtime_error("Cannot read input");return v;
}
static void wav(const std::filesystem::path & path,const heartcodec_stereo_v1 & audio) {
    const uint64_t bytes=uint64_t(audio.num_samples)*8;if(bytes>UINT32_MAX-36)throw std::runtime_error("WAV is too large");
    std::ofstream f(path,std::ios::binary);auto u32=[&](uint32_t x){f.write(reinterpret_cast<const char *>(&x),4);};
    auto u16=[&](uint16_t x){f.write(reinterpret_cast<const char *>(&x),2);};
    f.write("RIFF",4);u32(uint32_t(bytes+36));f.write("WAVEfmt ",8);u32(16);u16(3);u16(2);
    u32(uint32_t(audio.sample_rate));u32(uint32_t(audio.sample_rate)*8);u16(8);u16(32);f.write("data",4);u32(uint32_t(bytes));
    for(int64_t i=0;i<audio.num_samples;++i){f.write(reinterpret_cast<const char *>(audio.samples+i),4);f.write(reinterpret_cast<const char *>(audio.samples+audio.num_samples+i),4);}
    if(!f)throw std::runtime_error("Cannot write float32 WAV");
}
int main(int argc,char ** argv) {
    mulacover_context * context=nullptr;heartcodec_context * codec=nullptr;
    mulacover_frame_v1 frame{};frame.size=sizeof(frame);heartcodec_stereo_v1 audio{};audio.size=sizeof(audio);
    heartcodec_rng_result_v1 random{};random.size=sizeof(random);
    try {
        std::map<std::string,std::string> a;
        for(int i=1;i<argc;i+=2)if(i+1>=argc||!a.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("Missing/duplicate option");
        for(const auto & pair:a) {
            const auto & k=pair.first;
            if(k!="--model"&&k!="--input"&&k!="--output"&&k!="--prompt-frames"&&k!="--symbolic-frames"&&k!="--qwen-index"&&k!="--muq-index"&&
               k!="--frames"&&k!="--capacity"&&k!="--backend"&&k!="--guidance"&&k!="--temperature"&&k!="--top-k"&&k!="--seed"&&k!="--replay"&&
               k!="--flow-model"&&k!="--sq-model"&&k!="--decoder-seed"&&k!="--decoder-steps"&&k!="--segment-codes"&&k!="--rng"&&k!="--rng-cuda-blocks"&&k!="--rng-device")throw std::runtime_error("Unknown option: "+k);
        }
        auto get=[&](const char * k,const char * d){return a.count(k)?a.at(k):std::string(d);};
        const std::filesystem::path input(a.at("--input")),output(a.at("--output"));
        if(std::filesystem::exists(output/"run.json"))throw std::runtime_error("Use a fresh generation output directory");
        const int prompt=std::stoi(a.at("--prompt-frames")),symbolic=std::stoi(a.at("--symbolic-frames")),frames=std::stoi(get("--frames","100"));
        const int capacity=std::stoi(get("--capacity","256"));const bool replay=get("--replay","1")=="1";
        if(prompt<1||symbolic<1||symbolic>5000||frames<1||frames>8192||int64_t(prompt)+frames-1>capacity)
            throw std::runtime_error("Invalid dimensions or insufficient cache capacity");
        if(a.count("--flow-model")!=a.count("--sq-model"))throw std::runtime_error("Supply both codec models or neither");
        const std::string rng=get("--rng","native");
        if(rng!="native"&&rng!="torch-cuda")throw std::runtime_error("RNG must be native or torch-cuda");
        const auto * rng_api=heartcodec_get_rng_api(1);
        const auto rng_blocks=std::stoull(get("--rng-cuda-blocks","0"));
        if(rng_blocks>1048576)throw std::runtime_error("Invalid target CUDA RNG grid cap");
        if(rng=="native"&&(a.count("--rng-cuda-blocks")||a.count("--rng-device")))throw std::runtime_error("CUDA RNG options require torch-cuda mode");
        uint32_t target_rng_blocks=0;
        uint64_t sampler_offset_per_head=0,decoder_final_offset=0;int64_t decoder_rng_windows=0;
        auto tokens=read<int32_t>(input/"tokens.i32",size_t(prompt)*9);auto mask=read<uint8_t>(input/"mask.u8",size_t(prompt)*9);
        auto qwen=read<float>(input/"qwen.f32",1024),muq=read<float>(input/"muq.f32",512);
        auto piano=read<float>(input/"piano.f32",size_t(symbolic)*256),drum=read<float>(input/"drum.f32",size_t(symbolic)*128),chord=read<float>(input/"chord.f32",size_t(symbolic)*12);
        auto padding=read<uint8_t>(input/"context-mask.u8",size_t(symbolic));
        const auto * api=mulacover_get_api(1);const auto * sampling=mulacover_get_sampling_api(1);
        if(!api||api->size<MULACOVER_API_V1_SIZE||!sampling||sampling->size<MULACOVER_SAMPLING_API_V1_SIZE)throw std::runtime_error("Required ABI capability unavailable");
        mulacover_config_v1 config{};config.size=sizeof(config);api->config_init(&config);
        config.model_path=a.at("--model").c_str();const std::string backend=get("--backend","cuda");config.backend=backend.c_str();config.cache_capacity=capacity;
        mulacover_error_v1 error{};error.size=sizeof(error);
        if(api->create(&config,&context,&error))throw std::runtime_error(error.message);
        mulacover_condition_v1 condition{};condition.size=sizeof(condition);condition.tokens=tokens.data();condition.tokens_mask=mask.data();condition.prompt_frames=prompt;
        condition.qwen=qwen.data();condition.muq=muq.data();condition.qwen_index=std::stoll(a.at("--qwen-index"));condition.muq_index=std::stoll(a.at("--muq-index"));
        condition.pianoroll=piano.data();condition.drum_pianoroll=drum.data();condition.chord=chord.data();condition.context_mask=padding.data();condition.symbolic_frames=symbolic;condition.guidance=std::stof(get("--guidance","1.5"));
        if(api->session_begin(context,&condition,&error))throw std::runtime_error(error.message);
        mulacover_sampling_config_v1 options{};options.size=sizeof(options);sampling->config_init(&options);
        options.temperature=std::stof(get("--temperature","1"));options.top_k=std::stoi(get("--top-k","250"));options.seed=std::stoull(get("--seed","1701"));
        if(rng=="torch-cuda") {
            if(!rng_api||rng_api->size<HEARTCODEC_RNG_API_V1_SIZE)throw std::runtime_error("Torch CUDA RNG capability unavailable");
            heartcodec_rng_config_v1 rc{};rc.size=sizeof(rc);rng_api->config_init(&rc);
            rc.distribution=HEARTCODEC_RNG_EXPONENTIAL_V1;rc.seed=options.seed;rc.count=8197;rc.draws=int64_t(frames)*8;
            rc.max_blocks=uint32_t(rng_blocks);rc.device=std::stoi(get("--rng-device","0"));
            heartcodec_error_v1 re{};re.size=sizeof(re);if(rng_api->generate(&rc,&random,&re))throw std::runtime_error(re.message);
            target_rng_blocks=rc.max_blocks?rc.max_blocks:random.device_max_blocks;options.noise_count=8*8197;
            sampler_offset_per_head=random.next_offset/uint64_t(rc.draws);
        }
        std::vector<int32_t> codes;int32_t final_stop=0;
        const auto started=std::chrono::steady_clock::now();
        for(int run=0;run<(replay?2:1);++run) {
            if(run&&api->session_reset(context,&error))throw std::runtime_error(error.message);
            std::vector<int32_t> selected;int32_t stop=0;
            for(int f=0;f<frames;++f) {
                if(random.data)options.exponential_noise=random.data+size_t(f)*8*8197;
                if(sampling->sample_frame(context,&options,nullptr,&frame,&stop,&error))throw std::runtime_error(error.message);
                if(frame.cache_position!=prompt+f)throw std::runtime_error("Wrong generation cache position");
                if(run==0&&f==0) {
                    mulacover_frame_v1 probe{};probe.size=sizeof(probe);int32_t sentinel=123;auto bad=options;bad.top_k=0;
                    if(sampling->sample_frame(context,&bad,nullptr,&probe,&sentinel,&error)!=MULACOVER_INVALID_ARGUMENT_V1||sentinel!=123||probe.logits)
                        throw std::runtime_error("Invalid sample options consumed state/output");
                    if(sampling->sample_frame(context,&options,nullptr,&frame,&sentinel,&error)!=MULACOVER_INVALID_ARGUMENT_V1||sentinel!=123)
                        throw std::runtime_error("Occupied sample result consumed state/output");
                }
                if(stop==MULACOVER_PADDING_V1)throw std::runtime_error("Generator emitted padding; no audio is produced");
                if(stop==MULACOVER_EOS_V1){api->frame_free(&frame);break;}
                selected.insert(selected.end(),frame.codes,frame.codes+8);api->frame_free(&frame);
            }
            if(run==0){codes=std::move(selected);final_stop=stop;}
            else if(codes!=selected||final_stop!=stop)throw std::runtime_error("Seeded generation did not replay exactly");
        }
        if(codes.empty())throw std::runtime_error("Generator emitted EOS before any audio");
        // Force both terminal outcomes with supplied noise, without changing the
        // sampled sequence. All eight heads are computed, including after EOS.
        for(int token:{8193,8192}) {
            if(api->session_reset(context,&error))throw std::runtime_error(error.message);
            auto forced=options;forced.temperature=1000;forced.top_k=8197;std::vector<float> noise(8*8197,1);
            for(int q=0;q<8;++q)noise[size_t(q*8197+token)]=1e-30f;
            forced.exponential_noise=noise.data();forced.noise_count=int64_t(noise.size());int32_t stop=-1;
            if(sampling->sample_frame(context,&forced,nullptr,&frame,&stop,&error)||stop!=(token==8193?MULACOVER_EOS_V1:MULACOVER_PADDING_V1))
                throw std::runtime_error("Terminal sampling contract failed");
            api->frame_free(&frame);int32_t sentinel=123;
            if(sampling->sample_frame(context,&options,nullptr,&frame,&sentinel,&error)!=MULACOVER_INVALID_ARGUMENT_V1||frame.logits||sentinel!=123)
                throw std::runtime_error("Terminal session accepted another frame");
        }
        if(api->session_reset(context,&error))throw std::runtime_error(error.message);int32_t restart_stop=-1;
        if(random.data)options.exponential_noise=random.data;
        if(sampling->sample_frame(context,&options,nullptr,&frame,&restart_stop,&error)||restart_stop||!std::equal(frame.codes,frame.codes+8,codes.begin()))
            throw std::runtime_error("Reset after terminal state did not rewind generation");
        api->frame_free(&frame);
        mulacover_memory_v1 memory{};memory.size=sizeof(memory);if(api->get_memory(context,&memory,&error))throw std::runtime_error(error.message);
        const double generation_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        api->destroy(context);context=nullptr; // Release generator weights/caches before codec allocation.
        heartcodec_rng_result_free_v1(&random);options.exponential_noise=nullptr;options.noise_count=0;
        std::filesystem::create_directories(output);std::ofstream token_file(output/"codes.i32",std::ios::binary);
        token_file.write(reinterpret_cast<const char *>(codes.data()),std::streamsize(codes.size()*4));token_file.close();if(!token_file)throw std::runtime_error("Cannot write sampled tokens");
        double peak=0,sum=0,decode_seconds=0;uint64_t samples=0;
        if(a.count("--flow-model")) {
            const auto decode_started=std::chrono::steady_clock::now();
            heartcodec_config_v1 hc{};hc.size=sizeof(hc);heartcodec_config_init_v1(&hc);hc.model_path=a.at("--flow-model").c_str();hc.backend=backend.c_str();
            heartcodec_error_v1 e{};e.size=sizeof(e);
            if(heartcodec_context_create_v1(&hc,&codec,&e)||heartcodec_component_set_v1(codec,HEARTCODEC_SQ_V1,a.at("--sq-model").c_str(),&e))throw std::runtime_error(e.message);
            heartcodec_flow_config_v1 flow{};flow.size=sizeof(flow);heartcodec_flow_config_init_v1(&flow);
            flow.seed=std::stoull(get("--decoder-seed","1701"));flow.steps=std::stoi(get("--decoder-steps","10"));flow.segment_codes=std::stoll(get("--segment-codes","372"));
            heartcodec_tokens_v1 tokens_view{sizeof(tokens_view),codes.data(),int64_t(codes.size()/8),1,8};
            if(rng=="torch-cuda") {
                const int64_t window=flow.segment_codes;
                if(window<93||window%93||window>INT32_MAX/2)throw std::runtime_error("Invalid codec RNG window");
                const int64_t hop=window/93*80,overlap=window-hop;
                const int64_t padded=(std::max(window,tokens_view.frames)-overlap+hop-1)/hop*hop+overlap;
                heartcodec_rng_config_v1 rc{};rc.size=sizeof(rc);rng_api->config_init(&rc);
                rc.seed=flow.seed;rc.count=window*2*256;rc.draws=(padded-window)/hop+1;rc.max_blocks=target_rng_blocks;rc.device=std::stoi(get("--rng-device","0"));
                if(rng_api->generate(&rc,&random,&e))throw std::runtime_error(e.message);
                flow.noise=random.data;flow.noise_segments=random.draws;
            }
            if(heartcodec_tokens_decode_v1(codec,&tokens_view,&flow,&audio,&e))throw std::runtime_error(e.message);
            decoder_final_offset=random.next_offset;decoder_rng_windows=random.draws;
            heartcodec_rng_result_free_v1(&random);
            samples=uint64_t(audio.num_samples);
            if(samples!=uint64_t(codes.size()/8)*3840||audio.sample_rate!=48000)throw std::runtime_error("Unexpected decoded duration/rate");
            for(uint64_t i=0;i<samples*2;++i){double x=audio.samples[i];if(!std::isfinite(x))throw std::runtime_error("Nonfinite waveform");peak=std::max(peak,std::fabs(x));sum+=x*x;}
            wav(output/"audio.wav",audio);heartcodec_stereo_free_v1(&audio);heartcodec_context_destroy_v1(codec);codec=nullptr;
            decode_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-decode_started).count();
        }
        std::ofstream report(output/"run.json");
        report<<"{\"frames\":"<<codes.size()/8<<",\"prompt_frames\":"<<prompt<<",\"symbolic_frames\":"<<symbolic<<",\"cache_capacity\":"<<capacity
            <<",\"guidance\":"<<condition.guidance<<",\"temperature\":"<<options.temperature<<",\"top_k\":"<<options.top_k<<",\"seed\":"<<options.seed
            <<",\"eos\":"<<(final_stop==MULACOVER_EOS_V1?"true":"false")<<",\"replay_checked\":"<<(replay?"true":"false")
            <<",\"terminal_contract_passed\":true,\"generator_released_before_decode\":true,\"generation_with_checks_seconds\":"<<generation_seconds
            <<",\"weight_bytes\":"<<memory.weight_bytes<<",\"cache_bytes\":"<<memory.cache_bytes<<",\"graph_bytes\":"<<memory.graph_bytes
            <<",\"audio_samples_per_channel\":"<<samples<<",\"audio_peak\":"<<peak<<",\"audio_rms\":"<<(samples?std::sqrt(sum/double(samples*2)):0)
            <<",\"decode_seconds\":"<<decode_seconds<<",\"decoder_seed\":"<<std::stoull(get("--decoder-seed","1701"))
            <<",\"decoder_steps\":"<<std::stoi(get("--decoder-steps","10"))<<",\"segment_codes\":"<<std::stoll(get("--segment-codes","372"))
            <<",\"decoder_guidance\":1.25,\"rng\":\""<<rng<<"\",\"rng_cuda_blocks\":"<<target_rng_blocks
            <<",\"sampler_offset_per_head\":"<<sampler_offset_per_head
            <<",\"sampler_final_offset\":"<<((codes.size()/8+uint64_t(final_stop==MULACOVER_EOS_V1))*8*sampler_offset_per_head)
            <<",\"decoder_final_offset\":"<<decoder_final_offset<<",\"decoder_rng_windows\":"<<decoder_rng_windows
            <<",\"ggml_commit\":\""<<ggml_commit()<<"\"}\n";
        if(!report)throw std::runtime_error("Cannot write run report");
        std::cout<<"Generated "<<codes.size()/8<<" frames; "<<samples<<" stereo samples/channel\n";return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';heartcodec_rng_result_free_v1(&random);mulacover_frame_free_v1(&frame);mulacover_destroy_v1(context);heartcodec_stereo_free_v1(&audio);heartcodec_context_destroy_v1(codec);return 1;}
}

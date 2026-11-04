#include "mulacover_sampling.h"
#include "mulacover_tempo.h"
#include "mulacover_continuation.h"
#include "continuation-time.h"
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
template<class T>static void wav(const std::filesystem::path & path,const T & audio) {
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
    mulacover_continuation_audio_v1 suffix{};suffix.size=sizeof(suffix);
    try {
        std::map<std::string,std::string> a;
        for(int i=1;i<argc;i+=2)if(i+1>=argc||!a.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("Missing/duplicate option");
        for(const auto & pair:a) {
            const auto & k=pair.first;
            if(k!="--model"&&k!="--input"&&k!="--output"&&k!="--prompt-frames"&&k!="--symbolic-frames"&&k!="--qwen-index"&&k!="--muq-index"&&
               k!="--frames"&&k!="--capacity"&&k!="--backend"&&k!="--guidance"&&k!="--temperature"&&k!="--top-k"&&k!="--seed"&&k!="--replay"&&
               k!="--flow-model"&&k!="--sq-model"&&k!="--decoder-seed"&&k!="--decoder-steps"&&k!="--segment-codes"&&k!="--rng"&&k!="--rng-cuda-blocks"&&k!="--rng-device"&&
               k!="--experimental-bpm"&&k!="--tempo-width-beats"&&k!="--tempo-offset-beats"&&k!="--tempo-local-window"&&k!="--audio-prefix"&&k!="--sampler-offset"&&
               k!="--guide-seconds"&&k!="--guide-start-seconds"&&k!="--continuation-seconds"&&k!="--score-mode")throw std::runtime_error("Unknown option: "+k);
        }
        auto get=[&](const char * k,const char * d){return a.count(k)?a.at(k):std::string(d);};
        const std::filesystem::path input(a.at("--input")),output(a.at("--output"));
        if(std::filesystem::exists(output/"run.json"))throw std::runtime_error("Use a fresh generation output directory");
        const int prompt=std::stoi(a.at("--prompt-frames")),symbolic=std::stoi(a.at("--symbolic-frames"));
        const auto score_mode=get("--score-mode","provided");
        if(score_mode!="provided"&&score_mode!="none")throw std::runtime_error("Score mode must be provided or none");
        const bool replay=get("--replay","1")=="1";
        if(prompt<1||prompt>8192||symbolic<1||symbolic>5000)throw std::runtime_error("Invalid conditioning dimensions");
        if(a.count("--frames")&&a.count("--continuation-seconds"))throw std::runtime_error("Choose --frames or --continuation-seconds");
        if(a.count("--flow-model")!=a.count("--sq-model"))throw std::runtime_error("Supply both codec models or neither");
        std::vector<int32_t> prefix;size_t prefix_frames=0;
        if(a.count("--audio-prefix")) {
            const auto bytes=std::filesystem::file_size(a.at("--audio-prefix"));
            if(bytes==0||bytes%32)throw std::runtime_error("Prefix must contain complete RVQ frames");
            prefix=read<int32_t>(a.at("--audio-prefix"),size_t(bytes/4));prefix_frames=prefix.size()/8;
            if(std::any_of(prefix.begin(),prefix.end(),[](int32_t x){return x<0||x>=8192;}))throw std::runtime_error("Invalid audio-prefix RVQ code");
        }
        if(!a.count("--audio-prefix")&&(a.count("--guide-seconds")||a.count("--guide-start-seconds")))throw std::runtime_error("Guide selection requires --audio-prefix");
        const auto * continuation=mulacover_get_continuation_api(1);
        if(!continuation||continuation->size<MULACOVER_CONTINUATION_API_V1_SIZE)throw std::runtime_error("Continuation capability unavailable");
        mulacover_error_v1 error{};error.size=sizeof(error);
        mulacover_continuation_config_v1 timeline{};timeline.size=sizeof(timeline);continuation->config_init(&timeline);
        timeline.source_frames=int64_t(prefix_frames);timeline.source_start_frame=mula::duration_frames(get("--guide-start-seconds","0"));
        timeline.guide_frames=a.count("--guide-seconds")?mula::duration_frames(a.at("--guide-seconds")):0;
        if(a.count("--guide-seconds")&&timeline.guide_frames==0)throw std::runtime_error("Guide duration must include an 80-ms frame");
        if(timeline.source_start_frame>timeline.source_frames||timeline.guide_frames>timeline.source_frames-timeline.source_start_frame)
            throw std::runtime_error("Guide selection exceeds encoded audio");
        const int64_t guide=timeline.guide_frames?timeline.guide_frames:timeline.source_frames-timeline.source_start_frame;
        if(guide>8192)throw std::runtime_error("Guide exceeds the model context");
        if(a.count("--continuation-seconds"))timeline.new_frames=mula::duration_frames(a.at("--continuation-seconds"));
        else {
            const int64_t requested=std::stoll(get("--frames","100"));
            if(requested<1||requested>8192)throw std::runtime_error("Invalid total frame count");
            timeline.new_frames=requested-guide;
        }
        timeline.prompt_frames=prompt;timeline.cache_capacity=std::stoi(get("--capacity","0"));
        mulacover_continuation_plan_v1 plan{};plan.size=sizeof(plan);continuation->plan_init(&plan);
        if(continuation->plan(&timeline,&plan,&error))throw std::runtime_error(error.message);
        const int frames=int(plan.total_frames),capacity=plan.cache_capacity;
        if(!prefix.empty()) {
            const auto first=prefix.begin()+size_t(plan.source_start_frame)*8;
            std::vector<int32_t> selected(first,first+size_t(plan.guide_frames)*8);prefix.swap(selected);
        }
        prefix_frames=size_t(plan.guide_frames);
        const std::string rng=get("--rng","native");
        if(rng!="native"&&rng!="torch-cuda")throw std::runtime_error("RNG must be native or torch-cuda");
        const auto sampler_offset_text=get("--sampler-offset","0");
        if(sampler_offset_text.empty()||sampler_offset_text.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("Invalid sampler offset");
        const auto sampler_start_offset=std::stoull(sampler_offset_text);
        if(sampler_start_offset%4||(rng!="torch-cuda"&&sampler_start_offset))throw std::runtime_error("Sampler offset requires Torch CUDA mode and a multiple of four");
        const auto * rng_api=heartcodec_get_rng_api(1);
        const auto rng_blocks=std::stoull(get("--rng-cuda-blocks","0"));
        if(rng_blocks>1048576)throw std::runtime_error("Invalid target CUDA RNG grid cap");
        if(rng=="native"&&(a.count("--rng-cuda-blocks")||a.count("--rng-device")))throw std::runtime_error("CUDA RNG options require torch-cuda mode");
        uint32_t target_rng_blocks=0;
        uint64_t sampler_offset_per_head=0,decoder_final_offset=0;int64_t decoder_rng_windows=0;
        auto tokens=read<int32_t>(input/"tokens.i32",size_t(prompt)*9);auto mask=read<uint8_t>(input/"mask.u8",size_t(prompt)*9);
        auto qwen=read<float>(input/"qwen.f32",1024),muq=read<float>(input/"muq.f32",512);
        // Upstream _mask_condition uses -1 for dropped symbolic conditioning.
        // Retain the requested context length and an unpadded mask; do not read
        // any score files in this experimental audio-only continuation mode.
        auto piano=score_mode=="none"?std::vector<float>(size_t(symbolic)*256,-1.0f):read<float>(input/"piano.f32",size_t(symbolic)*256);
        auto drum=score_mode=="none"?std::vector<float>(size_t(symbolic)*128,-1.0f):read<float>(input/"drum.f32",size_t(symbolic)*128);
        auto chord=score_mode=="none"?std::vector<float>(size_t(symbolic)*12,-1.0f):read<float>(input/"chord.f32",size_t(symbolic)*12);
        auto padding=score_mode=="none"?std::vector<uint8_t>(size_t(symbolic),0):read<uint8_t>(input/"context-mask.u8",size_t(symbolic));
        const auto * api=mulacover_get_api(1);const auto * sampling=mulacover_get_sampling_api(1);
        if(!api||api->size<MULACOVER_API_V1_SIZE||!sampling||sampling->size<MULACOVER_SAMPLING_API_V1_SIZE)throw std::runtime_error("Required ABI capability unavailable");
        mulacover_config_v1 config{};config.size=sizeof(config);api->config_init(&config);
        config.model_path=a.at("--model").c_str();const std::string backend=get("--backend","cuda");config.backend=backend.c_str();config.cache_capacity=capacity;
        if(api->create(&config,&context,&error))throw std::runtime_error(error.message);
        mulacover_condition_v1 condition{};condition.size=sizeof(condition);condition.tokens=tokens.data();condition.tokens_mask=mask.data();condition.prompt_frames=prompt;
        condition.qwen=qwen.data();condition.muq=muq.data();condition.qwen_index=std::stoll(a.at("--qwen-index"));condition.muq_index=std::stoll(a.at("--muq-index"));
        condition.pianoroll=piano.data();condition.drum_pianoroll=drum.data();condition.chord=chord.data();condition.context_mask=padding.data();condition.symbolic_frames=symbolic;condition.guidance=std::stof(get("--guidance","1.5"));
        if(api->session_begin(context,&condition,&error))throw std::runtime_error(error.message);
        const auto * tempo_api=mulacover_get_tempo_api(1);
        mulacover_tempo_config_v1 tempo{};tempo.size=sizeof(tempo);tempo_api->config_init(&tempo);
        tempo.bpm=std::stod(get("--experimental-bpm","0"));tempo.width_beats=std::stod(get("--tempo-width-beats","1"));
        tempo.offset_beats=std::stod(get("--tempo-offset-beats","0"));
        const auto local_window=get("--tempo-local-window","0");
        if(local_window!="0"&&local_window!="1")throw std::runtime_error("Local score window must be 0 or 1");
        tempo.local_window=local_window=="1";
        if(!a.count("--experimental-bpm")&&(a.count("--tempo-width-beats")||a.count("--tempo-offset-beats")||a.count("--tempo-local-window")))throw std::runtime_error("Tempo options require --experimental-bpm");
        if(tempo_api->set(context,&tempo,&error))throw std::runtime_error(error.message);
        mulacover_sampling_config_v1 options{};options.size=sizeof(options);sampling->config_init(&options);
        options.temperature=std::stof(get("--temperature","1"));options.top_k=std::stoi(get("--top-k","250"));options.seed=std::stoull(get("--seed","1701"));
        if(rng=="torch-cuda") {
            if(!rng_api||rng_api->size<HEARTCODEC_RNG_API_V1_SIZE)throw std::runtime_error("Torch CUDA RNG capability unavailable");
            heartcodec_rng_config_v1 rc{};rc.size=sizeof(rc);rng_api->config_init(&rc);
            rc.distribution=HEARTCODEC_RNG_EXPONENTIAL_V1;rc.seed=options.seed;rc.count=8197;rc.draws=(int64_t(frames)-int64_t(prefix_frames))*8;
            rc.offset=sampler_start_offset;
            rc.max_blocks=uint32_t(rng_blocks);rc.device=std::stoi(get("--rng-device","0"));
            heartcodec_error_v1 re{};re.size=sizeof(re);if(rng_api->generate(&rc,&random,&re))throw std::runtime_error(re.message);
            target_rng_blocks=rc.max_blocks?rc.max_blocks:random.device_max_blocks;options.noise_count=8*8197;
            sampler_offset_per_head=(random.next_offset-sampler_start_offset)/uint64_t(rc.draws);
        }
        std::vector<int32_t> codes;int32_t final_stop=0;
        const auto started=std::chrono::steady_clock::now();
        for(int run=0;run<(replay?2:1);++run) {
            if(run&&api->session_reset(context,&error))throw std::runtime_error(error.message);
            std::vector<int32_t> selected;int32_t stop=0;
            for(int f=0;f<frames;++f) {
                if(size_t(f)<prefix_frames) {
                    if(api->teacher_frame(context,prefix.data()+size_t(f)*8,nullptr,&frame,&error))throw std::runtime_error(error.message);
                    stop=0;
                }else {
                    if(random.data)options.exponential_noise=random.data+(size_t(f)-prefix_frames)*8*8197;
                    if(sampling->sample_frame(context,&options,nullptr,&frame,&stop,&error))throw std::runtime_error(error.message);
                }
                if(frame.cache_position!=prompt+f)throw std::runtime_error("Wrong generation cache position");
                if(run==0&&f==0) {
                    if(tempo_api->set(context,&tempo,&error)!=MULACOVER_INVALID_ARGUMENT_V1)
                        throw std::runtime_error("Running session accepted score-clock change");
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
        if(codes.size()/8<=prefix_frames)throw std::runtime_error("Generator emitted EOS before any new audio after the guide");
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
        const int restart_status=prefix_frames?api->teacher_frame(context,prefix.data(),nullptr,&frame,&error):sampling->sample_frame(context,&options,nullptr,&frame,&restart_stop,&error);
        if(prefix_frames)restart_stop=0;
        if(restart_status||restart_stop||!std::equal(frame.codes,frame.codes+8,codes.begin()))
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
            wav(output/"audio.wav",audio);
            if(prefix_frames) {
                if(continuation->copy_audio(audio.samples,audio.num_samples,audio.sample_rate,int64_t(prefix_frames),&suffix,&error))throw std::runtime_error(error.message);
                wav(output/"continuation.wav",suffix);continuation->audio_free(&suffix);
            }
            heartcodec_stereo_free_v1(&audio);heartcodec_context_destroy_v1(codec);codec=nullptr;
            decode_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-decode_started).count();
        }
        std::ofstream report(output/"run.json");
        report<<"{\"frames\":"<<codes.size()/8<<",\"prompt_frames\":"<<prompt<<",\"symbolic_frames\":"<<symbolic<<",\"score_mode\":\""<<score_mode<<"\",\"cache_capacity\":"<<capacity
            <<",\"audio_prefix_frames\":"<<prefix_frames<<",\"newly_generated_frames\":"<<(codes.size()/8-prefix_frames)
            <<",\"guide_source_frames\":"<<timeline.source_frames<<",\"guide_source_start_frame\":"<<plan.source_start_frame
            <<",\"requested_new_frames\":"<<plan.new_frames<<",\"continuation_samples_per_channel\":"<<(samples?uint64_t(codes.size()/8-prefix_frames)*3840:0)
            <<",\"guidance\":"<<condition.guidance<<",\"temperature\":"<<options.temperature<<",\"top_k\":"<<options.top_k<<",\"seed\":"<<options.seed
            <<",\"experimental_tempo\":{\"method\":\"conditional-cross-attention-gaussian-v1\",\"bpm\":"<<tempo.bpm
            <<",\"width_beats\":"<<tempo.width_beats<<",\"offset_beats\":"<<tempo.offset_beats<<",\"local_window\":"<<tempo.local_window<<"}"
            <<",\"eos\":"<<(final_stop==MULACOVER_EOS_V1?"true":"false")<<",\"replay_checked\":"<<(replay?"true":"false")
            <<",\"terminal_contract_passed\":true,\"generator_released_before_decode\":true,\"generation_with_checks_seconds\":"<<generation_seconds
            <<",\"weight_bytes\":"<<memory.weight_bytes<<",\"cache_bytes\":"<<memory.cache_bytes<<",\"graph_bytes\":"<<memory.graph_bytes
            <<",\"audio_samples_per_channel\":"<<samples<<",\"audio_peak\":"<<peak<<",\"audio_rms\":"<<(samples?std::sqrt(sum/double(samples*2)):0)
            <<",\"decode_seconds\":"<<decode_seconds<<",\"decoder_seed\":"<<std::stoull(get("--decoder-seed","1701"))
            <<",\"decoder_steps\":"<<std::stoi(get("--decoder-steps","10"))<<",\"segment_codes\":"<<std::stoll(get("--segment-codes","372"))
            <<",\"decoder_guidance\":1.25,\"rng\":\""<<rng<<"\",\"rng_cuda_blocks\":"<<target_rng_blocks
            <<",\"sampler_offset_per_head\":"<<sampler_offset_per_head
            <<",\"sampler_start_offset\":"<<sampler_start_offset
            <<",\"sampler_final_offset\":"<<(sampler_start_offset+(codes.size()/8-prefix_frames+uint64_t(final_stop==MULACOVER_EOS_V1))*8*sampler_offset_per_head)
            <<",\"decoder_final_offset\":"<<decoder_final_offset<<",\"decoder_rng_windows\":"<<decoder_rng_windows
            <<",\"ggml_commit\":\""<<ggml_commit()<<"\"}\n";
        if(!report)throw std::runtime_error("Cannot write run report");
        std::cout<<"Generated "<<codes.size()/8<<" frames; "<<samples<<" stereo samples/channel\n";return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';mulacover_continuation_audio_free_v1(&suffix);heartcodec_rng_result_free_v1(&random);mulacover_frame_free_v1(&frame);mulacover_destroy_v1(context);heartcodec_stereo_free_v1(&audio);heartcodec_context_destroy_v1(codec);return 1;}
}

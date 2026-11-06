#include "mulacover.h"
#include "mulacover_sampling.h"
#include "mulacover_tempo.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T>std::vector<T> read(const std::filesystem::path & path,size_t count) {
    if(std::filesystem::file_size(path)!=count*sizeof(T))throw std::runtime_error("Wrong input length: "+path.string());
    std::vector<T> result(count);std::ifstream f(path,std::ios::binary);
    if(!f.read(reinterpret_cast<char *>(result.data()),std::streamsize(count*sizeof(T))))throw std::runtime_error("Cannot read input");return result;
}
int main(int argc,char ** argv) {
    mulacover_context * c=nullptr;mulacover_frame_v1 result{};result.size=sizeof(result);
    try {
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;i+=2)if(i+1>=argc||!args.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("Missing or duplicate option");
        for(auto key:{"--model","--input","--output","--prompt-frames","--symbolic-frames","--qwen-index","--muq-index"})
            if(!args.count(key))throw std::runtime_error(std::string("Missing ")+key);
        for(auto & pair:args)if(pair.first!="--model"&&pair.first!="--input"&&pair.first!="--output"&&pair.first!="--check-condition-reuse"&&pair.first!="--prompt-frames"&&pair.first!="--symbolic-frames"&&
            pair.first!="--qwen-index"&&pair.first!="--muq-index"&&pair.first!="--guidance"&&pair.first!="--backend"&&pair.first!="--capacity"&&pair.first!="--frames"&&pair.first!="--dump-frames"&&pair.first!="--sampling-noise"&&pair.first!="--experimental-bpm"&&pair.first!="--tempo-width-beats"&&pair.first!="--tempo-offset-beats"&&pair.first!="--tempo-local-window")throw std::runtime_error("Unknown option: "+pair.first);
        const std::filesystem::path input(args.at("--input")),output(args.at("--output"));
        int prompt=std::stoi(args.at("--prompt-frames")),symbolic=std::stoi(args.at("--symbolic-frames"));
        int frames=args.count("--frames")?std::stoi(args.at("--frames")):3;
        if(prompt<1||prompt>8192||symbolic<1||symbolic>5000||frames<1||frames>8192)throw std::runtime_error("Invalid test dimensions");
        auto tokens=read<int32_t>(input/"tokens.i32",size_t(prompt)*9);auto mask=read<uint8_t>(input/"mask.u8",size_t(prompt)*9);
        auto qwen=read<float>(input/"qwen.f32",1024),muq=read<float>(input/"muq.f32",512);
        auto piano=read<float>(input/"piano.f32",size_t(symbolic)*256),drum=read<float>(input/"drum.f32",size_t(symbolic)*128),chord=read<float>(input/"chord.f32",size_t(symbolic)*12);
        auto context_mask=read<uint8_t>(input/"context-mask.u8",size_t(symbolic));auto teacher=read<int32_t>(input/"teacher.i32",size_t(frames)*8);
        std::set<int> checkpoints;
        if(args.count("--dump-frames")) {
            std::istringstream list(args.at("--dump-frames"));std::string item;
            while(std::getline(list,item,',')){int f=std::stoi(item);if(f<0||f>=frames)throw std::runtime_error("Invalid dump frame");checkpoints.insert(f);}
        }else for(int f=0;f<frames;++f)checkpoints.insert(f);
        std::vector<float> sampling_noise;
        if(args.count("--sampling-noise"))sampling_noise=read<float>(args.at("--sampling-noise"),size_t(frames)*8*8197);
        mulacover_config_v1 config{};config.size=sizeof(config);mulacover_config_init_v1(&config);
        config.model_path=args.at("--model").c_str();if(args.count("--backend"))config.backend=args.at("--backend").c_str();
        if(args.count("--capacity"))config.cache_capacity=std::stoi(args.at("--capacity"));
        mulacover_error_v1 error{};error.size=sizeof(error);
        if(mulacover_create_v1(&config,&c,&error))throw std::runtime_error(error.message);
        mulacover_condition_v1 condition{};condition.size=sizeof(condition);condition.tokens=tokens.data();condition.tokens_mask=mask.data();condition.prompt_frames=prompt;
        condition.qwen=qwen.data();condition.muq=muq.data();condition.qwen_index=std::stoll(args.at("--qwen-index"));condition.muq_index=std::stoll(args.at("--muq-index"));
        condition.pianoroll=piano.data();condition.drum_pianoroll=drum.data();condition.chord=chord.data();condition.context_mask=context_mask.data();condition.symbolic_frames=symbolic;
        condition.guidance=args.count("--guidance")?std::stof(args.at("--guidance")):1.0f;
        if(mulacover_session_begin_v1(c,&condition,&error))throw std::runtime_error(error.message);
        if(args.count("--experimental-bpm")) {
            mulacover_tempo_config_v1 tempo{};tempo.size=sizeof(tempo);mulacover_tempo_config_init_v1(&tempo);
            tempo.bpm=std::stod(args.at("--experimental-bpm"));
            if(args.count("--tempo-width-beats"))tempo.width_beats=std::stod(args.at("--tempo-width-beats"));
            if(args.count("--tempo-offset-beats"))tempo.offset_beats=std::stod(args.at("--tempo-offset-beats"));
            if(args.count("--tempo-local-window")) {
                if(args.at("--tempo-local-window")!="0"&&args.at("--tempo-local-window")!="1")throw std::runtime_error("Local score window must be 0 or 1");
                tempo.local_window=args.at("--tempo-local-window")=="1";
            }
            if(mulacover_set_tempo_v1(c,&tempo,&error))throw std::runtime_error(error.message);
        }else if(args.count("--tempo-width-beats")||args.count("--tempo-offset-beats")||args.count("--tempo-local-window"))throw std::runtime_error("Tempo options require --experimental-bpm");
        std::filesystem::create_directories(output);const auto start=std::chrono::steady_clock::now();
        std::vector<float> first_logits;
        for(int replay=0;replay<2;++replay) {
            if(replay&&mulacover_session_reset_v1(c,&error))throw std::runtime_error(error.message);
            for(int frame=0;frame<frames;++frame) {
                const auto path=replay?(output/"replay").string():output.string();
                const char * dump=checkpoints.count(frame)?path.c_str():nullptr;
                if(sampling_noise.empty()) {
                    if(mulacover_teacher_frame_v1(c,teacher.data()+frame*8,dump,&result,&error))throw std::runtime_error(error.message);
                }else {
                    mulacover_sampling_config_v1 sampling{};sampling.size=sizeof(sampling);mulacover_sampling_config_init_v1(&sampling);
                    sampling.exponential_noise=sampling_noise.data()+size_t(frame)*8*8197;sampling.noise_count=8*8197;int32_t stop=-1;
                    if(mulacover_sample_frame_v1(c,&sampling,dump,&result,&stop,&error)||stop)throw std::runtime_error(std::string("Sampling parity failed: ")+error.message);
                }
                if(result.cache_position!=prompt+frame)throw std::runtime_error("Wrong cache position");
                if(!std::equal(result.codes,result.codes+8,teacher.data()+frame*8))throw std::runtime_error("Native codes differ from reference sequence");
                if(!replay)first_logits.insert(first_logits.end(),result.logits,result.logits+8*8197);
                else if(!std::equal(result.logits,result.logits+8*8197,first_logits.data()+frame*8*8197))throw std::runtime_error("Cache reset/replay logits differ");
                if(frame==0) {
                    mulacover_frame_v1 probe{};probe.size=sizeof(probe);
                    int32_t bad[8]={-1,0,0,0,0,0,0,0};
                    if(mulacover_teacher_frame_v1(c,bad,nullptr,&probe,&error)!=MULACOVER_INVALID_ARGUMENT_V1||probe.logits||probe.raw_logits)
                        throw std::runtime_error("Invalid teacher code was accepted");
                    if(mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&result,&error)!=MULACOVER_INVALID_ARGUMENT_V1)
                        throw std::runtime_error("Occupied frame output was accepted");
                }
                if(replay!=1||frame!=frames-1)mulacover_frame_free_v1(&result);
                std::cout<<"replay "<<replay<<", frame "<<frame<<" complete\n"<<std::flush;
            }
            if(config.cache_capacity==prompt+frames-1) {
                mulacover_frame_v1 probe{};probe.size=sizeof(probe);
                if(mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&probe,&error)!=MULACOVER_INVALID_ARGUMENT_V1||probe.logits||probe.raw_logits)
                    throw std::runtime_error("Cache exhaustion did not preserve state/output");
            }
        }
        mulacover_memory_v1 memory{};memory.size=sizeof(memory);if(mulacover_get_memory_v1(c,&memory,&error))throw std::runtime_error(error.message);
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream stats(output/"stats.json");stats<<"{\"weight_bytes\":"<<memory.weight_bytes<<",\"cache_bytes\":"<<memory.cache_bytes<<",\"graph_bytes\":"<<memory.graph_bytes
            <<",\"cache_capacity\":"<<config.cache_capacity<<",\"elapsed_seconds\":"<<elapsed<<",\"replay_bit_exact\":true,\"cache_tail_zero\":true,\"ggml_commit\":\""<<ggml_commit()
            <<"\",\"ggml_version\":\""<<ggml_version()<<"\"}\n";
        if(args.count("--check-condition-reuse")){
            if(args.at("--check-condition-reuse")!="1")throw std::runtime_error("Condition reuse check requires 1");
            mulacover_frame_free_v1(&result);const float original=piano[0];piano[0]=original==1?0:1;
            if(mulacover_session_begin_v1(c,&condition,&error)||mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&result,&error))throw std::runtime_error(error.message);
            if(std::equal(result.logits,result.logits+8*8197,first_logits.begin()))throw std::runtime_error("Changed score was not observed");
            std::vector<float> changed(result.logits,result.logits+8*8197);mulacover_frame_free_v1(&result);
            if(mulacover_session_reset_v1(c,&error)||mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&result,&error)||!std::equal(changed.begin(),changed.end(),result.logits))throw std::runtime_error("Changed-condition reset differs");
            mulacover_frame_free_v1(&result);piano[0]=original;
            if(mulacover_session_begin_v1(c,&condition,&error)||mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&result,&error)||!std::equal(first_logits.begin(),first_logits.begin()+8*8197,result.logits))throw std::runtime_error("Restored condition differs");
            mulacover_frame_free_v1(&result);const float original_style=qwen[0];qwen[0]+=1;
            if(mulacover_session_begin_v1(c,&condition,&error)||mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&result,&error))throw std::runtime_error(error.message);
            if(std::equal(first_logits.begin(),first_logits.begin()+8*8197,result.logits))throw std::runtime_error("Changed style was not observed");
            mulacover_frame_free_v1(&result);qwen[0]=original_style;
            // Reuse the same model/workspaces with different symbolic shapes and
            // CFG branch count, then restore the original cold result.
            auto short_condition=condition;uint8_t visible=0;short_condition.symbolic_frames=1;
            short_condition.context_mask=&visible;short_condition.guidance=1;
            if(mulacover_session_begin_v1(c,&short_condition,&error)||mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&result,&error)||result.branches!=1)throw std::runtime_error("Changed shape/branch begin failed");
            std::vector<float> short_logits(result.logits,result.logits+8*8197);mulacover_frame_free_v1(&result);
            if(mulacover_session_reset_v1(c,&error)||mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&result,&error)||!std::equal(short_logits.begin(),short_logits.end(),result.logits))throw std::runtime_error("Changed-shape reset differs");
            mulacover_frame_free_v1(&result);
            if(mulacover_session_begin_v1(c,&condition,&error)||mulacover_teacher_frame_v1(c,teacher.data(),nullptr,&result,&error)||!std::equal(first_logits.begin(),first_logits.begin()+8*8197,result.logits))throw std::runtime_error("Restored shape/branches differ");
            std::cout<<"condition replacement/reset/restoration exact\n";
            std::cout<<"style/shape/branch replacement and restoration exact\n";
        }
        const float saved=result.logits[0];mulacover_destroy_v1(c);c=nullptr;
        if(!std::isfinite(result.logits[0])||result.logits[0]!=saved)throw std::runtime_error("Frame result did not survive context destruction");
        mulacover_frame_free_v1(&result);mulacover_frame_free_v1(&result);return 0;
    }catch(const std::exception & ex){std::cerr<<ex.what()<<'\n';mulacover_frame_free_v1(&result);mulacover_destroy_v1(c);return 1;}
}

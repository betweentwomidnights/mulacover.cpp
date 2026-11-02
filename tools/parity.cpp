#include "mulacover.h"
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
        for(auto & pair:args)if(pair.first!="--model"&&pair.first!="--input"&&pair.first!="--output"&&pair.first!="--prompt-frames"&&pair.first!="--symbolic-frames"&&
            pair.first!="--qwen-index"&&pair.first!="--muq-index"&&pair.first!="--guidance"&&pair.first!="--backend"&&pair.first!="--capacity"&&pair.first!="--frames")throw std::runtime_error("Unknown option: "+pair.first);
        const std::filesystem::path input(args.at("--input")),output(args.at("--output"));
        int prompt=std::stoi(args.at("--prompt-frames")),symbolic=std::stoi(args.at("--symbolic-frames"));
        int frames=args.count("--frames")?std::stoi(args.at("--frames")):3;
        if(prompt<1||prompt>8192||symbolic<1||symbolic>5000||frames<1||frames>8192)throw std::runtime_error("Invalid test dimensions");
        auto tokens=read<int32_t>(input/"tokens.i32",size_t(prompt)*9);auto mask=read<uint8_t>(input/"mask.u8",size_t(prompt)*9);
        auto qwen=read<float>(input/"qwen.f32",1024),muq=read<float>(input/"muq.f32",512);
        auto piano=read<float>(input/"piano.f32",size_t(symbolic)*256),drum=read<float>(input/"drum.f32",size_t(symbolic)*128),chord=read<float>(input/"chord.f32",size_t(symbolic)*12);
        auto context_mask=read<uint8_t>(input/"context-mask.u8",size_t(symbolic));auto teacher=read<int32_t>(input/"teacher.i32",size_t(frames)*8);
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
        std::filesystem::create_directories(output);const auto start=std::chrono::steady_clock::now();
        std::vector<float> first_logits;
        for(int replay=0;replay<2;++replay) {
            if(replay&&mulacover_session_reset_v1(c,&error))throw std::runtime_error(error.message);
            for(int frame=0;frame<frames;++frame) {
                const auto path=replay?(output/"replay").string():output.string();
                if(mulacover_teacher_frame_v1(c,teacher.data()+frame*8,path.c_str(),&result,&error))throw std::runtime_error(error.message);
                if(result.cache_position!=prompt+frame)throw std::runtime_error("Wrong cache position");
                if(!std::equal(result.codes,result.codes+8,teacher.data()+frame*8))throw std::runtime_error("Teacher codes changed");
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
        }
        mulacover_memory_v1 memory{};memory.size=sizeof(memory);if(mulacover_get_memory_v1(c,&memory,&error))throw std::runtime_error(error.message);
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream stats(output/"stats.json");stats<<"{\"weight_bytes\":"<<memory.weight_bytes<<",\"cache_bytes\":"<<memory.cache_bytes<<",\"graph_bytes\":"<<memory.graph_bytes
            <<",\"cache_capacity\":"<<config.cache_capacity<<",\"elapsed_seconds\":"<<elapsed<<",\"replay_bit_exact\":true,\"cache_tail_zero\":true,\"ggml_commit\":\""<<ggml_commit()
            <<"\",\"ggml_version\":\""<<ggml_version()<<"\"}\n";
        const float saved=result.logits[0];mulacover_destroy_v1(c);c=nullptr;
        if(!std::isfinite(result.logits[0])||result.logits[0]!=saved)throw std::runtime_error("Frame result did not survive context destruction");
        mulacover_frame_free_v1(&result);mulacover_frame_free_v1(&result);return 0;
    }catch(const std::exception & ex){std::cerr<<ex.what()<<'\n';mulacover_frame_free_v1(&result);mulacover_destroy_v1(c);return 1;}
}

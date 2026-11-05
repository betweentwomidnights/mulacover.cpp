#pragma once
#include "mulacover_conditioning.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace conditioning_tool {
inline std::string text(const std::string & file){std::ifstream f(std::filesystem::u8path(file),std::ios::binary);if(!f)throw std::runtime_error("Cannot open UTF-8 text file");return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
template<class T>void write(const std::filesystem::path & path,const T * data,size_t count){std::ofstream f(path,std::ios::binary);if(!f.write(reinterpret_cast<const char *>(data),std::streamsize(count*sizeof(T))))throw std::runtime_error("Cannot write conditioning output");}
inline void MULACOVER_CALL progress(void *,int64_t done,int64_t total){if(!done||done==total||done%7==0)std::cout<<"style "<<done<<"/"<<total<<std::endl;}
struct Prepared {
    mulacover_prepared_condition_v1 value{};
    mulacover_conditioning_context * context=nullptr;
    std::string lyrics,tags;
    Prepared(){value.size=sizeof(value);mulacover_prepared_condition_init_v1(&value);}
    Prepared(const Prepared &)=delete;
    ~Prepared(){mulacover_prepared_condition_free_v1(&value);mulacover_conditioning_destroy_v1(context);}
    int prepare(const std::map<std::string,std::string> & a,int64_t max_frames,mulacover_job_cancel_callback_v1 cancel,void * user,mulacover_error_v1 & e){
        auto get=[&](const char * key,const char * fallback){auto it=a.find(key);return it==a.end()?std::string(fallback):it->second;};
        lyrics=text(a.at("--lyrics-file"));tags=text(a.at("--tags-file"));
        auto * api=mulacover_get_conditioning_api(1);if(!api||api->size<MULACOVER_CONDITIONING_API_V1_SIZE)throw std::runtime_error("Conditioning capability unavailable");
        int status=api->create(a.at("--tokenizers-model").c_str(),&context,&e);if(status)return status;
        mulacover_conditioning_request_v1 r{};r.size=sizeof(r);api->request_init(&r);r.lyrics=lyrics.c_str();r.tags=tags.c_str();r.style_model=a.count("--style-model")?a.at("--style-model").c_str():nullptr;
        const auto backend=get("--backend","cuda");r.backend=backend.c_str();r.threads=std::stoi(get("--threads","4"));r.guidance=std::stof(get("--guidance","1.5"));r.max_audio_frames=max_frames;
        const auto score=get("--score-mode","none");if(score!="none"&&score!="provided")throw std::runtime_error("Score mode must be none or provided");r.score_mode=score=="provided"?1:0;
        r.symbolic_frames=std::stoll(get("--symbolic-frames",r.score_mode?"0":"1"));
        r.melody_midi=a.count("--melody-midi")?a.at("--melody-midi").c_str():nullptr;r.chord_midi=a.count("--chord-midi")?a.at("--chord-midi").c_str():nullptr;r.drum_midi=a.count("--drum-midi")?a.at("--drum-midi").c_str():nullptr;
        r.diagnostic_directory=a.count("--diagnostic-directory")?a.at("--diagnostic-directory").c_str():nullptr;r.should_cancel=cancel;r.callback_user=user;r.on_style_progress=progress;
        status=api->prepare(context,&r,&value,&e);api->destroy(context);context=nullptr;return status;
    }
    void save(const std::filesystem::path & output){
        const auto & c=value.condition;std::filesystem::create_directories(output);
        write(output/"tokens.i32",value.tokens,size_t(c.prompt_frames)*9);write(output/"mask.u8",value.tokens_mask,size_t(c.prompt_frames)*9);write(output/"qwen.f32",value.qwen,1024);write(output/"muq.f32",value.muq,512);
        write(output/"piano.f32",value.pianoroll,size_t(c.symbolic_frames)*256);write(output/"drum.f32",value.drum_pianoroll,size_t(c.symbolic_frames)*128);write(output/"chord.f32",value.chord,size_t(c.symbolic_frames)*12);write(output/"context-mask.u8",value.context_mask,size_t(c.symbolic_frames));
        write(output/"style-ids.i32",value.style_tokens,size_t(value.style_token_count));write(output/"normalized-lyrics.txt",value.normalized_lyrics,std::strlen(value.normalized_lyrics));write(output/"normalized-tags.txt",value.normalized_tags,std::strlen(value.normalized_tags));
        std::ofstream f(output/"run.json");f<<"{\"native_conditioning\":true,\"prompt_frames\":"<<c.prompt_frames<<",\"symbolic_frames\":"<<c.symbolic_frames<<",\"qwen_index\":"<<c.qwen_index<<",\"muq_index\":"<<c.muq_index<<",\"style_tokens\":"<<value.style_token_count<<",\"style_weight_bytes\":"<<value.style_weight_bytes<<",\"style_graph_bytes\":"<<value.style_graph_bytes<<"}\n";if(!f)throw std::runtime_error("Cannot write conditioning report");
    }
};
}

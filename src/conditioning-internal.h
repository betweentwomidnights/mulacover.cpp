#pragma once
#include "mulacover_conditioning.h"
#include "internal.h"
#include <unordered_map>

namespace mula::frontend {
struct Tokenizer {
    std::unordered_map<std::string,int32_t> vocab,merges;
    std::vector<std::pair<std::string,int32_t>> specials;
    std::array<std::string,256> bytes;
    bool nfc=false,ignore_merges=false,three_digits=false;
    void load(const gguf_context *,const std::string &);
    std::vector<int32_t> encode(const std::string &,bool eos) const;
    void piece(const std::string &,std::vector<int32_t> &) const;
};
struct TextAssets {
    Tokenizer prompt,style;
    std::map<std::string,int32_t> config;
    explicit TextAssets(const char *);
};
std::string lower(const std::string &);
std::string strip(const std::string &);
std::vector<mulacover_note_v1> midi(const char *,bool);
void validate_notes(std::vector<mulacover_note_v1> &,bool);
std::vector<float> style(const char *,const char *,int,const std::vector<int32_t> &,const char *,
                         const mulacover_conditioning_request_v1 &,uint64_t &,uint64_t &);
void checkpoint(const mulacover_conditioning_request_v1 &);
void progress(const mulacover_conditioning_request_v1 &,int64_t,int64_t);
}
struct mulacover_conditioning_context {mula::frontend::TextAssets assets;explicit mulacover_conditioning_context(const char * p):assets(p){}};

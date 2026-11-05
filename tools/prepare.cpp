#include "conditioning-args.h"
#include <csignal>
#include <set>
static volatile std::sig_atomic_t interrupted=0;
static void interrupt(int){interrupted=1;}
static int32_t MULACOVER_CALL cancel(void *){return interrupted?1:0;}
int main(int argc,char ** argv){try{
    if(argc==1)throw std::runtime_error("Usage: mulacover-prepare --tokenizers-model GGUF --style-model GGUF --lyrics-file UTF8 --tags-file UTF8 --output FRESH_DIR [--backend cpu|cuda --score-mode none|provided --symbolic-frames N --melody-midi MIDI --chord-midi MIDI --drum-midi MIDI --max-audio-frames N]");
    const std::set<std::string> known={"--tokenizers-model","--style-model","--lyrics-file","--tags-file","--output","--backend","--threads","--guidance","--score-mode","--symbolic-frames","--melody-midi","--chord-midi","--drum-midi","--max-audio-frames","--diagnostic-directory"};std::map<std::string,std::string> a;
    for(int i=1;i<argc;i+=2)if(i+1>=argc||!known.count(argv[i])||!a.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("Unknown/missing/duplicate option");
    const std::filesystem::path output(a.at("--output"));if(std::filesystem::exists(output))throw std::runtime_error("Use a fresh output directory");
    std::signal(SIGINT,interrupt);conditioning_tool::Prepared prepared;mulacover_error_v1 e{};e.size=sizeof(e);const int status=prepared.prepare(a,a.count("--max-audio-frames")?std::stoll(a.at("--max-audio-frames")):375,cancel,nullptr,e);
    if(status){std::cerr<<e.message<<std::endl;return status;}prepared.save(output);std::cout<<"Prepared "<<prepared.value.condition.prompt_frames<<" prompt and "<<prepared.value.condition.symbolic_frames<<" symbolic frames"<<std::endl;return 0;
}catch(const std::exception & e){std::cerr<<e.what()<<std::endl;return 1;}}

#include "server/service.h"
#include "mulacover_ui_html.h"
#include <atomic>
#include <csignal>
#include <iostream>
#include <set>

namespace {
volatile std::sig_atomic_t interrupted=0;
void stop(int){interrupted=1;}
}
int main(int argc,char ** argv){try{
    using namespace mulacover::server;
    Config config;config.backend=MULACOVER_DEFAULT_BACKEND;
    bool print_props=false;
    for(int i=1;i<argc;++i){
        const std::string key=argv[i];
        if(key=="--version"){std::cout<<"mulacover-server "<<MULACOVER_VERSION<<'\n';return 0;}
        if(key=="--props"){print_props=true;continue;}
        if(key=="--help"){
            std::cout<<"mulacover-server "<<MULACOVER_VERSION<<"\n"
              "  --model-dir DIR         generator, tokenizers and style GGUFs\n"
              "  --codec-model-dir DIR   encoder, flow and SQ GGUFs\n"
              "  --backend cpu|cuda|vulkan|metal\n"
              "  --port 8010 --threads 4 --max-cache-capacity 1536\n"
              "  --transcription-port 8011  local Yue2/SheetSage2 service (0 disables)\n"
              "  --allow-origin URL      optional exact downstream browser origin\n"
              "  --props | --version     model-free installer probes\n"
              "Listens on 127.0.0.1. Open / for the embedded continuation UI.\n";return 0;
        }
        if(i+1==argc)throw std::invalid_argument("Missing value for "+key);
        const std::string value=argv[++i];
        auto integer=[&]{size_t used=0;int n=std::stoi(value,&used);if(used!=value.size())throw std::invalid_argument("Invalid integer for "+key);return n;};
        if(key=="--model-dir")config.model_dir=value;
        else if(key=="--codec-model-dir")config.codec_model_dir=value;
        else if(key=="--backend")config.backend=value;
        else if(key=="--port")config.port=integer();
        else if(key=="--transcription-port")config.transcription_port=integer();
        else if(key=="--threads")config.threads=integer();
        else if(key=="--max-cache-capacity")config.max_cache_capacity=integer();
        else if(key=="--allow-origin")config.allow_origin=value;
        else throw std::invalid_argument("Unknown option: "+key);
    }
    if(!std::set<std::string>{"cpu","cuda","vulkan","metal"}.count(config.backend)||config.port<1||config.port>65535||config.threads<1||config.threads>1024||config.max_cache_capacity<256||config.max_cache_capacity>8192)
        throw std::invalid_argument("Invalid backend, port, threads or cache budget");
    if(config.allow_origin.find_first_of("\r\n")!=std::string::npos||config.allow_origin=="*")throw std::invalid_argument("--allow-origin must be an exact origin");
    if(config.transcription_port<0||config.transcription_port>65535||config.transcription_port==config.port)throw std::invalid_argument("Invalid transcription port");
    config.resolve_models();
    if(print_props){std::cout<<props(config)<<'\n';return 0;}
    std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
    Service service(config);
    HttpServerOptions options;options.port=config.port;options.max_request_body_bytes=128ULL*1024*1024;
    std::atomic<bool> stopped{false};
    std::thread signals([&]{while(!stopped.load()){if(interrupted){stopped=true;break;}std::this_thread::sleep_for(std::chrono::milliseconds(50));}});
    try{
        serve_http(options,[&](const HttpRequest & req){
            if(req.path=="/"){
                if(req.method!="GET")return error_response(405,"Method not allowed","method_not_allowed");
                return HttpResponse{200,"text/html; charset=utf-8",std::string(reinterpret_cast<const char *>(mulacover::ui::mulacover_ui_html),mulacover::ui::mulacover_ui_html_size),{{"Cache-Control","no-store"},{"Content-Security-Policy","default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'"}}};
            }
            return service.handle(req);
        },stopped);
    }catch(...){stopped=true;signals.join();service.shutdown();throw;}
    stopped=true;signals.join();service.shutdown();return 0;
}catch(const std::exception & e){std::cerr<<"mulacover-server: "<<e.what()<<'\n';return 1;}}

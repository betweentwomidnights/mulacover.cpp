#include "server/service.h"
#include <atomic>
#include <iostream>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error("check failed: " #x);}while(0)
using namespace mulacover::server;
std::string wave(){
    std::string s;auto u32=[&](uint32_t n){for(int i=0;i<4;++i)s.push_back(char(n>>(i*8)));};auto u16=[&](int n){s.push_back(char(n));s.push_back(char(n>>8));};
    s+="RIFF";u32(36+9600);s+="WAVEfmt ";u32(16);u16(1);u16(1);u32(48000);u32(96000);u16(2);u16(16);s+="data";u32(9600);s.append(9600,'\0');return s;
}
HttpRequest submission(std::string options="{}"){
    HttpRequest r;r.method="POST";r.path="/continue";r.headers["content-type"]="multipart/form-data; boundary=test";
    r.body="--test\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"../../ignored.wav\"\r\n\r\n"+wave()+"\r\n--test\r\nContent-Disposition: form-data; name=\"options\"\r\n\r\n"+options+"\r\n--test--\r\n";return r;
}
HttpResponse route(Service & s,const std::string & path,const std::string & method="GET"){HttpRequest r;r.method=method;r.path=path;return s.handle(r);}
json::Value terminal(Service & s,const std::string & id){for(int i=0;i<400;++i){auto r=route(s,"/poll_status/"+id);CHECK(r.status==200);auto j=json::parse(r.body);if(!json::boolean(j,"generation_in_progress",true))return j;std::this_thread::sleep_for(std::chrono::milliseconds(5));}throw std::runtime_error("worker timeout");}
int main(){try{
    Config config;config.backend="cpu";config.resolve_models();
    Input opening;opening.audio=wave_io::read_wave_bytes(wave());opening.options=json::parse("{\"audio_context_seconds\":1}");
    auto short_plan=plan_audio_input(config,opening);CHECK(short_plan.context_start_sample==0&&short_plan.cursor_sample==3840&&short_plan.guide_frames==1);
    opening.audio.samples.resize(96000);auto one_second=plan_audio_input(config,opening);CHECK(one_second.context_start_sample==0&&one_second.cursor_sample==46080&&one_second.guide_frames==12);
    opening.options=json::parse("{\"audio_context_seconds\":1,\"transcribe\":true,\"transcription_scope\":\"source\",\"match_source_duration\":true}");
    auto remix=plan_audio_input(config,opening);CHECK(remix.context_start_sample==0&&remix.guide_frames==12&&remix.new_frames==13); // 0.96 + 1.04 = 2s
    opening.options=json::parse("{\"audio_context_seconds\":0.08,\"transcribe\":true,\"transcription_scope\":\"source\",\"match_source_duration\":true}");
    for(int n=1;n<=40;++n){opening.audio.samples.resize(size_t(n+1)*3840+17);auto p=plan_audio_input(config,opening);CHECK(p.guide_frames==1&&p.new_frames==n&&p.context_start_sample==0);}
    opening.audio.samples.resize(4800);opening.options=json::parse("{\"audio_context_seconds\":0.08,\"transcribe\":true,\"transcription_scope\":\"source\",\"match_source_duration\":true}");
    bool no_room=false;try{plan_audio_input(config,opening);}catch(const std::invalid_argument &){no_room=true;}CHECK(no_room);
    std::atomic<bool> entered{false},finish{false};std::atomic<int> active{0},maximum{0};
    Service s(config,[&](const Config &,const Input & in,const Progress & progress,const Cancel & cancel){
        ++active;maximum.store(std::max(maximum.load(),active.load()));struct Exit{std::atomic<int>& a;~Exit(){--a;}} exit{active};
        if(json::string(in.options,"tags","")=="fail")throw std::runtime_error("injected failure");
        CHECK(json::u64(in.options,"seed",UINT64_MAX)==UINT64_MAX);
        progress(.7,"sampling",1,2);progress(.4,"sampling",1,2);entered=true;
        while(!finish&&!cancel())std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Outcome out;out.wav="test-wav";out.meta.object["seed"]=json::from_string("18446744073709551615");return out;
    });
    CHECK(route(s,"/health").status==200);CHECK(route(s,"/continue").status==405);
    auto bad=submission("{\"seed\":\"18446744073709551616\"}");CHECK(s.handle(bad).status==400);
    for(const auto * v:{"{\"unknown\":1}","{\"cursor_sample\":99999999}","{\"score_mode\":\"provided\"}","{\"rng\":\"torch-cuda\"}","{\"seed\":\"+1\"}","{\"seed\":1.5}","{\"natural_max_seconds\":655.36}","{\"guide_samples\":-1}","{\"top_k\":0}","{\"lyrics\":\"a\\u0000b\"}","{\"seed\":1,\"seed\":2}"})CHECK(s.handle(submission(v)).status==400);
    auto invalid=submission();invalid.body="bad";CHECK(s.handle(invalid).status==400);
    CHECK(s.handle(submission("{\"audio_context_seconds\":1,\"cursor_sample\":1}")).status==400);
    CHECK(s.handle(submission("{\"audio_context_seconds\":0.01}")).status==400);
    CHECK(s.handle(submission("{\"transcription_scope\":\"invalid\"}")).status==400);
    auto origin=submission();origin.headers["origin"]="https://untrusted.example";CHECK(s.handle(origin).status==403);
    auto r=s.handle(submission("{\"seed\":\"18446744073709551615\"}"));CHECK(r.status==202);auto id=json::string(json::parse(r.body),"session_id");
    for(int i=0;i<400&&!entered;++i)std::this_thread::sleep_for(std::chrono::milliseconds(5));CHECK(entered);
    auto running=json::parse(route(s,"/poll_status/"+id).body);CHECK(json::number(running,"progress",0)==.7);
    CHECK(s.handle(submission()).status==409);CHECK(route(s,"/unload","POST").status==409);CHECK(route(s,"/audio/"+id).status==409);
    CHECK(route(s,"/cancel/"+id,"POST").status==200);CHECK(json::string(terminal(s,id),"status")=="cancelled");CHECK(route(s,"/audio/"+id).status==409);
    CHECK(route(s,"/release/"+id,"POST").status==200);CHECK(route(s,"/poll_status/"+id).status==404);
    r=s.handle(submission("{\"tags\":\"fail\"}"));CHECK(r.status==202);id=json::string(json::parse(r.body),"session_id");CHECK(json::string(terminal(s,id),"status")=="failed");
    finish=true;r=s.handle(submission("{\"audio_context_seconds\":1}"));CHECK(r.status==202);id=json::string(json::parse(r.body),"session_id");auto done=terminal(s,id);CHECK(json::string(done,"status")=="completed");CHECK(route(s,json::string(done,"audio_url")).body=="test-wav");CHECK(route(s,"/full_audio/"+id).status==404);
    CHECK(route(s,"/unload","POST").status==200);CHECK(route(s,"/audio/"+id).status==404);CHECK(maximum==1);
    finish=false;entered=false;r=s.handle(submission());CHECK(r.status==202);s.shutdown();CHECK(active==0);s.shutdown();
    std::cout<<"server admission, validation, cancellation, replay lifecycle, download and shutdown passed\n";return 0;
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}}

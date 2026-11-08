#include "server/transcription.h"
#include "server/base64.h"
#include <iostream>
#include <limits>
#define CHECK(x) do{if(!(x))throw std::runtime_error("check failed: " #x);}while(0)
using namespace mulacover::server;
int main(){try{
    Config c;wave_io::Wave a;a.rate=48000;a.channels=1;a.samples.assign(4800,0.25f);
    // NaNs outside the selected opening prove the provider sees only context.
    std::fill(a.samples.begin()+3840,a.samples.end(),std::numeric_limits<float>::quiet_NaN());
    const std::string fake_midi="MThd0123456789";int posts=0,polls=0,consumed=0,cancels=0;bool cancelled=false;
    auto completed=[&]{auto v=json::object();v.object["status"]=json::from_string("completed");auto files=json::object();auto b64=base64_encode(reinterpret_cast<const uint8_t *>(fake_midi.data()),fake_midi.size());for(const char * name:{"melody.mid","melody_vocal.mid","melody_instrumental.mid","chords.mid"})files.object[name]=json::from_string(b64);v.object["midi_files"]=files;return json_response(json::stringify(v));};
    Exchange exchange=[&](const HttpRequest & r){
        if(r.path=="/transcribe"){
            ++posts;CHECK(r.method=="POST");auto p=json::parse(r.body);CHECK(!json::boolean(p,"keep_models",true)&&json::string(p,"pickup")=="keep");auto b=base64_decode(json::string(p,"audio_data"));CHECK(b.size()==44+3840*4);auto w=wave_io::read_wave_bytes({reinterpret_cast<const char *>(b.data()),b.size()});CHECK(w.rate==48000&&w.samples.size()==3840&&w.samples.back()==.25f);return json_response("{\"session_id\":\"owned-session\"}");
        }
        CHECK(r.path.find("owned-session")!=std::string::npos);
        if(r.path=="/cancel/owned-session"){++cancels;return json_response("{}");}
        if(r.path.find("consume=1")!=std::string::npos){++consumed;return completed();}
        ++polls;if(cancelled)return json_response("{\"status\":\"cancelled\"}");return completed();
    };
    auto score=transcribe_score(c,a,0,3840,"vocal",[](double,const std::string &,int64_t,int64_t){},[]{return false;},exchange);
    CHECK(score.melody==fake_midi&&score.chords==fake_midi&&posts==1&&consumed==1&&cancels==0);
    bool failed=false;try{transcribe_score(c,a,0,4800,"combined",[](double,const std::string &,int64_t,int64_t){},[]{return false;},exchange);}catch(const std::invalid_argument &){failed=true;}CHECK(failed&&posts==1);
    // Cancel after submission: cancel and await/consume this provider session;
    // no global /unload or unrelated-session action is permitted.
    int checks=0;failed=false;try{transcribe_score(c,a,0,3840,"combined",[](double,const std::string &,int64_t,int64_t){},[&]{cancelled=++checks>1;return cancelled;},exchange);}catch(const std::runtime_error &){failed=true;}
    CHECK(failed&&cancels==1&&consumed==2);
    std::cout<<"SheetSage2 opening-only PCM, lane transport, model release and owned-session cancellation passed\n";return 0;
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}}

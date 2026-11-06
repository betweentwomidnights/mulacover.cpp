#include "transcription.h"
#include "base64.h"
#include <chrono>
#include <cmath>
#include <thread>
namespace mulacover::server {
namespace {
std::string source_wav(const wave_io::Wave & a,int64_t start,int64_t count){
    if(start<0||count<=0||start+count>int64_t(a.samples.size()/a.channels)||uint64_t(count)>(UINT32_MAX-36)/4)throw std::invalid_argument("Invalid transcription audio window");
    std::string out;out.reserve(size_t(count)*4+44);
    auto u32=[&](uint32_t n){for(int i=0;i<4;++i)out.push_back(char(n>>(8*i)));};auto u16=[&](int n){out.push_back(char(n));out.push_back(char(n>>8));};
    out+="RIFF";u32(uint32_t(36+4*count));out+="WAVEfmt ";u32(16);u16(3);u16(1);u32(a.rate);u32(a.rate*4);u16(4);u16(32);out+="data";u32(uint32_t(count*4));
    for(int64_t i=start;i<start+count;++i){float x=a.samples[size_t(i)*a.channels];if(a.channels==2)x=.5f*(x+a.samples[size_t(i)*2+1]);if(!std::isfinite(x))throw std::invalid_argument("Nonfinite transcription context");uint32_t bits;std::memcpy(&bits,&x,4);u32(bits);}
    return out;
}
std::string midi(const json::Value & response,const char * name,bool required){
    const auto * files=response.find("midi_files");
    if(!files||files->type!=json::Type::object)throw std::runtime_error("SheetSage2 returned no MIDI lanes");
    auto text=json::string(*files,name,"");if(text.empty()){if(required)throw std::runtime_error("SheetSage2 returned no selected melody lane");return {};}
    if(text.size()>6*1024*1024)throw std::runtime_error("SheetSage2 MIDI exceeds limit");auto bytes=base64_decode(text);
    if(bytes.size()>4*1024*1024)throw std::runtime_error("SheetSage2 MIDI exceeds limit");
    if(bytes.size()<14||std::memcmp(bytes.data(),"MThd",4))throw std::runtime_error("SheetSage2 returned invalid MIDI");
    return {reinterpret_cast<const char *>(bytes.data()),bytes.size()};
}
}
TranscribedScore transcribe_score(const Config & config,const wave_io::Wave & audio,int64_t start,int64_t count,
    const std::string & lane,const Progress & progress,const Cancel & cancel,Exchange exchange){
    if(lane!="combined"&&lane!="vocal"&&lane!="instrumental")throw std::invalid_argument("Unknown transcription melody lane");
    if(!exchange)exchange=[&](const HttpRequest & r){return request_loopback(config.transcription_port,r);};
    auto call=[&](const std::string & path,const std::string & method,const std::string & body=""){
        HttpRequest r;r.path=path;r.method=method;r.body=body;auto response=exchange(r);
        if(response.status<200||response.status>=300)throw std::runtime_error("SheetSage2 request failed (HTTP "+std::to_string(response.status)+"): "+response.body.substr(0,1024));
        return json::parse(response.body);
    };
    if(cancel())throw std::runtime_error("Transcription cancelled");progress(0,"transcribing audio",0,0);
    const auto bytes=source_wav(audio,start,count);auto payload=json::object();payload.object["audio_data"]=json::from_string(base64_encode(reinterpret_cast<const uint8_t *>(bytes.data()),bytes.size()));
    payload.object["transcription_mode"]=json::from_string("full");payload.object["pickup"]=json::from_string("keep");payload.object["keep_models"]=json::from_bool(false);
    auto submitted=call("/transcribe","POST",json::stringify(payload));const auto id=json::string(submitted,"session_id","");
    if(id.empty()||id.size()>128||id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")!=std::string::npos)throw std::runtime_error("Invalid SheetSage2 session ID");
    bool terminal=false,cancel_sent=false;
    // On any failure, best-effort cancel only the session we created. No global
    // unload is issued: it could affect another application's work.
    struct Cleanup {Exchange & exchange;const std::string & id;bool & terminal;~Cleanup(){if(!terminal)try{HttpRequest r;r.method="POST";r.path="/cancel/"+id;exchange(r);}catch(...){}}} cleanup{exchange,id,terminal};
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::minutes(30);
    for(;;){
        if(cancel()&&!cancel_sent){call("/cancel/"+id,"POST","{}");cancel_sent=true;}
        auto status=call("/poll_status/"+id,"GET");const auto state=json::string(status,"status","");
        if(state=="completed"||state=="cancelled"||state=="failed"){
            terminal=true;try{call("/poll_status/"+id+"?consume=1","GET");}catch(...){}
            if(cancel_sent||cancel()||state=="cancelled")throw std::runtime_error("Transcription cancelled");
            if(state!="completed")throw std::runtime_error("SheetSage2 transcription failed: "+json::string(status,"error","Unknown error"));
            TranscribedScore score;score.melody=midi(status,lane=="vocal"?"melody_vocal.mid":lane=="instrumental"?"melody_instrumental.mid":"melody.mid",true);score.chords=midi(status,"chords.mid",false);
            score.metadata.object["engine"]=json::from_string("yue2/SheetSage2");score.metadata.object["lane"]=json::from_string(lane);
            score.metadata.object["source_start_sample"]=json::from_integer(start);score.metadata.object["source_samples"]=json::from_integer(count);
            score.metadata.object["pickup"]=json::from_string("keep");score.metadata.object["keep_models"]=json::from_bool(false);
            progress(1,"transcription complete",1,1);return score;
        }
        if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("SheetSage2 transcription timed out");
        const auto step=json::number(status,"step",0),total=json::number(status,"total_steps",0);
        progress(std::clamp(json::number(status,"progress",0)/100.0,0.0,.99),"transcribing audio",int64_t(step),int64_t(total));
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
}
}

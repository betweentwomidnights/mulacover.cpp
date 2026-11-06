#include "conditioning-internal.h"
#include "backend-select.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace mula::frontend {
struct Cancelled{};
void checkpoint(const mulacover_conditioning_request_v1 & r){if(r.should_cancel&&r.should_cancel(r.callback_user))throw Cancelled{};}
void progress(const mulacover_conditioning_request_v1 & r,int64_t done,int64_t total){checkpoint(r);if(r.on_style_progress)r.on_style_progress(r.callback_user,done,total);checkpoint(r);}
template<class T>void init(T * p){if(!p||p->size<sizeof(*p))return;auto n=p->size;std::memset(p,0,sizeof(*p));p->size=n;}
template<class F>int guard(mulacover_error_v1 * e,F && f){int status=0;std::string message;try{f();}catch(const Cancelled &){status=3;message="Conditioning cancelled";}catch(const std::invalid_argument & x){status=1;message=x.what();}catch(const std::exception & x){status=2;message=x.what();}catch(...){status=2;message="Unknown conditioning failure";}if(e&&e->size>=sizeof(*e)){e->status=status;std::snprintf(e->message,sizeof(e->message),"%s",message.c_str());}return status;}
template<class T>std::unique_ptr<T[]> owned(const std::vector<T> & v){auto p=std::make_unique<T[]>(v.size());std::copy(v.begin(),v.end(),p.get());return p;}
static std::unique_ptr<char[]> owned(const std::string & v){auto p=std::make_unique<char[]>(v.size()+1);std::copy(v.begin(),v.end(),p.get());p[v.size()]=0;return p;}
}
using namespace mula::frontend;
extern "C" {
void MULACOVER_CALL mulacover_conditioning_request_init_v1(mulacover_conditioning_request_v1 * r){if(!r||r->size<sizeof(*r))return;init(r);r->lyrics="[Instrumental]";r->tags="";r->backend="cuda";r->threads=4;r->symbolic_frames=1;r->guidance=1.5f;r->max_audio_frames=375;}
void MULACOVER_CALL mulacover_prepared_condition_init_v1(mulacover_prepared_condition_v1 * r){init(r);if(r&&r->size>=sizeof(*r))r->condition.size=sizeof(r->condition);}
void MULACOVER_CALL mulacover_prepared_condition_free_v1(mulacover_prepared_condition_v1 * r){if(!r||r->size<sizeof(*r))return;delete[]r->tokens;delete[]r->tokens_mask;delete[]r->qwen;delete[]r->muq;delete[]r->pianoroll;delete[]r->drum_pianoroll;delete[]r->chord;delete[]r->context_mask;delete[]r->style_tokens;delete[]r->normalized_lyrics;delete[]r->normalized_tags;mulacover_prepared_condition_init_v1(r);}
int MULACOVER_CALL mulacover_conditioning_create_v1(const char * path,mulacover_conditioning_context ** out,mulacover_error_v1 * e){return guard(e,[&]{if(!out||*out)throw std::invalid_argument("Invalid/occupied conditioning context output");auto c=std::make_unique<mulacover_conditioning_context>(path);*out=c.release();});}
void MULACOVER_CALL mulacover_conditioning_destroy_v1(mulacover_conditioning_context * c){delete c;}
int MULACOVER_CALL mulacover_conditioning_prepare_v1(mulacover_conditioning_context * context,const mulacover_conditioning_request_v1 * request,mulacover_prepared_condition_v1 * output,mulacover_error_v1 * error){
    return guard(error,[&]{
        if(!context||!request||request->size<sizeof(*request)||!output||output->size<sizeof(*output)||output->tokens||output->tokens_mask||output->qwen||output->muq||output->pianoroll||output->drum_pianoroll||output->chord||output->context_mask||output->style_tokens||output->style_token_count||output->normalized_lyrics||output->normalized_tags||output->condition.tokens||output->condition.prompt_frames)throw std::invalid_argument("Invalid request/context or occupied conditioning output");
        const auto & r=*request;if(!r.lyrics||!r.tags||std::strlen(r.lyrics)>262144||std::strlen(r.tags)>262144||!mula::known_backend(r.backend)||r.threads<1||r.threads>1024||!std::isfinite(r.guidance)||r.guidance<=0||r.max_audio_frames<1||r.max_audio_frames>8192||r.symbolic_frames<0||r.symbolic_frames>5000||(r.score_mode!=0&&r.score_mode!=1))throw std::invalid_argument("Invalid conditioning configuration");
        for(auto pair:{std::make_pair(r.melody,r.melody_count),std::make_pair(r.chords,r.chord_count),std::make_pair(r.drums,r.drum_count)})if(pair.second<0||pair.second>1000000||(pair.second&&!pair.first))throw std::invalid_argument("Invalid note buffer");
        checkpoint(r);auto lyric=lower(r.lyrics);auto tags=lower(strip(r.tags));if(strip(lyric).empty())throw std::invalid_argument("Lyrics must not be empty");
        if(tags.size()>=11&&tags.compare(0,5,"<tag>")==0&&tags.compare(tags.size()-6,6,"</tag>")==0)tags=strip(tags.substr(5,tags.size()-11));
        auto & assets=context->assets;auto tag_ids=assets.prompt.encode(tags,false);auto lyric_ids=assets.prompt.encode(lyric,false);auto style_ids=assets.style.encode(tags,true);
        if(style_ids.size()>8192)throw std::invalid_argument("Qwen style exceeds 8192 tokens");
        std::vector<int32_t> prompt{assets.config.at("tag_start_id")};prompt.insert(prompt.end(),tag_ids.begin(),tag_ids.end());prompt.push_back(assets.config.at("tag_end_id"));
        const int64_t qi=int64_t(prompt.size())+1,mi=int64_t(prompt.size())+4;
        for(const char * k:{"qwen_start_id","empty_id","qwen_end_id","muq_start_id","empty_id","muq_end_id","text_bos_id"})prompt.push_back(assets.config.at(k));
        prompt.insert(prompt.end(),lyric_ids.begin(),lyric_ids.end());prompt.push_back(assets.config.at("text_eos_id"));
        if(int64_t(prompt.size())+r.max_audio_frames>8192)throw std::invalid_argument("Prompt and maximum audio exceed upstream context budget");
        std::vector<int32_t> tokens(prompt.size()*9,assets.config.at("empty_id"));std::vector<uint8_t> mask(tokens.size(),0);for(size_t i=0;i<prompt.size();++i){tokens[i*9+8]=prompt[i];mask[i*9+8]=1;}
        std::vector<mulacover_note_v1> melody,chords,drums;
        if(r.score_mode==1){
            const bool files=r.melody_midi||r.chord_midi||r.drum_midi;
            if(files){if(r.melody_count||r.chord_count||r.drum_count)throw std::invalid_argument("Choose MIDI or note arrays");melody=midi(r.melody_midi,false);if(r.chord_midi)chords=midi(r.chord_midi,false);if(r.drum_midi)drums=midi(r.drum_midi,true);}
            else{if(r.melody_count)melody.assign(r.melody,r.melody+r.melody_count);if(r.chord_count)chords.assign(r.chords,r.chords+r.chord_count);if(r.drum_count)drums.assign(r.drums,r.drums+r.drum_count);}
            validate_notes(melody,false);validate_notes(chords,true);validate_notes(drums,false);if(melody.empty())throw std::invalid_argument("Provided score requires a melody note");
        }else if(r.melody_count||r.chord_count||r.drum_count||r.melody_midi||r.chord_midi||r.drum_midi)throw std::invalid_argument("Score none must not include note/MIDI inputs");
        int64_t length=r.score_mode?0:r.symbolic_frames;
        for(const auto * lane:{&melody,&chords,&drums})for(const auto & n:*lane)length=std::max(length,n.onset+n.duration);
        if(length<1||length>5000||(r.score_mode&&r.symbolic_frames&&r.symbolic_frames!=length))throw std::invalid_argument("Invalid symbolic length or expected-length mismatch");
        const float fill=r.score_mode?0.0f:-1.0f;std::vector<float> piano(size_t(length)*256,fill),drum(size_t(length)*128,fill),chroma(size_t(length)*12,fill);std::vector<uint8_t> padding(size_t(length),0);
        for(const auto & n:melody){piano[size_t(n.onset)*256+size_t(n.pitch)*2]=1;for(int64_t t=n.onset+1;t<n.onset+n.duration;++t)piano[size_t(t)*256+size_t(n.pitch)*2+1]=1;}
        for(const auto & n:drums)for(int64_t t=n.onset;t<n.onset+n.duration;++t)drum[size_t(t)*128+n.pitch]=1;
        for(const auto & n:chords)for(int64_t t=n.onset;t<n.onset+n.duration;++t)chroma[size_t(t)*12+n.pitch]=1;
        uint64_t weights=0,graph=0;std::vector<float> qwen(1024,0),muq(512,0);checkpoint(r);
        if(!tags.empty())qwen=style(r.style_model,r.backend,r.threads,style_ids,r.diagnostic_directory,r,weights,graph);
        checkpoint(r);auto ot=owned(tokens);auto om=owned(mask);auto oq=owned(qwen);auto ou=owned(muq);auto op=owned(piano);auto od=owned(drum);auto oc=owned(chroma);auto ox=owned(padding);auto oi=owned(style_ids);auto ol=owned(lyric);auto os=owned(tags);
        mulacover_prepared_condition_v1 result{};result.size=output->size;
        result.tokens=ot.release();result.tokens_mask=om.release();result.qwen=oq.release();result.muq=ou.release();result.pianoroll=op.release();result.drum_pianoroll=od.release();result.chord=oc.release();result.context_mask=ox.release();result.style_tokens=oi.release();result.style_token_count=int64_t(style_ids.size());result.normalized_lyrics=ol.release();result.normalized_tags=os.release();result.style_weight_bytes=weights;result.style_graph_bytes=graph;
        result.condition={sizeof(result.condition),result.tokens,result.tokens_mask,int64_t(prompt.size()),result.qwen,result.muq,qi,mi,result.pianoroll,result.drum_pianoroll,result.chord,result.context_mask,length,r.guidance};*output=result;
    });
}
const mulacover_conditioning_api_v1 * MULACOVER_CALL mulacover_get_conditioning_api(uint32_t v){static const mulacover_conditioning_api_v1 api={MULACOVER_CONDITIONING_API_V1_SIZE,1,mulacover_conditioning_request_init_v1,mulacover_prepared_condition_init_v1,mulacover_prepared_condition_free_v1,mulacover_conditioning_create_v1,mulacover_conditioning_destroy_v1,mulacover_conditioning_prepare_v1};return v==1?&api:nullptr;}
}

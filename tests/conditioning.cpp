#include "mulacover_conditioning.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"conditioning check failed line %d: %s (%s)\n",__LINE__,#x,e.message);return 1;}}while(0)
static int32_t MULACOVER_CALL cancel(void * p){return *static_cast<bool *>(p)?1:0;}
static void MULACOVER_CALL progress(void * p,int64_t done,int64_t){if(done>=3)*static_cast<bool *>(p)=true;}
int main(int argc,char ** argv){
    mulacover_error_v1 e{};e.size=sizeof(e);const auto * api=mulacover_get_conditioning_api(1);CHECK(api&&api->size>=MULACOVER_CONDITIONING_API_V1_SIZE&&api->version==1&&!mulacover_get_conditioning_api(0)&&!mulacover_get_conditioning_api(2));
    struct {mulacover_prepared_condition_v1 value;uint64_t tail;} ext{};ext.value.size=sizeof(ext);ext.tail=123;api->result_init(&ext.value);api->result_free(&ext.value);api->result_free(&ext.value);CHECK(ext.value.size==sizeof(ext)&&ext.tail==123);
    mulacover_conditioning_request_v1 r{};r.size=sizeof(r);api->request_init(&r);CHECK(r.max_audio_frames==375&&r.symbolic_frames==1&&!r.tags[0]);
    mulacover_conditioning_context * ctx=nullptr;CHECK(api->create(nullptr,&ctx,&e)==1&&!ctx);auto empty=ext.value;CHECK(api->prepare(nullptr,&r,&ext.value,&e)==1&&!std::memcmp(&empty,&ext.value,sizeof(empty)));api->destroy(nullptr);
    if(argc==1)return 0;CHECK(!api->create(argv[1],&ctx,&e)&&ctx);
    bool cancelled=true;r.should_cancel=cancel;r.callback_user=&cancelled;CHECK(api->prepare(ctx,&r,&ext.value,&e)==3&&!std::memcmp(&empty,&ext.value,sizeof(empty)));cancelled=false;
    r.backend="cpu";r.lyrics="[Instrumental]";r.tags="  ";r.style_model="unused-empty-style.gguf";r.score_mode=1;r.symbolic_frames=0;
    std::vector<mulacover_note_v1> melody={{0,60,0},{0,61,3},{1,61,2},{4,72,1}};std::vector<mulacover_note_v1> chords={{0,60,2},{0,72,0},{2,67,3}};std::vector<mulacover_note_v1> drums={{0,36,2},{3,38,1}};
    r.melody=melody.data();r.melody_count=melody.size();r.chords=chords.data();r.chord_count=chords.size();r.drums=drums.data();r.drum_count=drums.size();
    CHECK(!api->prepare(ctx,&r,&ext.value,&e));const auto & c=ext.value.condition;CHECK(c.symbolic_frames==5&&c.prompt_frames==14&&c.qwen_index==3&&c.muq_index==6&&ext.value.style_token_count==1);
    CHECK(c.pianoroll[60*2]==1&&c.pianoroll[256+61*2]==1&&c.pianoroll[256+61*2+1]==1&&c.chord[0]==1&&c.chord[2*12+7]==1&&c.drum_pianoroll[128+36]==1);
    for(int i=0;i<1024;++i)CHECK(c.qwen[i]==0);
    const auto occupied=ext.value;CHECK(api->prepare(ctx,&r,&ext.value,&e)==1&&!std::memcmp(&occupied,&ext.value,sizeof(occupied)));api->result_free(&ext.value);
    for(int test=0;test<7;++test){auto bad=r;if(test==0)bad.lyrics=" \n ";if(test==1)bad.tags="\xc0\x80";if(test==2)bad.melody_count=-1;if(test==3)bad.symbolic_frames=4;if(test==4)bad.max_audio_frames=8192;if(test==5)bad.score_mode=0;if(test==6)bad.guidance=NAN;CHECK(api->prepare(ctx,&bad,&ext.value,&e)==1&&!std::memcmp(&empty,&ext.value,sizeof(empty)));}
    melody[0].pitch=128;CHECK(api->prepare(ctx,&r,&ext.value,&e)==1);melody[0].pitch=60;
    if(argc>2){r.tags="warm acoustic pop";r.style_model=argv[2];r.backend=argc>3?argv[3]:"cuda";r.on_style_progress=progress;CHECK(api->prepare(ctx,&r,&ext.value,&e)==3&&!std::memcmp(&empty,&ext.value,sizeof(empty)));cancelled=false;r.on_style_progress=nullptr;CHECK(!api->prepare(ctx,&r,&ext.value,&e)&&ext.value.style_weight_bytes>0);}
    else CHECK(!api->prepare(ctx,&r,&ext.value,&e));
    api->destroy(ctx);CHECK(ext.value.tokens&&ext.value.pianoroll&&ext.tail==123);api->result_free(&ext.value);CHECK(!std::memcmp(&empty,&ext.value,sizeof(empty)));std::puts("conditioning ownership, score semantics, validation and cancellation passed");return 0;
}

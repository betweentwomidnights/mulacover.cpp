#include "mulacover_conditioning.h"
#include "../tools/conditioning-args.h"
#include <cstring>
#include <memory>
int main(int argc,char ** argv){try{
    if(argc!=3)throw std::runtime_error("score-check ASSETS SCORE_FIXTURES");mulacover_conditioning_context * raw=nullptr;mulacover_error_v1 e{};e.size=sizeof(e);auto * api=mulacover_get_conditioning_api(1);
    if(api->create(argv[1],&raw,&e))throw std::runtime_error(e.message);
    std::unique_ptr<mulacover_conditioning_context,decltype(api->destroy)> ctx(raw,api->destroy);
    mulacover_conditioning_request_v1 r{};r.size=sizeof(r);api->request_init(&r);r.backend="cpu";r.score_mode=1;r.symbolic_frames=0;
    const std::filesystem::path root(argv[2]);const auto melody=(root/"melody.mid").string(),chord=(root/"chord.mid").string(),drum=(root/"drum.mid").string();r.melody_midi=melody.c_str();r.chord_midi=chord.c_str();r.drum_midi=drum.c_str();
    mulacover_prepared_condition_v1 out{};out.size=sizeof(out);api->result_init(&out);const auto empty=out;
    struct Owner{const mulacover_conditioning_api_v1 * a;mulacover_prepared_condition_v1 * p;~Owner(){a->result_free(p);}} owner{api,&out};
    if(api->prepare(ctx.get(),&r,&out,&e))throw std::runtime_error(e.message);
    for(auto pair:{std::make_pair("pianoroll.f32",static_cast<const void *>(out.pianoroll)),{"drum_pianoroll.f32",out.drum_pianoroll},{"chord.f32",out.chord},{"context_mask.u8",out.context_mask}}){auto bytes=conditioning_tool::text((root/pair.first).string());const std::string name=pair.first;const size_t row=name=="pianoroll.f32"?256*4:name=="drum_pianoroll.f32"?128*4:name=="chord.f32"?12*4:1;if(bytes.size()!=size_t(out.condition.symbolic_frames)*row||std::memcmp(pair.second,bytes.data(),bytes.size()))throw std::runtime_error("Score fixture shape or values differ");}
    api->result_free(&out);std::ifstream file(root/"invalid.txt");std::string line;int count=0;
    while(std::getline(file,line)){if(line.empty())continue;const auto path=(root/line).string();r.melody_midi=path.c_str();if(api->prepare(ctx.get(),&r,&out,&e)!=MULACOVER_INVALID_ARGUMENT_V1||std::memcmp(&empty,&out,sizeof(out)))throw std::runtime_error("Negative MIDI was accepted or published output: "+line);++count;}
    if(!count)throw std::runtime_error("Missing negative MIDI fixtures");std::cout<<"Score tensors exact; "<<count<<" upstream-rejected MIDI cases rejected without output"<<std::endl;return 0;
}catch(const std::exception & e){std::cerr<<e.what()<<std::endl;return 1;}}

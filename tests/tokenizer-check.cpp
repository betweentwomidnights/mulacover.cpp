#include "conditioning-internal.h"
#include "../tools/conditioning-args.h"
#include "../tools/wave-io.h"
#include <cstdio>
int main(int argc,char ** argv){try{
    if(argc!=4)throw std::runtime_error("tokenizer-check ASSETS CORPUS COUNT");mula::frontend::TextAssets assets(argv[1]);const std::filesystem::path root(argv[2]);const int count=std::stoi(argv[3]);if(count<1||count>100000)throw std::runtime_error("Invalid corpus count");
    for(int i=0;i<count;++i){auto folder=root/std::to_string(i);auto input=conditioning_tool::text((folder/"text.txt").string());
        for(bool style:{false,true}){const auto name=style?"style":"prompt";const auto path=folder/(std::string(name)+".i32");auto expected=wave_io::read<int32_t>(path,std::filesystem::file_size(path)/4);auto actual=(style?assets.style:assets.prompt).encode(input,style);if(actual!=expected)throw std::runtime_error("Tokenizer mismatch case "+std::to_string(i)+" "+name);}
        if(mula::frontend::lower(input)!=conditioning_tool::text((folder/"lower.txt").string())||mula::frontend::strip(input)!=conditioning_tool::text((folder/"strip.txt").string()))throw std::runtime_error("Normalization mismatch case "+std::to_string(i));
    }std::printf("%d text cases exact for both tokenizers, Python lower and strip\n",count);return 0;
}catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}}

#include "sampling.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

int main(int argc,char ** argv) {
    try {
        std::map<std::string,std::string> a;
        for(int i=1;i<argc;i+=2)if(i+1>=argc||!a.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("Invalid options");
        std::vector<float> logits(8197),noise(8*8197);
        auto read=[&](const char * key,float * dst,size_t count){std::ifstream f(a.at(key),std::ios::binary);
            if(!f.read(reinterpret_cast<char *>(dst),std::streamsize(count*4))||f.peek()!=EOF)throw std::runtime_error("Wrong input size");};
        read("--logits",logits.data(),8197);read("--noise",noise.data(),8197);
        for(int q=1;q<8;++q)std::copy_n(noise.data(),8197,noise.data()+q*8197);
        mulacover_sampling_config_v1 c{};c.size=sizeof(c);c.temperature=std::stof(a.at("--temperature"));c.top_k=std::stoi(a.at("--top-k"));
        c.exponential_noise=noise.data();c.noise_count=int64_t(noise.size());mula::validate_sampling(c);
        auto p=mula::sampling_probabilities(logits.data(),c.temperature,c.top_k);
        std::ofstream f(a.at("--probabilities"),std::ios::binary);f.write(reinterpret_cast<const char *>(p.data()),std::streamsize(p.size()*4));
        if(!f)throw std::runtime_error("Cannot write probabilities");
        std::cout<<mula::sample_logits(logits.data(),c,0,0)<<'\n';return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
}

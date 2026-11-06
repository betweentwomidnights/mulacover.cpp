#pragma once
// Native tool-only WAV I/O, following HeartCodec's encoder and the legacy driver.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace wave_io {
template<class T> static std::vector<T> read(const std::filesystem::path & p,size_t count) {
    if(std::filesystem::file_size(p)!=count*sizeof(T))throw std::runtime_error("Wrong encoder input length: "+p.string());
    std::vector<T> result(count);std::ifstream file(p,std::ios::binary);
    if(!file.read(reinterpret_cast<char *>(result.data()),std::streamsize(count*sizeof(T))))throw std::runtime_error("Cannot read encoder input");return result;
}
struct Wave {std::vector<float> samples;int rate=0,channels=0;};
static Wave read_wave_bytes(std::string_view bytes_view) {
    const auto bytes=bytes_view.size();
    if(bytes<44||bytes>UINT32_MAX)throw std::runtime_error("Invalid RIFF WAV length");
    const auto * data_ptr=reinterpret_cast<const uint8_t *>(bytes_view.data());
    struct View {const uint8_t * p;size_t n;const uint8_t * data()const{return p;}size_t size()const{return n;}uint8_t operator[](size_t i)const{return p[i];}} raw{data_ptr,size_t(bytes)};
    auto u16=[&](size_t i){return uint16_t(raw[i]|uint16_t(raw[i+1])<<8);};
    auto u32=[&](size_t i){return uint32_t(raw[i])|uint32_t(raw[i+1])<<8|uint32_t(raw[i+2])<<16|uint32_t(raw[i+3])<<24;};
    if(std::memcmp(raw.data(),"RIFF",4)||std::memcmp(raw.data()+8,"WAVE",4))throw std::runtime_error("Expected little-endian RIFF WAV");
    size_t fmt=0,format_size=0,data=0,data_size=0;
    for(size_t i=12;i+8<=raw.size();) {
        const size_t n=u32(i+4),payload=i+8;if(n>raw.size()-payload)throw std::runtime_error("Truncated WAV chunk");
        if(!std::memcmp(raw.data()+i,"fmt ",4)){fmt=payload;format_size=n;}
        if(!std::memcmp(raw.data()+i,"data",4)){data=payload;data_size=n;}
        i=payload+n+(n%2);
    }
    if(!fmt||format_size<16||!data||!data_size)throw std::runtime_error("Missing WAV format/audio");
    int format=u16(fmt),channels=u16(fmt+2),bits=u16(fmt+14);const int rate=int(u32(fmt+4));
    if(format==65534&&format_size>=40)format=u16(fmt+24);
    if((channels!=1&&channels!=2)||!((format==1&&(bits==16||bits==24||bits==32))||(format==3&&bits==32))||
       u16(fmt+12)!=channels*bits/8||data_size%(channels*bits/8))throw std::runtime_error("Unsupported PCM WAV format");
    Wave result;result.rate=rate;result.channels=channels;result.samples.resize(data_size/(bits/8));
    for(size_t i=0;i<result.samples.size();++i) {
        const size_t p=data+i*size_t(bits/8);float value=0;
        if(format==3)std::memcpy(&value,raw.data()+p,4);
        else if(bits==16)value=float(int16_t(u16(p)))/32768.0f;
        else if(bits==24){int32_t v=int32_t(raw[p]|uint32_t(raw[p+1])<<8|uint32_t(raw[p+2])<<16);if(v&0x800000)v|=int32_t(0xff000000u);value=float(v)/8388608.0f;}
        else value=float(int32_t(u32(p)))/2147483648.0f;
        result.samples[i]=value;
    }
    return result;
}
static Wave read_wave(const std::filesystem::path & path) {
    const auto bytes=std::filesystem::file_size(path);
    if(bytes<44||bytes>UINT32_MAX)throw std::runtime_error("Invalid RIFF WAV length");
    auto raw=read<char>(path,size_t(bytes));
    return read_wave_bytes(std::string_view(raw.data(),raw.size()));
}
template<class T>static void wav(const std::filesystem::path & path,const T & audio) {
    const uint64_t bytes=uint64_t(audio.num_samples)*8;if(bytes>UINT32_MAX-36)throw std::runtime_error("WAV is too large");
    std::ofstream f(path,std::ios::binary);auto u32=[&](uint32_t x){f.write(reinterpret_cast<const char *>(&x),4);};
    auto u16=[&](uint16_t x){f.write(reinterpret_cast<const char *>(&x),2);};
    f.write("RIFF",4);u32(uint32_t(bytes+36));f.write("WAVEfmt ",8);u32(16);u16(3);u16(2);
    u32(uint32_t(audio.sample_rate));u32(uint32_t(audio.sample_rate)*8);u16(8);u16(32);f.write("data",4);u32(uint32_t(bytes));
    for(int64_t i=0;i<audio.num_samples;++i){f.write(reinterpret_cast<const char *>(audio.samples+i),4);f.write(reinterpret_cast<const char *>(audio.samples+audio.num_samples+i),4);}
    if(!f)throw std::runtime_error("Cannot write float32 WAV");
}

}

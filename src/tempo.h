#pragma once
#include "mulacover_tempo.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace mula {
inline void validate_tempo(const mulacover_tempo_config_v1 & c) {
    if(c.size<sizeof(c)||!std::isfinite(c.bpm)||(c.bpm!=0&&(c.bpm<20||c.bpm>400))||
       !std::isfinite(c.width_beats)||c.width_beats<0.0625||c.width_beats>16||
       !std::isfinite(c.offset_beats)||c.offset_beats< -256||c.offset_beats>1250||c.local_window>1)
        throw std::invalid_argument("Invalid experimental score-clock configuration");
}
inline std::vector<uint8_t> tempo_padding(const mulacover_tempo_config_v1 & c,int pos,int prompt,
                                          const std::vector<uint8_t> & padding,int branch) {
    if(c.bpm==0||branch||!c.local_window)return padding;
    const int frame=std::max(0,pos-(prompt-1));
    const double center=4*c.offset_beats+frame*.08*c.bpm/15;
    auto result=padding;int nearest=-1;double distance=1e30;
    for(size_t k=0;k<padding.size();++k)if(!padding[k]) {
        const double d=std::abs(double(k)-center);
        if(d<distance){nearest=int(k);distance=d;}
        if(d>4*c.width_beats)result[k]=1;
    }
    if(nearest<0)throw std::invalid_argument("Score clock requires an unpadded row");
    result[size_t(nearest)]=0;return result;
}
inline std::vector<float> tempo_bias(const mulacover_tempo_config_v1 & c,int pos,int queries,int prompt,
                                     const std::vector<uint8_t> & padding,int branch) {
    if(c.bpm==0||branch)return {};
    const int keys=int(padding.size());int first=0,last=keys-1;
    while(first<keys&&padding[size_t(first)])++first;
    while(last>=0&&padding[size_t(last)])--last;
    if(first>last)throw std::invalid_argument("Score clock requires an unpadded row");
    std::vector<float> bias(size_t(queries)*keys,0);
    for(int t=0;t<queries;++t) {
        const int frame=pos+t-(prompt-1);
        if(frame<0)continue;
        const double center=std::clamp(4*c.offset_beats+frame*0.08*c.bpm/15.0,double(first),double(last));
        for(int k=0;k<keys;++k) {
            const double d=(k-center)/(4*c.width_beats);
            // Finite floor preserves soft attention even outside a narrow band.
            bias[size_t(t)*keys+k]=float(std::max(-80.0,-0.5*d*d));
        }
    }
    return bias;
}
}

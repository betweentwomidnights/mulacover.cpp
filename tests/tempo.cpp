#include "tempo.h"
#include <cstdlib>
#include <limits>
#include <iostream>
static void check(bool b){if(!b){std::cerr<<"Score clock check failed\n";std::exit(1);}}
int main() {
    mulacover_tempo_config_v1 c{sizeof(c),120,1,0};mula::validate_tempo(c);
    std::vector<uint8_t> padding(600,0);
    // P=25: the final prefill row predicts audio frame zero, not 1.92s.
    auto b=mula::tempo_bias(c,0,25,25,padding,0);
    check(b[23*600+100]==0&&b[24*600]==0&&b[24*600+4]==-0.5f);
    // Frame 100 = 8s = 16 quarter notes = 64 sixteenths at 120 BPM.
    b=mula::tempo_bias(c,124,1,25,padding,0);check(b[64]==0&&b[68]==-0.5f&&b[0]==-80);
    c.bpm=60;b=mula::tempo_bias(c,124,1,25,padding,0);check(b[32]==0);
    c.offset_beats=2;b=mula::tempo_bias(c,124,1,25,padding,0);check(b[40]==0);
    check(mula::tempo_bias(c,124,1,25,padding,1).empty());
    c.bpm=0;check(mula::tempo_bias(c,124,1,25,padding,0).empty());
    c.bpm=120;c.offset_beats=0;c.local_window=1;
    auto p=mula::tempo_padding(c,124,25,padding,0);
    check(p[59]&& !p[60]&&!p[64]&&!p[68]&&p[69]);
    check(mula::tempo_padding(c,124,25,padding,1)==padding);
    c.bpm=0;check(mula::tempo_padding(c,124,25,padding,0)==padding);
    c.bpm=400;c.offset_beats=1250;padding[599]=1;
    b=mula::tempo_bias(c,8191,1,25,padding,0);check(b[598]==0);
    for(double bad:{-1.0,19.0,401.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        c.bpm=bad;bool caught=false;try{mula::validate_tempo(c);}catch(const std::invalid_argument &){caught=true;}check(caught);
    }
    c.bpm=120;c.width_beats=0;bool caught=false;try{mula::validate_tempo(c);}catch(const std::invalid_argument &){caught=true;}check(caught);
    return 0;
}

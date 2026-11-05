#include "conditioning-internal.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <limits>

namespace mula::frontend {
void validate_notes(std::vector<mulacover_note_v1> & notes,bool chord){
    std::vector<mulacover_note_v1> result;result.reserve(notes.size());
    for(auto n:notes){
        if(n.onset<0||n.duration<0||n.pitch<0||n.pitch>127)throw std::invalid_argument("Notes require nonnegative sixteenth-grid times and MIDI pitch 0..127");
        if(chord&&n.duration==0)continue;
        if(n.onset>5000||n.duration>5000||n.onset>5000-n.duration)throw std::invalid_argument("Symbolic input exceeds 5000 sixteenths");
        if(chord)n.pitch%=12;else n.duration=std::max<int64_t>(1,n.duration);
        result.push_back(n);
    }notes.swap(result);
}
std::vector<mulacover_note_v1> midi(const char * path,bool include_drums){
    if(!path||!*path)throw std::invalid_argument("MIDI path required");
    std::ifstream f(path,std::ios::binary|std::ios::ate);auto length=f.tellg();
    if(length<14||length>64*1024*1024)throw std::invalid_argument("Invalid MIDI file length");
    std::vector<uint8_t> bytes(static_cast<size_t>(length));f.seekg(0);if(!f.read(reinterpret_cast<char *>(bytes.data()),length))throw std::runtime_error("Cannot read MIDI");
    size_t pos=0,limit=bytes.size();
    auto byte=[&](){if(pos>=limit)throw std::invalid_argument("Truncated MIDI event");return bytes[pos++];};
    auto u16=[&](){auto a=byte();return uint32_t(a)*256+byte();};
    auto u32=[&](){uint32_t a=u16();return a*65536+u16();};
    auto magic=[&](const char * expected){for(int i=0;i<4;++i)if(byte()!=uint8_t(expected[i]))throw std::invalid_argument("Invalid MIDI chunk");};
    auto variable=[&](){uint32_t v=0;for(int i=0;i<4;++i){auto b=byte();v=v*128+(b&127);if(!(b&128))return v;}throw std::invalid_argument("MIDI variable length exceeds four bytes");};
    magic("MThd");const auto header=u32();if(header!=6)throw std::invalid_argument("MIDI header must have length six");
    const auto type=u16(),tracks=u16(),ppq=u16();if(type>1||!tracks||!ppq||(ppq&0x8000)||(type==0&&tracks!=1))throw std::invalid_argument("MIDI requires format 0/1 and positive PPQ");
    std::vector<mulacover_note_v1> notes;int64_t first_tempo_tick=INT64_MAX;uint32_t first_tempo=500000;
    for(uint32_t track=0;track<tracks;++track){
        limit=bytes.size();magic("MTrk");const auto n=u32();if(n>bytes.size()-pos)throw std::invalid_argument("Truncated MIDI track");limit=pos+n;
        int64_t tick=0;uint8_t running=0;std::map<int,std::deque<int64_t>> active;
        while(pos<limit){const auto delta=variable();if(tick>INT64_MAX-delta)throw std::invalid_argument("MIDI tick overflow");tick+=delta;
            uint8_t status=byte();if(status<128){if(running<128||running>=240)throw std::invalid_argument("Invalid MIDI running status");--pos;status=running;}
            if(status==255){auto meta=byte();auto count=variable();if(count>limit-pos)throw std::invalid_argument("Truncated MIDI meta event");
                if(meta==81){if(count!=3)throw std::invalid_argument("Invalid MIDI tempo length");uint32_t tempo=uint32_t(bytes[pos])*65536+uint32_t(bytes[pos+1])*256+bytes[pos+2];if(tick<first_tempo_tick){first_tempo_tick=tick;first_tempo=tempo;}}
                pos+=count;continue;
            }
            if(status==240||status==247){running=0;auto count=variable();if(count>limit-pos)throw std::invalid_argument("Truncated MIDI sysex");pos+=count;continue;}
            if(status>=240)throw std::invalid_argument("Unsupported system MIDI event");running=status;
            const auto a=byte();const auto kind=status>>4;const auto b=(kind==12||kind==13)?0:byte();if(a>127||b>127)throw std::invalid_argument("Invalid MIDI data byte");
            if((kind!=8&&kind!=9)||(!include_drums&&(status&15)==9))continue;
            auto & queue=active[(status&15)*128+a];if(kind==9&&b){queue.push_back(tick);continue;}
            if(queue.empty())throw std::invalid_argument("Unmatched MIDI note off");auto onset=queue.front();queue.pop_front();
            if(tick>(INT64_MAX-ppq)/8)throw std::invalid_argument("MIDI grid overflow");
            const int64_t start=(onset*8+ppq)/(2*ppq),end=(tick*8+ppq)/(2*ppq);notes.push_back({start,int32_t(a),end-start});
            if(notes.size()>1000000)throw std::invalid_argument("Too many MIDI notes");
        }
        for(const auto & entry:active)if(!entry.second.empty())throw std::invalid_argument("Unterminated MIDI note");
    }
    if(!first_tempo)throw std::invalid_argument("Initial MIDI tempo must be positive");return notes;
}
}

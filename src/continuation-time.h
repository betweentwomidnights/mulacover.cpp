#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
namespace mula {
/* Exact floor(decimal seconds / .08), independent of locale/binary FP.
 * CLI adapter only; the public continuation ABI uses integer RVQ frames. */
inline int64_t duration_frames(const std::string & text) {
    auto invalid=[](){throw std::invalid_argument("Duration must be a finite nonnegative decimal number");};
    size_t i=0;if(i<text.size()&&text[i]=='+')++i;
    std::string digits;int64_t fraction=0;bool point=false,has_digit=false;
    for(;i<text.size()&&text[i]!='e'&&text[i]!='E';++i) {
        const char c=text[i];if(c=='.'&&!point){point=true;continue;}
        if(c<'0'||c>'9')invalid();has_digit=true;digits+=c;if(point)++fraction;
    }
    if(!has_digit)invalid();int64_t exponent=0;
    if(i<text.size()) {
        ++i;bool negative=false;if(i<text.size()&&(text[i]=='+'||text[i]=='-'))negative=text[i++]=='-';
        if(i==text.size())invalid();
        for(;i<text.size();++i){if(text[i]<'0'||text[i]>'9'||exponent>100000)invalid();exponent=exponent*10+(text[i]-'0');}
        if(negative)exponent=-exponent;
    }
    auto first=digits.find_first_not_of('0');if(first==std::string::npos)return 0;digits.erase(0,first);
    const int64_t shift=exponent-fraction+2;
    if(shift<0){if(-shift>=int64_t(digits.size()))return 0;digits.resize(digits.size()-size_t(-shift));}
    else {if(shift>21||digits.size()+size_t(shift)>21)throw std::invalid_argument("Duration is too large");digits.append(size_t(shift),'0');}
    int carry=0;int64_t value=0;
    for(char c:digits) {
        const int next=carry*10+(c-'0'),digit=next/8;carry=next%8;
        if(value>(std::numeric_limits<int64_t>::max()-digit)/10)throw std::invalid_argument("Duration is too large");
        value=value*10+digit;
    }
    return value;
}
}

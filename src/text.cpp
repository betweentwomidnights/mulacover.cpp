#include "conditioning-internal.h"
#include "text-unicode.h"
#include <queue>
#include <limits>

namespace mula::frontend {
std::string strip(const std::string & s){auto cp=text::decode_utf8(s);size_t a=0,b=cp.size();while(a<b&&(text::is_whitespace(cp[a])||(cp[a]>=0x1c&&cp[a]<=0x1f)))++a;while(b>a&&(text::is_whitespace(cp[b-1])||(cp[b-1]>=0x1c&&cp[b-1]<=0x1f)))--b;return text::encode_codepoints(cp,a,b);}
std::string lower(const std::string & s){
    auto cp=text::decode_utf8(s);std::string out;
    const auto cased=[](uint32_t c){return text::in_ranges(c,std::begin(text_detail::cased_ranges),std::end(text_detail::cased_ranges));};
    const auto ignorable=[](uint32_t c){return text::in_ranges(c,std::begin(text_detail::ignorable_ranges),std::end(text_detail::ignorable_ranges));};
    for(size_t i=0;i<cp.size();++i){auto c=cp[i];
        if(c==0x3a3){size_t a=i,b=i+1;while(a&&ignorable(cp[a-1]))--a;while(b<cp.size()&&ignorable(cp[b]))++b;if(a&&cased(cp[a-1])&&(b==cp.size()||!cased(cp[b]))){text::append_utf8(out,0x3c2);continue;}}
        auto it=std::lower_bound(std::begin(text_detail::lower_map),std::end(text_detail::lower_map),c,[](const auto & x,uint32_t y){return x.from<y;});
        if(it!=std::end(text_detail::lower_map)&&it->from==c){text::append_utf8(out,it->first);if(it->second)text::append_utf8(out,it->second);}else text::append_utf8(out,c);
    }return out;
}
static int key(const gguf_context * g,const std::string & name,gguf_type type){int k=gguf_find_key(g,name.c_str());if(k<0||gguf_get_kv_type(g,k)!=type)throw std::runtime_error("Missing/invalid tokenizer metadata: "+name);return k;}
void Tokenizer::load(const gguf_context * g,const std::string & p){
    auto strings=[&](const char * suffix){const int k=key(g,p+suffix,GGUF_TYPE_ARRAY);if(gguf_get_arr_type(g,k)!=GGUF_TYPE_STRING)throw std::runtime_error("Wrong tokenizer string array");std::vector<std::string> v;for(size_t i=0;i<gguf_get_arr_n(g,k);++i)v.emplace_back(gguf_get_arr_str(g,k,i));return v;};
    auto tokens=strings(".tokens"),ranks=strings(".merges"),added=strings(".specials");
    if(tokens.size()!=(p=="prompt"?128000:151643)||ranks.empty()||ranks.size()>1000000||added.empty()||added.size()>1024)throw std::runtime_error("Invalid pinned tokenizer sizes");
    for(size_t i=0;i<tokens.size();++i)if(!vocab.emplace(tokens[i],int32_t(i)).second)throw std::runtime_error("Duplicate tokenizer token");
    for(size_t i=0;i<ranks.size();++i)merges.emplace(ranks[i],int32_t(i));
    int ids=key(g,p+".special_ids",GGUF_TYPE_ARRAY);if(gguf_get_arr_type(g,ids)!=GGUF_TYPE_INT32||gguf_get_arr_n(g,ids)!=added.size())throw std::runtime_error("Invalid added-token IDs");
    auto * numbers=static_cast<const int32_t *>(gguf_get_arr_data(g,ids));for(size_t i=0;i<added.size();++i){if(added[i].empty()||numbers[i]<int32_t(tokens.size())||numbers[i]>=(p=="prompt"?128256:151669))throw std::runtime_error("Invalid added token");specials.emplace_back(added[i],numbers[i]);}
    three_digits=gguf_get_val_u32(g,key(g,p+".digits",GGUF_TYPE_UINT32))==3;nfc=gguf_get_val_bool(g,key(g,p+".nfc",GGUF_TYPE_BOOL));ignore_merges=gguf_get_val_bool(g,key(g,p+".ignore_merges",GGUF_TYPE_BOOL));
    if(three_digits!=(p=="prompt")||nfc!=(p=="style")||ignore_merges!=(p=="prompt"))throw std::runtime_error("Unsupported tokenizer policy");
    std::array<bool,256> visible{};for(int b=33;b<=126;++b)visible[b]=true;for(int b=161;b<=172;++b)visible[b]=true;for(int b=174;b<=255;++b)visible[b]=true;
    uint32_t extra=256;for(size_t b=0;b<256;++b)text::append_utf8(bytes[b],visible[b]?uint32_t(b):extra++);
    for(const auto & b:bytes)if(!vocab.count(b))throw std::runtime_error("Tokenizer missing byte alphabet");
}
TextAssets::TextAssets(const char * path){
    if(!path||!*path)throw std::invalid_argument("Tokenizer GGUF path required");
    std::unique_ptr<gguf_context,decltype(&gguf_free)> g(gguf_init_from_file(path,{true,nullptr}),gguf_free);
    if(!g||std::string(gguf_get_val_str(g.get(),key(g.get(),"general.architecture",GGUF_TYPE_STRING)))!="mulacover-tokenizers"||gguf_get_val_u32(g.get(),key(g.get(),"mulacover.tokenizer_version",GGUF_TYPE_UINT32))!=1||gguf_get_n_tensors(g.get()))throw std::runtime_error("Invalid tokenizer GGUF");
    prompt.load(g.get(),"prompt");style.load(g.get(),"style");
    for(auto item:{std::make_pair("text_bos_id",128000),{"text_eos_id",128001},{"audio_eos_id",8193},{"empty_id",0},{"tag_start_id",128021},{"tag_end_id",128022},{"muq_start_id",128023},{"muq_end_id",128024},{"qwen_start_id",128025},{"qwen_end_id",128026}}){const auto value=gguf_get_val_u32(g.get(),key(g.get(),std::string("prompt.config.")+item.first,GGUF_TYPE_UINT32));if(value!=uint32_t(item.second))throw std::runtime_error("Unsupported pinned prompt config token");config[item.first]=int32_t(value);}
}
void Tokenizer::piece(const std::string & raw,std::vector<int32_t> & output) const {
    if(raw.empty())return;
    std::string encoded;for(unsigned char b:raw)encoded+=bytes[b];auto whole=vocab.find(encoded);if(ignore_merges&&whole!=vocab.end()){output.push_back(whole->second);return;}
    struct Symbol {std::string value;int previous,next;uint64_t generation=0;};
    struct Candidate {int rank,left,right;uint64_t a,b;bool operator<(const Candidate & x)const{return rank>x.rank||(rank==x.rank&&left>x.left);}};
    std::vector<Symbol> symbols;symbols.reserve(raw.size());for(size_t i=0;i<raw.size();++i)symbols.push_back({bytes[static_cast<unsigned char>(raw[i])],int(i)-1,i+1==raw.size()?-1:int(i)+1});
    std::priority_queue<Candidate> queue;
    auto add=[&](int l,int r){if(l<0||r<0)return;auto it=merges.find(symbols[l].value+" "+symbols[r].value);if(it!=merges.end())queue.push({it->second,l,r,symbols[l].generation,symbols[r].generation});};
    for(int i=1;i<int(symbols.size());++i)add(i-1,i);
    while(!queue.empty()){auto c=queue.top();queue.pop();auto & l=symbols[c.left];auto & r=symbols[c.right];if(l.next!=c.right||l.generation!=c.a||r.generation!=c.b||l.value.empty()||r.value.empty())continue;
        l.value+=r.value;l.next=r.next;++l.generation;r.value.clear();++r.generation;if(r.next>=0)symbols[r.next].previous=c.left;add(l.previous,c.left);add(c.left,l.next);
    }
    for(int i=0;i>=0;i=symbols[i].next){auto it=vocab.find(symbols[i].value);if(it==vocab.end())throw std::runtime_error("BPE produced a missing vocabulary token");output.push_back(it->second);}
}
std::vector<int32_t> Tokenizer::encode(const std::string & input,bool eos) const {
    text::decode_utf8(input);std::vector<int32_t> out;size_t pos=0;
    while(pos<input.size()){
        size_t found=input.size();const std::pair<std::string,int32_t> * selected=nullptr;
        for(const auto & token:specials){auto where=input.find(token.first,pos);if(where<found||(where==found&&where!=std::string::npos&&selected&&token.first.size()>selected->first.size())){found=where;selected=&token;}}
        auto cp=text::decode_utf8(input.substr(pos,found-pos));if(nfc)cp=text::normalize_nfc(cp);for(const auto & s:text::pieces(cp,three_digits))piece(s,out);
        if(!selected)break;out.push_back(selected->second);pos=found+selected->first.size();
    }
    if(eos)out.push_back(151643);return out;
}
}

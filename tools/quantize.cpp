// Model policy follows sa3-quantize; formats and quantization come from GGML.
#include "weight-policy.h"
#include "gguf.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

int main(int argc,char ** argv) {
    try {
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;i+=2) {
            if(i+1>=argc)throw std::runtime_error("Missing option value");
            const std::string key=argv[i];
            if(key!="--input"&&key!="--output"&&key!="--mix")throw std::runtime_error("Unknown option: "+key);
            if(!args.emplace(key,argv[i+1]).second)throw std::runtime_error("Duplicate option: "+key);
        }
        if(!args.count("--input")||!args.count("--output")||!args.count("--mix"))
            throw std::runtime_error("Usage: mulacover-quantize --input generator-f32.gguf --output generator-quant.gguf --mix f16|q8_0|q6_k|q5_k_m|q4_k_m");
        const auto & mix=args.at("--mix");mula::weight_mix(mix);
        const std::filesystem::path input_path(args.at("--input")),output_path(args.at("--output"));
        const auto partial=output_path.string()+".partial";
        if(std::filesystem::weakly_canonical(input_path)==std::filesystem::weakly_canonical(output_path)||
           std::filesystem::exists(output_path)||std::filesystem::exists(partial))
            throw std::runtime_error("Output/partial must not exist and must differ from input");
        using Gguf=std::unique_ptr<gguf_context,decltype(&gguf_free)>;
        using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
        Gguf input(gguf_init_from_file(input_path.string().c_str(),{true,nullptr}),gguf_free);
        if(!input)throw std::runtime_error("Cannot read source GGUF");
        auto string_key=[&](const char * key,const char * value) {
            int id=gguf_find_key(input.get(),key);
            return id>=0&&gguf_get_kv_type(input.get(),id)==GGUF_TYPE_STRING&&std::string(gguf_get_val_str(input.get(),id))==value;
        };
        int version=gguf_find_key(input.get(),"mulacover.format_version");
        if(!string_key("general.architecture","mulacover")||version<0||
           gguf_get_kv_type(input.get(),version)!=GGUF_TYPE_UINT32||gguf_get_val_u32(input.get(),version)!=1)
            throw std::runtime_error("Expected MuLaCover GGUF format 1");
        int prior=gguf_find_key(input.get(),"mulacover.weight_precision");
        if(prior>=0&&!string_key("mulacover.weight_precision","f32")&&!string_key("mulacover.weight_precision","F32"))
            throw std::runtime_error("Quantize only from the original F32 model");
        Gguf output(gguf_init_empty(),gguf_free);
        Context ctx(ggml_init({8*1024*1024,nullptr,true}),ggml_free);
        if(!output||!ctx)throw std::runtime_error("Cannot allocate quantizer metadata");
        gguf_set_kv(output.get(),input.get());
        gguf_set_val_str(output.get(),"mulacover.weight_precision",mix.c_str());
        gguf_set_val_str(output.get(),"mulacover.quantization_policy",mula::weight_policy);
        gguf_set_val_u32(output.get(),"general.quantization_version",GGML_QNT_VERSION);
        const int64_t count=gguf_get_n_tensors(input.get());
        std::vector<ggml_tensor *> plan;
        std::map<std::string,size_t> counts,bytes;
        for(int64_t i=0;i<count;++i) {
            const std::string name=gguf_get_tensor_name(input.get(),i);
            if(gguf_get_tensor_type(input.get(),i)!=GGML_TYPE_F32)throw std::runtime_error("Source tensor is not F32: "+name);
            const int dims=gguf_get_tensor_ndims(input.get(),i);const auto * ne=gguf_get_tensor_ne(input.get(),i);
            const int policy_dims=dims==2&&ne[1]==1?1:dims;
            auto type=mula::weight_type(name,policy_dims,ne[0],mix);
            auto * tensor=ggml_new_tensor(ctx.get(),type,dims,ne);ggml_set_name(tensor,name.c_str());
            gguf_add_tensor(output.get(),tensor);plan.push_back(tensor);
            counts[ggml_type_name(type)]++;bytes[ggml_type_name(type)]+=ggml_nbytes(tensor);
        }
        if(!gguf_write_to_file(output.get(),partial.c_str(),true))throw std::runtime_error("Cannot write output metadata");
        std::ifstream source(input_path,std::ios::binary);
        std::fstream destination(partial,std::ios::binary|std::ios::in|std::ios::out);
        const uint64_t length=std::filesystem::file_size(input_path);
        const size_t alignment=gguf_get_alignment(output.get());
        const size_t output_offset=gguf_get_meta_size(output.get());
        for(int64_t i=0;i<count;++i) {
            auto * tensor=plan[size_t(i)];const size_t input_bytes=gguf_get_tensor_size(input.get(),i);
            const uint64_t offset=gguf_get_data_offset(input.get())+gguf_get_tensor_offset(input.get(),i);
            if(offset>length||input_bytes>length-offset)throw std::runtime_error("Truncated source tensor: "+std::string(tensor->name));
            std::vector<float> data(input_bytes/sizeof(float));source.seekg(std::streamoff(offset));
            if(!source.read(reinterpret_cast<char *>(data.data()),std::streamsize(input_bytes)))throw std::runtime_error("Cannot read source tensor");
            if(!ggml_validate_row_data(GGML_TYPE_F32,data.data(),input_bytes))throw std::runtime_error("Nonfinite source tensor");
            const size_t output_bytes=ggml_nbytes(tensor),padded=(output_bytes+alignment-1)/alignment*alignment;
            std::vector<uint8_t> blob(padded,0);
            if(tensor->type==GGML_TYPE_F32)std::memcpy(blob.data(),data.data(),input_bytes);
            else if(tensor->type==GGML_TYPE_F16)ggml_fp32_to_fp16_row(data.data(),reinterpret_cast<ggml_fp16_t *>(blob.data()),int64_t(data.size()));
            else {
                ggml_quantize_init(tensor->type);
                if(ggml_quantize_chunk(tensor->type,data.data(),blob.data(),0,tensor->ne[1],tensor->ne[0],nullptr)!=output_bytes)
                    throw std::runtime_error("Unexpected quantized byte count");
            }
            if(!ggml_validate_row_data(tensor->type,blob.data(),output_bytes))throw std::runtime_error("Invalid quantized tensor");
            destination.seekp(std::streamoff(output_offset+gguf_get_tensor_offset(output.get(),i)));
            if(!destination.write(reinterpret_cast<char *>(blob.data()),std::streamsize(blob.size())))throw std::runtime_error("Cannot write quantized tensor");
        }
        destination.close();if(!destination)throw std::runtime_error("Cannot flush quantized model");
        std::filesystem::rename(partial,output_path);
        std::ofstream report(output_path.string()+".quantization.json");
        report<<"{\"mix\":\""<<mix<<"\",\"policy\":\""<<mula::weight_policy<<"\",\"source_bytes\":"<<length
              <<",\"file_bytes\":"<<std::filesystem::file_size(output_path)<<",\"tensor_count\":"<<count<<",\"types\":{";
        bool first=true;for(const auto & item:counts) {
            if(!first)report<<',';first=false;
            report<<'"'<<item.first<<"\":{\"tensors\":"<<item.second<<",\"bytes\":"<<bytes[item.first]<<'}';
        }
        report<<"}}\n";if(!report)throw std::runtime_error("Cannot write quantization report");
        ggml_quantize_free();std::cout<<mix<<": "<<std::filesystem::file_size(output_path)<<" bytes\n";return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
}

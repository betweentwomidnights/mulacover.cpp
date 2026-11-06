#include "internal.h"
#include "ggml-cpu.h"
#include <stdexcept>
#include <iostream>

// Exercise changing shapes, outputs and failure unwinding on the existing CPU
// backend without a checkpoint; actual model parity is a separate fixture run.
int main(){try{
    mula::Model m;m.backend=ggml_backend_cpu_init();m.graph_reuse=true;
    if(!m.backend)throw std::runtime_error("CPU backend missing");
    auto compute=[&](const char * key,int frames,bool extra){
        mula::Graph g(m,key);std::vector<float> input(size_t(frames)*2);
        for(size_t i=0;i<input.size();++i)input[i]=float(i+1);
        auto * x=g.input(input,2,frames);auto * y=ggml_add(g.ctx,ggml_sqr(g.ctx,x),x);
        g.keep("answer",y);if(extra)g.keep("extra",ggml_scale(g.ctx,x,3));g.execute();
        const auto actual=g.read("answer");
        for(size_t i=0;i<input.size();++i)if(actual[i]!=input[i]*input[i]+input[i])throw std::runtime_error("Reused graph changed result");
        if(extra){const auto actual_extra=g.read("extra");for(size_t i=0;i<input.size();++i)if(actual_extra[i]!=3*input[i])throw std::runtime_error("Changed output set is stale");}
        return g.ctx;
    };
    auto * first=compute("local.1",3,false);
    if(compute("local.1",1,true)!=first||compute("local.1",7,false)!=first)throw std::runtime_error("Arena was not reused");
    // Output lifetime flags can change without changing node counts or shapes.
    for(bool keep_middle:{false,true,false,true}){
        mula::Graph g(m,"local.1");auto * x=g.input({1,2,3,4},4,1);
        auto * middle=ggml_sqr(g.ctx,x);auto * y=ggml_sqr(g.ctx,middle);
        if(keep_middle)g.keep("middle",middle);g.keep("last",y);g.execute();
        if(g.read("last")!=std::vector<float>({1,16,81,256})||(keep_middle&&g.read("middle")!=std::vector<float>({1,4,9,16})))throw std::runtime_error("Changed lifetime plan is stale");
    }
    if(compute("local.2",2,false)==first)throw std::runtime_error("Different slots share metadata");
    try{mula::Graph g(m,"local.1");throw std::runtime_error("intentional unwind");}catch(const std::runtime_error &){}
    compute("local.1",3,false);
    {mula::Graph held(m,"local.1");bool rejected=false;try{mula::Graph nested(m,"local.1");}catch(const std::logic_error &){rejected=true;}
        if(!rejected)throw std::runtime_error("Concurrent workspace borrow accepted");}
    compute("local.1",1,false);
    // The independent default path must still work with no workspace allocation.
    m.graph_reuse=false;compute("unused",3,true);if(m.workspaces.count("unused"))throw std::runtime_error("Default path allocated a workspace");
    // Profiling after graph-construction failure must not query an allocator
    // that has never reserved a buffer.
    m.graph_reuse=true;{mula::Graph unexecuted(m,"unexecuted");}m.profiling=true;
    std::cout<<"Workspace shape/output changes and unwind passed\n";return 0;
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}}

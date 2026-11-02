#include "internal.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
template<class T> void init(T * p) {
    if(!p||p->size<sizeof(*p))return;
    const auto size=p->size;std::memset(p,0,sizeof(*p));p->size=size;
}
int status(mulacover_error_v1 * e,int code,const char * text) {
    if(e&&e->size>=sizeof(*e)){e->status=code;std::snprintf(e->message,sizeof(e->message),"%s",text);}return code;
}
template<class F> int guard(mulacover_error_v1 * e,F && f) {
    try{f();return status(e,0,"");}
    catch(const std::invalid_argument & ex){return status(e,1,ex.what());}
    catch(const std::exception & ex){return status(e,2,ex.what());}
    catch(...){return status(e,2,"Unknown generator error");}
}
void context(mulacover_context * c){if(!c||!c->session)throw std::invalid_argument("Missing generator context");}
}
extern "C" {
void MULACOVER_CALL mulacover_config_init_v1(mulacover_config_v1 * c) {
    if(!c||c->size<sizeof(*c))return;init(c);c->backend="cpu";c->threads=4;c->cache_capacity=8192;
}
void MULACOVER_CALL mulacover_frame_init_v1(mulacover_frame_v1 * r){init(r);}
void MULACOVER_CALL mulacover_frame_free_v1(mulacover_frame_v1 * r) {
    if(!r||r->size<sizeof(*r))return;delete[]r->raw_logits;delete[]r->logits;init(r);
}
int MULACOVER_CALL mulacover_create_v1(const mulacover_config_v1 * c,mulacover_context ** out,mulacover_error_v1 * e) {
    if(out)*out=nullptr;
    return guard(e,[&]{
        if(!c||c->size<sizeof(*c)||!out||!c->model_path||!*c->model_path||!c->backend||c->threads<1||c->threads>1024||c->cache_capacity<1||c->cache_capacity>8192)
            throw std::invalid_argument("Invalid generator configuration");
        if(std::strcmp(c->backend,"cpu")&&std::strcmp(c->backend,"cuda"))throw std::invalid_argument("Backend must be cpu or cuda");
        auto p=std::make_unique<mulacover_context>();p->session=std::make_unique<mula::Session>(*c);*out=p.release();
    });
}
void MULACOVER_CALL mulacover_destroy_v1(mulacover_context * c){delete c;}
int MULACOVER_CALL mulacover_session_begin_v1(mulacover_context * c,const mulacover_condition_v1 * condition,mulacover_error_v1 * e) {
    return guard(e,[&]{context(c);if(!condition)throw std::invalid_argument("Missing prepared conditioning");c->session->begin(*condition);});
}
int MULACOVER_CALL mulacover_session_reset_v1(mulacover_context * c,mulacover_error_v1 * e) {
    return guard(e,[&]{context(c);c->session->reset();});
}
int MULACOVER_CALL mulacover_teacher_frame_v1(mulacover_context * c,const int32_t * codes,const char * dump,mulacover_frame_v1 * frame,mulacover_error_v1 * e) {
    return guard(e,[&]{
        context(c);if(!frame||frame->size<sizeof(*frame)||frame->raw_logits||frame->logits)throw std::invalid_argument("Invalid or occupied frame result");
        try{c->session->frame(codes,dump,*frame);}
        catch(const std::invalid_argument &){throw;} // Validation precedes any graph/cache mutation.
        catch(...){c->session->reset();throw;}
    });
}
int MULACOVER_CALL mulacover_get_memory_v1(mulacover_context * c,mulacover_memory_v1 * m,mulacover_error_v1 * e) {
    return guard(e,[&]{context(c);if(!m||m->size<sizeof(*m))throw std::invalid_argument("Invalid memory result");
        m->weight_bytes=ggml_backend_buffer_get_size(c->session->model.buffer);
        m->cache_bytes=(c->session->backbone.buffer?ggml_backend_buffer_get_size(c->session->backbone.buffer):0)+
                       (c->session->decoder.buffer?ggml_backend_buffer_get_size(c->session->decoder.buffer):0);
        m->graph_bytes=c->session->model.graph_bytes;
    });
}
const mulacover_api_v1 * MULACOVER_CALL mulacover_get_api(uint32_t version) {
    static const mulacover_api_v1 api={uint32_t(MULACOVER_API_V1_SIZE),1,mulacover_config_init_v1,mulacover_frame_init_v1,
        mulacover_frame_free_v1,mulacover_create_v1,mulacover_destroy_v1,mulacover_session_begin_v1,
        mulacover_session_reset_v1,mulacover_teacher_frame_v1,mulacover_get_memory_v1};
    return version==1?&api:nullptr;
}
}

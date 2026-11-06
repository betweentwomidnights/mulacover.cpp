#include "mulacover_runtime.h"
#include "ggml.h"
namespace {
const char * MULACOVER_CALL library_version(){return MULACOVER_LIBRARY_VERSION;}
const char * MULACOVER_CALL runtime_ggml_version(){return ggml_version();}
const char * MULACOVER_CALL runtime_ggml_commit(){return ggml_commit();}
}
extern "C" const mulacover_runtime_api_v1 * MULACOVER_CALL mulacover_get_runtime_api(uint32_t version){
    static const mulacover_runtime_api_v1 api={sizeof(api),1,library_version,runtime_ggml_version,runtime_ggml_commit};
    return version==1?&api:nullptr;
}

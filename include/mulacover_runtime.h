#ifndef MULACOVER_RUNTIME_H
#define MULACOVER_RUNTIME_H
#include "mulacover.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Separate model-free metadata capability; no changes to earlier tables.
 * Strings are borrowed from the loaded library, valid until library unload.
 * Getters create no backend/model contexts and may be called concurrently. */
typedef struct mulacover_runtime_api_v1 {
    uint32_t size;
    uint32_t version;
    const char * (MULACOVER_CALL * library_version)(void);
    const char * (MULACOVER_CALL * ggml_version)(void);
    const char * (MULACOVER_CALL * ggml_commit)(void);
} mulacover_runtime_api_v1;
#define MULACOVER_RUNTIME_API_V1_SIZE (offsetof(mulacover_runtime_api_v1,ggml_commit)+sizeof(((mulacover_runtime_api_v1 *)0)->ggml_commit))
MULACOVER_API const mulacover_runtime_api_v1 * MULACOVER_CALL mulacover_get_runtime_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif

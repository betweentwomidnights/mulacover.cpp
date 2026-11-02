#ifndef MULACOVER_H
#define MULACOVER_H
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32) && defined(MULACOVER_SHARED)
# if defined(MULACOVER_BUILD)
#  define MULACOVER_API __declspec(dllexport)
# else
#  define MULACOVER_API __declspec(dllimport)
# endif
#elif defined(__GNUC__)
# define MULACOVER_API __attribute__((visibility("default")))
#else
# define MULACOVER_API
#endif
#ifdef _WIN32
# define MULACOVER_CALL __cdecl
#else
# define MULACOVER_CALL
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mulacover_context mulacover_context;
typedef int32_t mulacover_status_v1;
enum { MULACOVER_OK_V1=0, MULACOVER_INVALID_ARGUMENT_V1=1, MULACOVER_ERROR_V1=2 };
typedef struct mulacover_error_v1 { uint32_t size; int32_t status; char message[512]; } mulacover_error_v1;
typedef struct mulacover_config_v1 {
    uint32_t size;
    const char * model_path;
    const char * backend; /* exact cpu or cuda, no fallback */
    int32_t threads;
    int32_t cache_capacity; /* 1..8192, default 8192 */
} mulacover_config_v1;
/* Prepared conditioning, copied by session_begin. Qwen/style and symbolic
 * preprocessing are supplied by the caller; this milestone does not tokenize text.
 * All arrays are contiguous, frame-major. Mask bytes must be zero or one. */
typedef struct mulacover_condition_v1 {
    uint32_t size;
    const int32_t * tokens; /* [prompt_frames,9], eight audio IDs and one text ID */
    const uint8_t * tokens_mask; /* [prompt_frames,9] */
    int64_t prompt_frames;
    const float * qwen; /* 1024 */
    const float * muq; /* 512 */
    int64_t qwen_index;
    int64_t muq_index;
    const float * pianoroll; /* [symbolic_frames,256] */
    const float * drum_pianoroll; /* [symbolic_frames,128] */
    const float * chord; /* [symbolic_frames,12] */
    const uint8_t * context_mask; /* symbolic_frames; one means padding */
    int64_t symbolic_frames;
    float guidance; /* <=1 conditional only; >1 conditional+unconditional */
} mulacover_condition_v1;
typedef struct mulacover_frame_v1 {
    uint32_t size;
    float * raw_logits; /* library-owned [8,branches,8197], C then U */
    float * logits; /* library-owned [8,8197], after guidance */
    int32_t codes[8]; /* teacher-forced or selected codes; see sampling extension */
    int32_t branches;
    int64_t cache_position; /* active backbone cache length after this frame */
} mulacover_frame_v1;
/* Frame buffers outlive reset/context destruction until frame_free is called.
 * Keep the shared library loaded until all its results have been released. */
typedef struct mulacover_memory_v1 {
    uint32_t size;
    uint64_t weight_bytes;
    uint64_t cache_bytes;
    uint64_t graph_bytes; /* maximum graph allocation so far, excludes workspace */
} mulacover_memory_v1;
MULACOVER_API void MULACOVER_CALL mulacover_config_init_v1(mulacover_config_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_frame_init_v1(mulacover_frame_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_frame_free_v1(mulacover_frame_v1 *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_create_v1(const mulacover_config_v1 *, mulacover_context **, mulacover_error_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_destroy_v1(mulacover_context *);
/* Context calls are serialized. Begin/reset clears all self-attention KV state. */
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_session_begin_v1(mulacover_context *, const mulacover_condition_v1 *, mulacover_error_v1 *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_session_reset_v1(mulacover_context *, mulacover_error_v1 *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_teacher_frame_v1(mulacover_context *, const int32_t codes[8], const char * dump_directory, mulacover_frame_v1 *, mulacover_error_v1 *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_get_memory_v1(mulacover_context *, mulacover_memory_v1 *, mulacover_error_v1 *);
typedef struct mulacover_api_v1 {
    uint32_t size;
    uint32_t version;
    void (MULACOVER_CALL * config_init)(mulacover_config_v1 *);
    void (MULACOVER_CALL * frame_init)(mulacover_frame_v1 *);
    void (MULACOVER_CALL * frame_free)(mulacover_frame_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * create)(const mulacover_config_v1 *,mulacover_context **,mulacover_error_v1 *);
    void (MULACOVER_CALL * destroy)(mulacover_context *);
    mulacover_status_v1 (MULACOVER_CALL * session_begin)(mulacover_context *,const mulacover_condition_v1 *,mulacover_error_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * session_reset)(mulacover_context *,mulacover_error_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * teacher_frame)(mulacover_context *,const int32_t *,const char *,mulacover_frame_v1 *,mulacover_error_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * get_memory)(mulacover_context *,mulacover_memory_v1 *,mulacover_error_v1 *);
} mulacover_api_v1;
#define MULACOVER_ABI_VERSION_1 1u
#define MULACOVER_API_V1_SIZE (offsetof(mulacover_api_v1,get_memory)+sizeof(((mulacover_api_v1 *)0)->get_memory))
MULACOVER_API const mulacover_api_v1 * MULACOVER_CALL mulacover_get_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif

#ifndef MULACOVER_SAMPLING_H
#define MULACOVER_SAMPLING_H
#include "mulacover.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { MULACOVER_CONTINUE_V1=0, MULACOVER_EOS_V1=1, MULACOVER_PADDING_V1=2 };
typedef struct mulacover_sampling_config_v1 {
    uint32_t size;
    float temperature; /* finite, >0; default 1 */
    int32_t top_k; /* 1..8197, threshold ties retained; default 250 */
    uint64_t seed; /* native SplitMix64/exponential race, not a PyTorch seed */
    const float * exponential_noise; /* optional borrowed [8,8197], finite >0 */
    int64_t noise_count; /* exactly 8*8197 with noise, otherwise 0 */
} mulacover_sampling_config_v1;
MULACOVER_API void MULACOVER_CALL mulacover_sampling_config_init_v1(mulacover_sampling_config_v1 *);
/* Selects each head before computing the next, with shared C/U selections.
 * Validates before mutation. Runtime failure rewinds the session. stop_reason
 * is written only on success. EOS (any code>=8193) takes priority over padding
 * (8192). Discard terminal frames; never send their codes to HeartCodec.
 * A terminal session requires reset/begin before any more frames.
 * Reset rewinds seed/frame counters. Results use the existing frame_free. */
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_sample_frame_v1(
    mulacover_context *,const mulacover_sampling_config_v1 *,const char * dump_directory,
    mulacover_frame_v1 *,int32_t * stop_reason,mulacover_error_v1 *);

/* Separate capability table: the frozen teacher-forcing v1 table is unchanged. */
typedef struct mulacover_sampling_api_v1 {
    uint32_t size;
    uint32_t version;
    const mulacover_api_v1 * core;
    void (MULACOVER_CALL * config_init)(mulacover_sampling_config_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * sample_frame)(mulacover_context *,const mulacover_sampling_config_v1 *,
        const char *,mulacover_frame_v1 *,int32_t *,mulacover_error_v1 *);
} mulacover_sampling_api_v1;
#define MULACOVER_SAMPLING_API_V1_SIZE (offsetof(mulacover_sampling_api_v1,sample_frame)+sizeof(((mulacover_sampling_api_v1 *)0)->sample_frame))
MULACOVER_API const mulacover_sampling_api_v1 * MULACOVER_CALL mulacover_get_sampling_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif

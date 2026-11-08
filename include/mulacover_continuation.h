#ifndef MULACOVER_CONTINUATION_H
#define MULACOVER_CONTINUATION_H
#include "mulacover.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Model-independent continuation primitives. One RVQ frame is 80 ms.
 * Frame-major guide buffers can be selected using the returned start/count.
 * No inference, Python, filesystem, global state or model allocation here. */
typedef struct mulacover_continuation_config_v1 {
    uint32_t size;
    int64_t source_frames;
    int64_t source_start_frame;
    int64_t guide_frames; /* zero selects all remaining source frames */
    int64_t new_frames; /* requested new audio, >0; sampling may reach EOS */
    int64_t prompt_frames;
    int32_t cache_capacity; /* zero rounds required positions up to 256 */
} mulacover_continuation_config_v1;
typedef struct mulacover_continuation_plan_v1 {
    uint32_t size;
    int64_t source_start_frame;
    int64_t guide_frames;
    int64_t new_frames;
    int64_t total_frames;
    int32_t cache_capacity;
} mulacover_continuation_plan_v1;
typedef struct mulacover_continuation_audio_v1 {
    uint32_t size;
    float * samples; /* library-owned planar [2,num_samples], new audio only */
    int64_t num_samples; /* per channel, 48000 Hz */
    int32_t sample_rate;
} mulacover_continuation_audio_v1;
MULACOVER_API void MULACOVER_CALL mulacover_continuation_config_init_v1(mulacover_continuation_config_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_continuation_plan_init_v1(mulacover_continuation_plan_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_continuation_audio_init_v1(mulacover_continuation_audio_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_continuation_audio_free_v1(mulacover_continuation_audio_v1 *);
/* Validate before writing result; invalid input leaves it unchanged. */
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_continuation_plan_build_v1(
    const mulacover_continuation_config_v1 *,mulacover_continuation_plan_v1 *,mulacover_error_v1 *);
/* Copy a suffix from a borrowed native HeartCodec waveform. Input must be
 * planar stereo at 48000 Hz, with exactly 3840 samples per RVQ frame.
 * Keeps sample values unchanged; does not crossfade, stretch or normalize.
 * Result must be initialized and empty. It outlives the input and all contexts;
 * release it with audio_free before unloading this library. On error it stays
 * unchanged. Decode guide+new tokens together before calling this function. */
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_continuation_copy_audio_v1(
    const float * samples,int64_t num_samples,int32_t sample_rate,int64_t guide_frames,
    mulacover_continuation_audio_v1 *,mulacover_error_v1 *);
typedef struct mulacover_continuation_api_v1 {
    uint32_t size;
    uint32_t version;
    const mulacover_api_v1 * core;
    void (MULACOVER_CALL * config_init)(mulacover_continuation_config_v1 *);
    void (MULACOVER_CALL * plan_init)(mulacover_continuation_plan_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * plan)(const mulacover_continuation_config_v1 *,mulacover_continuation_plan_v1 *,mulacover_error_v1 *);
    void (MULACOVER_CALL * audio_init)(mulacover_continuation_audio_v1 *);
    void (MULACOVER_CALL * audio_free)(mulacover_continuation_audio_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * copy_audio)(const float *,int64_t,int32_t,int64_t,mulacover_continuation_audio_v1 *,mulacover_error_v1 *);
} mulacover_continuation_api_v1;
#define MULACOVER_CONTINUATION_API_V1_SIZE (offsetof(mulacover_continuation_api_v1,copy_audio)+sizeof(((mulacover_continuation_api_v1 *)0)->copy_audio))
MULACOVER_API const mulacover_continuation_api_v1 * MULACOVER_CALL mulacover_get_continuation_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif

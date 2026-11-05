#ifndef MULACOVER_JOB_H
#define MULACOVER_JOB_H
#include "mulacover_continuation.h"
#include "mulacover_sampling.h"
#ifdef __cplusplus
extern "C" {
#endif
enum { MULACOVER_JOB_CANCELLED_V1=3 };
enum { MULACOVER_JOB_SCORE_NONE_V1=0, MULACOVER_JOB_SCORE_PROVIDED_V1=1 };
enum { MULACOVER_JOB_RNG_NATIVE_V1=0, MULACOVER_JOB_RNG_TORCH_CUDA_V1=1 };
enum { MULACOVER_JOB_LOADING_ENCODER_V1=0, MULACOVER_JOB_ENCODING_V1=1,
       MULACOVER_JOB_LOADING_GENERATOR_V1=2, MULACOVER_JOB_PRIMING_V1=3,
       MULACOVER_JOB_SAMPLING_V1=4, MULACOVER_JOB_LOADING_DECODER_V1=5,
       MULACOVER_JOB_DECODING_V1=6, MULACOVER_JOB_DONE_V1=7 };
typedef struct mulacover_job_progress_v1 {
    uint32_t size;
    int32_t stage;
    const char * stage_name; /* borrowed for callback only */
    int64_t completed;
    int64_t total;
    double fraction; /* monotonic overall 0..1; EOS may finish below the ceiling */
} mulacover_job_progress_v1;
typedef void (MULACOVER_CALL * mulacover_job_progress_callback_v1)(void *,const mulacover_job_progress_v1 *);
typedef int32_t (MULACOVER_CALL * mulacover_job_cancel_callback_v1)(void *);
typedef struct mulacover_job_audio_view_v1 {
    uint32_t size;
    const float * samples; /* borrowed mono/stereo F32 PCM */
    int64_t num_samples; /* per channel */
    int32_t sample_rate;
    int32_t channels;
    int64_t channel_stride; /* elements; planar: num_samples, interleaved: 1 */
    int64_t sample_stride; /* elements; planar: 1, interleaved: channels */
} mulacover_job_audio_view_v1;
typedef struct mulacover_job_request_v1 {
    uint32_t size;
    const char * encoder_model;
    const char * generator_model;
    const char * flow_model;
    const char * sq_model;
    const char * backend; /* exact cpu or cuda, never falls back */
    int32_t cpu_threads;
    mulacover_job_audio_view_v1 input_audio;
    int64_t cursor_sample; /* exclusive original-clock position; -1 = input end */
    int64_t context_samples; /* past PCM ending at cursor; 0 = all available past */
    int64_t guide_samples; /* tail of encoded context to prime; 0 = all context */
    double natural_max_seconds; /* new audio ceiling, floor to 80ms; EOS honored */
    int32_t cache_capacity; /* 0 = automatic */
    mulacover_condition_v1 conditioning; /* borrowed prepared preset; no text/Qwen preprocessing */
    int32_t score_mode;
    mulacover_sampling_config_v1 sampling; /* exponential_noise must be null */
    int32_t rng_mode;
    uint32_t rng_cuda_blocks; /* positive in Torch mode; default 288 matches Spark oracle */
    int32_t rng_device;
    uint64_t sampler_offset; /* Philox output offset, multiple of four */
    uint64_t decoder_seed; /* independent stream starting at offset zero */
    int32_t decoder_steps;
    int64_t segment_codes;
    int32_t include_full_audio; /* default 0; optionally retain reconstructed guide+new */
    mulacover_job_progress_callback_v1 on_progress;
    mulacover_job_cancel_callback_v1 should_cancel;
    void * callback_user;
} mulacover_job_request_v1;
typedef struct mulacover_job_plan_v1 {
    uint32_t size;
    int64_t cursor_sample;
    int64_t context_start_sample;
    int64_t context_samples;
    int64_t context_frames;
    int64_t guide_start_frame;
    int64_t guide_frames;
    int64_t new_frames;
    int32_t cache_capacity;
} mulacover_job_plan_v1;
typedef struct mulacover_job_result_v1 {
    uint32_t size;
    float * samples; /* owned planar stereo, NEW audio only */
    int64_t num_samples; /* per channel */
    int32_t sample_rate; /* 48000 */
    float * full_samples; /* optional owned planar guide+new */
    int64_t full_num_samples;
    int32_t * codes; /* owned frame-major [guide_frames+new_frames,8] */
    int64_t guide_frames;
    int64_t new_frames;
    int32_t stop_reason; /* CONTINUE at ceiling, or EOS */
    mulacover_job_plan_v1 plan;
    uint64_t sampler_final_offset;
    uint64_t decoder_final_offset;
    int64_t decoder_rng_windows;
    double encode_seconds;
    double generate_seconds;
    double decode_seconds;
    uint64_t encoder_weight_bytes;
    uint64_t encoder_graph_bytes;
    uint64_t generator_weight_bytes;
    uint64_t generator_cache_bytes;
    uint64_t generator_graph_bytes;
} mulacover_job_result_v1;
/* Synchronous worker-thread operation, not an audio callback. Requests/PCM and
 * conditioning remain alive until return. Callbacks cannot throw/reenter.
 * Models are released in order encoder -> generator -> flow -> SQ. Cancellation
 * is cooperative between graphs/frames/flow steps, never inside a running graph.
 * Source PCM after the cursor is never supplied to the encoder. Context length
 * is floored to complete 80ms input frames backwards from the exact cursor.
 * No PCM splice, normalization or time-stretching is performed.
 * Results survive all contexts; free through this API before unloading DLLs.
 * Failure/cancellation leaves an initialized empty result unchanged. */
MULACOVER_API void MULACOVER_CALL mulacover_job_request_init_v1(mulacover_job_request_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_job_plan_init_v1(mulacover_job_plan_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_job_result_init_v1(mulacover_job_result_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_job_result_free_v1(mulacover_job_result_v1 *);
/* Plan is model-free and does not read PCM/conditioning arrays or invoke callbacks. */
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_job_plan_build_v1(const mulacover_job_request_v1 *,mulacover_job_plan_v1 *,mulacover_error_v1 *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_job_run_v1(const mulacover_job_request_v1 *,mulacover_job_result_v1 *,mulacover_error_v1 *);
typedef struct mulacover_job_api_v1 {
    uint32_t size;
    uint32_t version;
    void (MULACOVER_CALL * request_init)(mulacover_job_request_v1 *);
    void (MULACOVER_CALL * plan_init)(mulacover_job_plan_v1 *);
    void (MULACOVER_CALL * result_init)(mulacover_job_result_v1 *);
    void (MULACOVER_CALL * result_free)(mulacover_job_result_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * plan)(const mulacover_job_request_v1 *,mulacover_job_plan_v1 *,mulacover_error_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * run)(const mulacover_job_request_v1 *,mulacover_job_result_v1 *,mulacover_error_v1 *);
} mulacover_job_api_v1;
#define MULACOVER_JOB_API_V1_SIZE (offsetof(mulacover_job_api_v1,run)+sizeof(((mulacover_job_api_v1 *)0)->run))
MULACOVER_API const mulacover_job_api_v1 * MULACOVER_CALL mulacover_get_job_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif

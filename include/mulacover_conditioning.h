#ifndef MULACOVER_CONDITIONING_H
#define MULACOVER_CONDITIONING_H
#include "mulacover_job.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mulacover_conditioning_context mulacover_conditioning_context;
enum { MULACOVER_CONDITIONING_CANCELLED_V1=3 };
typedef struct mulacover_note_v1 {int64_t onset;int32_t pitch;int64_t duration;} mulacover_note_v1;
typedef struct mulacover_conditioning_request_v1 {
    uint32_t size;
    const char * lyrics; /* literal, valid UTF-8; no implicit file reads */
    const char * tags;
    const char * style_model; /* strict F32 Qwen GGUF; unused for normalized-empty tags */
    const char * backend; /* exact cpu or cuda; no fallback */
    int32_t threads;
    int32_t score_mode; /* NONE or PROVIDED from job API */
    int64_t symbolic_frames; /* none mode length; provided mode: 0=derive, otherwise exact check */
    const mulacover_note_v1 * melody;int64_t melody_count;
    const mulacover_note_v1 * chords;int64_t chord_count;
    const mulacover_note_v1 * drums;int64_t drum_count;
    const char * melody_midi; /* provided mode: either MIDI paths or triplets, never both */
    const char * chord_midi;
    const char * drum_midi;
    float guidance;
    int64_t max_audio_frames; /* upstream prompt+audio budget, 1..8192 */
    mulacover_job_cancel_callback_v1 should_cancel;
    void * callback_user;
    void (MULACOVER_CALL * on_style_progress)(void *,int64_t completed,int64_t total);
    const char * diagnostic_directory; /* optional offline tensor dumps */
} mulacover_conditioning_request_v1;
typedef struct mulacover_prepared_condition_v1 {
    uint32_t size;
    mulacover_condition_v1 condition; /* borrowed view of owned fields below; one conditional branch */
    int32_t * tokens;uint8_t * tokens_mask;
    float * qwen;float * muq;
    float * pianoroll;float * drum_pianoroll;float * chord;uint8_t * context_mask;
    int32_t * style_tokens;int64_t style_token_count;
    char * normalized_lyrics;char * normalized_tags;
    uint64_t style_weight_bytes;uint64_t style_graph_bytes;
} mulacover_prepared_condition_v1;
/* Worker-thread calls, serialized context. Context owns CPU tokenizer maps only;
 * Qwen is loaded per nonempty-style prepare and released before return. Inputs
 * and callbacks remain borrowed until return; callbacks cannot throw/reenter.
 * Triplet timing is integer sixteenths; MIDI tempo is metadata, never BPM input.
 * No transcription/lane selection occurs. Caller owns any past-only score trim.
 * Cooperative cancellation between style graphs/layers, not inside graphs or
 * initial weight loading. Failure/cancellation leaves output unchanged. Results
 * survive context destruction; free through this DLL before unloading it.
 * Frozen generator/job and all earlier capability layouts remain unchanged. */
MULACOVER_API void MULACOVER_CALL mulacover_conditioning_request_init_v1(mulacover_conditioning_request_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_prepared_condition_init_v1(mulacover_prepared_condition_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_prepared_condition_free_v1(mulacover_prepared_condition_v1 *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_conditioning_create_v1(const char * tokenizer_gguf,mulacover_conditioning_context **,mulacover_error_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_conditioning_destroy_v1(mulacover_conditioning_context *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_conditioning_prepare_v1(mulacover_conditioning_context *,const mulacover_conditioning_request_v1 *,mulacover_prepared_condition_v1 *,mulacover_error_v1 *);
typedef struct mulacover_conditioning_api_v1 {
    uint32_t size;uint32_t version;
    void (MULACOVER_CALL * request_init)(mulacover_conditioning_request_v1 *);
    void (MULACOVER_CALL * result_init)(mulacover_prepared_condition_v1 *);
    void (MULACOVER_CALL * result_free)(mulacover_prepared_condition_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * create)(const char *,mulacover_conditioning_context **,mulacover_error_v1 *);
    void (MULACOVER_CALL * destroy)(mulacover_conditioning_context *);
    mulacover_status_v1 (MULACOVER_CALL * prepare)(mulacover_conditioning_context *,const mulacover_conditioning_request_v1 *,mulacover_prepared_condition_v1 *,mulacover_error_v1 *);
} mulacover_conditioning_api_v1;
#define MULACOVER_CONDITIONING_API_V1_SIZE (offsetof(mulacover_conditioning_api_v1,prepare)+sizeof(((mulacover_conditioning_api_v1 *)0)->prepare))
MULACOVER_API const mulacover_conditioning_api_v1 * MULACOVER_CALL mulacover_get_conditioning_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif

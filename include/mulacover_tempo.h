#ifndef MULACOVER_TEMPO_H
#define MULACOVER_TEMPO_H
#include "mulacover.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental inference-time score clock, not a trained BPM input or a tempo
 * guarantee. Adds a Gaussian log-attention prior to conditional score reads
 * (content and gate), without changing audio rate, backbone positions or RNG.
 * The frozen core and sampling tables are unchanged. */
typedef struct mulacover_tempo_config_v1 {
    uint32_t size;
    double bpm; /* 0 disables; otherwise finite 20..400 */
    double width_beats; /* Gaussian sigma in quarter notes, finite 1/16..16 */
    double offset_beats; /* score position at audio t=0, finite -256..1250 */
    uint32_t local_window; /* 0: Gaussian prior only; 1: also restrict adaptor
                           * and cross-attention keys to +/- width_beats */
} mulacover_tempo_config_v1;
MULACOVER_API void MULACOVER_CALL mulacover_tempo_config_init_v1(mulacover_tempo_config_v1 *);
/* Set before the first frame, or after reset. Reset/begin retain this setting.
 * Invalid configuration or a running/terminal session leaves state unchanged.
 * At score boundaries the attention center holds at the nearest unpadded row.
 * Prompt rows before the first audio prediction receive no Gaussian bias;
 * local_window also restricts their score context to the initial window. */
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_set_tempo_v1(
    mulacover_context *,const mulacover_tempo_config_v1 *,mulacover_error_v1 *);
typedef struct mulacover_tempo_api_v1 {
    uint32_t size;
    uint32_t version;
    const mulacover_api_v1 * core;
    void (MULACOVER_CALL * config_init)(mulacover_tempo_config_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * set)(mulacover_context *,const mulacover_tempo_config_v1 *,mulacover_error_v1 *);
} mulacover_tempo_api_v1;
#define MULACOVER_TEMPO_API_V1_SIZE (offsetof(mulacover_tempo_api_v1,set)+sizeof(((mulacover_tempo_api_v1 *)0)->set))
MULACOVER_API const mulacover_tempo_api_v1 * MULACOVER_CALL mulacover_get_tempo_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif

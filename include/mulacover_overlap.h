#ifndef MULACOVER_OVERLAP_H
#define MULACOVER_OVERLAP_H
#include "mulacover_job.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental, separately negotiated past-only staging. Legacy job tables,
 * layouts and zero-overlap behavior remain unchanged. No EOS suppression. */
typedef struct mulacover_overlap_config_v1 {
    uint32_t size;
    int64_t overlap_frames; /* 80ms each; 0--8192, default 0; clamped to guide_frames-1 */
} mulacover_overlap_config_v1;
typedef struct mulacover_overlap_plan_v1 {
    uint32_t size;
    mulacover_job_plan_v1 source; /* original clock/cursor and logical prefix */
    int64_t model_guide_frames; /* source.guide_frames minus overlap_frames */
    int64_t overlap_frames;
    int64_t generated_frames_ceiling; /* overlap plus requested NEW frames */
} mulacover_overlap_plan_v1;
typedef struct mulacover_overlap_result_v1 {
    uint32_t size;
    /* NEW audio begins at source.cursor_sample. The optional full preview and
     * codes include model guide + regenerated overlap + NEW audio. In this
     * capability job.guide_frames denotes that logical prefix; its final
     * staging.overlap_frames are generated, not original encoder codes. */
    mulacover_job_result_v1 job;
    mulacover_overlap_plan_v1 staging;
} mulacover_overlap_result_v1;
MULACOVER_API void MULACOVER_CALL mulacover_overlap_config_init_v1(mulacover_overlap_config_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_overlap_plan_init_v1(mulacover_overlap_plan_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_overlap_result_init_v1(mulacover_overlap_result_v1 *);
MULACOVER_API void MULACOVER_CALL mulacover_overlap_result_free_v1(mulacover_overlap_result_v1 *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_overlap_plan_build_v1(const mulacover_job_request_v1 *,const mulacover_overlap_config_v1 *,mulacover_overlap_plan_v1 *,mulacover_error_v1 *);
MULACOVER_API mulacover_status_v1 MULACOVER_CALL mulacover_overlap_run_v1(const mulacover_job_request_v1 *,const mulacover_overlap_config_v1 *,mulacover_overlap_result_v1 *,mulacover_error_v1 *);
typedef struct mulacover_overlap_api_v1 {
    uint32_t size;
    uint32_t version;
    void (MULACOVER_CALL * config_init)(mulacover_overlap_config_v1 *);
    void (MULACOVER_CALL * plan_init)(mulacover_overlap_plan_v1 *);
    void (MULACOVER_CALL * result_init)(mulacover_overlap_result_v1 *);
    void (MULACOVER_CALL * result_free)(mulacover_overlap_result_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * plan)(const mulacover_job_request_v1 *,const mulacover_overlap_config_v1 *,mulacover_overlap_plan_v1 *,mulacover_error_v1 *);
    mulacover_status_v1 (MULACOVER_CALL * run)(const mulacover_job_request_v1 *,const mulacover_overlap_config_v1 *,mulacover_overlap_result_v1 *,mulacover_error_v1 *);
} mulacover_overlap_api_v1;
#define MULACOVER_OVERLAP_API_V1_SIZE (offsetof(mulacover_overlap_api_v1,run)+sizeof(((mulacover_overlap_api_v1 *)0)->run))
MULACOVER_API const mulacover_overlap_api_v1 * MULACOVER_CALL mulacover_get_overlap_api(uint32_t version);
#ifdef __cplusplus
}
#endif
#endif

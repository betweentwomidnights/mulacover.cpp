# Shared GGML baseline validation

November 1, 2026. MuLaCover consumes HeartCodec's coordinated GGML
`b77ae790e628270af46a8af1eca38441c23f1252`. Its normal parity harness checks that
clean source SHA and the executable's linked commit. No new GGML revision is
adopted by either consumer.

On the unchanged baseline, the shared Spark CUDA build succeeds and native C ABI
smoke tests pass, but the small F32 matmul regression fails. The original frozen
MuLaCover FP32 oracle then fails **130/300 conditional tensors and 124/300 CFG
tensors** under the existing gates. Reset/replay is bit exact and inactive cache
tails remain zero. These are arithmetic parity failures, not a cache-reset pass
being substituted for model parity. Earlier `07b77b0` passing reports are retained.

An isolated GGML proposal based on this exact commit routes an explicit F32 src1
request away from MMF's reduced operand conversion, for ordinary F32/F32 matmul.
See sibling [HeartCodec's report](../../heartcodec.cpp/docs/GGML_B77_ALIGNMENT.md)
for scope, reproducer and independent review. Default/quantized/F16/BF16 and
`MUL_MAT_ID` dispatch are outside this change. Parity uses `GGML_CUDA_TF32=0`
without a forced reduced cuBLAS compute type.

The candidate uses its own clean checkout and build directory in dedicated Spark
Docker containers. An explicit `--review-ggml-revision FULL_SHA` permits candidate
comparison only when the source matches that SHA, is clean, descends from
`b77ae79`, and the executable reports the same linked commit. Its report records
both shared baseline and `unadopted_review_candidate: true`. Normal defaults
continue to require `b77ae79`. No gates or oracle tensors are changed.

`reference/run-spark.sh build|parity` now uses `build-shared-b77-cuda` and
`refdata/b77-native-baseline`, leaving the historical build/results intact. The
baseline CUDA regression is expected to fail until a reviewed fix is adopted.

Draft [GGML PR #21](https://github.com/betweentwomidnights/ggml/pull/21) targets
`feature/shared-sa3-acestep-v0.17`, based on `b77ae79`. The unadopted candidate
`0f2ed045eaff6bf01bed074878b6988248efbc22` passes all 600 tensors, both exact
replays, zero cache tails, and all logit argmax checks. Maximum guided-logit error
is 3.624e-5; maximum KV error is 4.483e-5. The same two historical CFG hidden
activation flat-absolute diagnostics remain, passing the existing scale-aware gate.
No tolerances or reference payloads were altered.

Compact reports: [baseline](validation/native-f32-b77-baseline.json) and
[candidate](validation/native-f32-b77-review-candidate.json). Independent review
found no remaining blockers for this narrowly scoped draft after correcting the
test's backend scope. Fresh laptop Windows static/DLL C ABI builds and four Python
metric tests pass on `b77ae79`. Review and coordinated adoption remain separate.

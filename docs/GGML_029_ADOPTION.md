# Coordinated GGML 0.29 adoption

The supported shared-fork identity is
`e3db5210fe3ecdaa4d65b4c93d515f2cf15524b7` (upstream GGML 0.29.0), recorded in
`dependencies/ggml.json`. [GGML #22](https://github.com/betweentwomidnights/ggml/pull/22)
subsumes #21's CUDA F32-source fix and includes Vulkan/Metal precision fixes.
The historical branch name `feature/shared-sa3-acestep-v0.17` is unchanged.

HeartCodec #3 merged the coordinated gitlink into `main` before the encoder
and controlled-work branches were rebased in that order. Range-diff confirms
all six feature commits retain their source changes, including discrete-device
selection and the broadcast ADD needed by Metal. Existing negotiated ABI layouts
are unchanged. MuLaCover consumes that single normal GGML target through
`heartcodec::heartcodec`, or reuses a compatible target supplied by its parent.

The old `dependencies/ggml-development.json`, CMake source override,
`-DevelopmentGgml` PowerShell paths and Spark `*-development` commands are retired.
Old isolated worktrees and benchmark trees are retained as evidence; this cleanup
does not delete another agent's worktrees or alter the sibling migrations.

## Build and validate

From the sibling laptop checkouts:

```powershell
./scripts/build-windows.ps1 -Backend cpu
./scripts/build-windows.ps1 -Backend cpu -Shared
./scripts/build-windows.ps1 -Backend cuda
./scripts/build-windows.ps1 -Backend vulkan
./scripts/evaluate-windows.ps1 -Backend cuda -Capacity 256
```

CPU/CUDA/Vulkan build directories are `build-windows-{backend}` with `-dll` for
shared builds. Use fresh directories when moving from the retired override.
An old nonempty CMake override fails explicitly, rather than silently testing a
different target. Metal is selected with `-DMULACOVER_METAL=ON` in an Apple build.
Backend selection and unsupported-operation failures remain explicit; no CPU
fallback is introduced. The default Torch CUDA RNG still needs a CUDA build/device;
Metal/Vulkan jobs use explicit native RNG until that optional capability exists.

Parity/generation provenance tools require a clean exact coordinated GGML checkout,
and record the linked binary's GGML revision. The removed review-candidate flag
cannot bypass that check. For CUDA reference work retain TF32 disabled,
cuBLAS compute type `auto` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

On Spark, host actions remain file/Git/Docker orchestration only. Source GGML is
mounted read-only from HeartCodec; no development overlay is used:

```bash
bash reference/run-spark.sh build
bash reference/run-spark.sh parity
bash reference/run-spark.sh conditioning-native
```

## Evidence and historical boundaries

The coordinated bump-check reports in `ggml-bump-check` cover all seven consumers:
`migrations/2026-11-05-heartcodec-mulacover.md` and
`migrations/2026-11-06-m4-metal.md`. They include CUDA/Vulkan runs and M4 Metal/CPU
controls. Missing/skipped work and existing failures retain their original scope.
This repository's follow-up validation is recorded in
`docs/validation/ggml-029-adoption.json`.

Fresh Spark Docker CUDA checks pass all 12 CTests, the original 600-tensor
teacher-forced suite and all five sampled native conditioning cases. The
teacher-forced suite uses its unchanged strict public logit/KV limits and mixed
activation rule. Two CFG activations exceed the separate flat absolute diagnostic
(maxima 0.00146484375 and 0.001953125); these failures are retained. This does not
revalidate the longer 128-prompt suite or waive its recorded activation failures.

Windows CPU static, CPU DLL and laptop CUDA each pass all 12 CTests; HeartCodec's
standalone CPU build passes all six. Laptop CUDA also passes the five conditioning
cases and style cancellation/retry. Model-backed job tests pass cancellation in
all four stages, poisoned-future PCM handling, and two same-pin byte-exact code/audio
replays. Controlled codec calls pass context reuse and exact legacy-call replay.
These short integration checks do not replace a full-song listening test.

GGML 0.29 changes some floating-point reduction order and sampled autoregressive
trajectories. Earlier `b77ae79`, `0f2ed045` and `07b77b0` reports and audio hashes
remain historical evidence; do not relabel them as `e3db5210` passes. In particular,
the accepted byte-exact continuation replay belongs to its recorded earlier pin.
New-pin checks use the unchanged numerical gates and same-pin replay controls.
Known encoder, CPU style and strict full-song waveform gaps are not waived by
the dependency refresh. No model quantization or generic backend code is changed.

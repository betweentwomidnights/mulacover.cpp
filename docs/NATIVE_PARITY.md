# Native FP32 frame/logit/cache parity

Validated November 1, 2026, in dedicated Spark Docker containers. Native source
development and Windows static/DLL smoke tests run on the laptop. This milestone
consumes prepared prompt/Qwen/symbolic tensors and teacher-forced RVQ codes; it
does not yet tokenize text, execute Qwen natively, sample free-running music, or
use quantized generator weights.

## Scope and result

Both conditional and CFG=1.5 cases pass **300 tensors each** across three audio
frames: prompt prefill (37 backbone positions), then two incremental frames
(positions 37 and 38). Each frame computes all eight heads. Both cases reset and
replay all native tensors bit-for-bit. Cache tails beyond active positions remain
zero, including after local-decoder resets between audio frames.

| Quantity | Conditional maximum error | CFG=1.5 maximum error |
| --- | ---: | ---: |
| Raw logits, 24 tensors per case | 1.98e-5 | 2.44e-5 |
| Guided logits, 24 tensors per case | 1.98e-5 | 3.63e-5 |
| Backbone/local KV prefixes, 186 tensors per case | 4.44e-5 | 4.49e-5 |
| Selected hidden/style/adaptor/local states, 66 tensors per case | 8.93e-4 | 1.954e-3 |

All compared logit argmax results agree. Teacher-forced codes are injected, not
sampled; this does not establish stochastic sampling equivalence. The native CLI
also checks invalid-code/occupied-result rejection without consuming a cache
position, output lifetime after context destruction, and repeated result release.

See `validation/native-f32-cuda.json` for compact results. The detailed 600-tensor
report and raw native dumps live under ignored Spark
`refdata/native-cuda-source-pin`; a detailed report copy is also available locally.

## Gates and floating-point diagnostics

Public raw/guided logits and all KV values retain the original gates: maximum
absolute error <=1e-3 **and** relative L2 <=1e-4.

For hidden activations the gate uses the standard elementwise combination
`abs(native-reference) <= 1e-3 + 1e-6*abs(reference)`, also requiring relative L2
<=1e-4. The flat absolute diagnostic remains in the report. In the first run, two
CFG prefill intermediates exceeded the flat 1e-3 threshold:

- `backbone.layers.27`: at reference -4549.4668, error 0.00146484375 = 3 float32 ULPs.
- `peft_layers.27.0`: at reference -4548.7246, error 0.001953125 = 4 float32 ULPs.

Their relative L2 errors are 2.11e-7 and 2.30e-7. A scale-aware activation gate
accounts for that float32 spacing instead of imposing one absolute threshold at
every magnitude. Tests verify this handling does not relax logit/cache gates,
accept small-magnitude drift, or accept nonfinite/shape-mismatched outputs. The two
flat-absolute failures are preserved explicitly, rather than omitted.

## Implemented model path

The strict converter loads all 659 upstream state entries with no missing,
unexpected, mismatched or ignored keys. It writes 669 F32 runtime tensors:
seven transposed local output heads replace the original stacked `audio_head`,
and four exact upstream nonpersistent RoPE caches are included for head dimensions
12, 64, 128 and 384. Checkpoint tensors originating in BF16 are losslessly upcast;
the parity oracle runs FP32 arithmetic on those weights.

The graph implements masked nine-slot prompt/audio embedding sums, Qwen/MuQ
prefix substitution, the 28-layer 3072-dimensional GQA backbone, Llama3 scaled
positions, RMSNorm/SwiGLU, per-layer StyleMLPs, eight symbolic adaptors (two 512-D
transformer layers each), cumulative symbolic context, gated cross-attention,
and the three-layer local decoder with seven subsequent output heads.

All operations use existing GGML primitives. Exact RoPE cos/sin arrays and separate
products/sums mirror the upstream FP32 expression, including the 12-D attention
gate. CFG executes conditional and unconditional branches sequentially. The local
decoder clears its cache for every audio frame; backbone caches persist until
session reset. Native GQA stores eight K/V heads in the backbone and four in the
local decoder. Torchtune stores repeated query-head copies (24 and 8); debug dumps
expand the native logical view for direct comparisons of every reference value.

## Memory and runtime

| Native owned buffer | Bytes |
| --- | ---: |
| F32 model, including exact RoPE arrays | 17,516,732,416 |
| Conditional caches, full 8192-position backbone capacity | 1,879,343,104 |
| CFG caches, full 8192-position backbone capacity | 3,758,686,208 |
| Maximum graph allocation | 98,073,472 |

These are backend buffer sizes, not total device peaks; backend/driver workspaces
and host graph metadata are excluded. Full FP32 is a Spark workload. The
conditional and CFG runs took about 2.03 and 3.92 seconds respectively for **three
frames plus their three-frame replay**, with debugging enabled and initial weight
loading excluded. These single validation runs are not throughput benchmarks.

## Dependency provenance and reproduction

The canonical result was rebuilt against clean, unchanged HeartCodec GGML
`07b77b0112cc30861b49116b057bac33e819ee06`, mounted read-only. The harness checks
the full Git SHA and clean status, and the native binary reports `07b77b01`.
CUDA matmul uses explicit F32 precision, `GGML_CUDA_TF32=0`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Native compiler is GCC 11.4 with CUDA 13.0 for
GB10 `121a-real`; reference is Torch 2.7.1+cu128 in the pinned HeartCodec reference
image. No production containers or host Python environments were changed.

The repositories are not all aligned today: see `GGML_PIN_INVENTORY.md`. This
milestone makes no new GGML commit, PR or consumer pin change. A candidate
installed-package dependency patch was deferred because the existing in-tree
target already links correctly. Earlier installed-package diagnostic runs are
not the canonical result used here.

```bash
bash reference/run-spark.sh convert
bash reference/run-spark.sh build
bash reference/run-spark.sh parity
```

Weights and source revisions are recorded in `validation/model-f32.json` and the
original oracle manifests. Q4_K_M, BF16 native math, longer/more varied prompts,
native preprocessing, free-running sampling and end-to-end cover generation
remain separate validation work.

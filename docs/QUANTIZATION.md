# Experimental generator quantization

November 1, 2026, using MuLaCover-only GGML development revision `0f2ed045`.
**Q4_K_M runs on the 8GB RTX 5070 Laptop.** Shared GGML pins and sibling repositories
remain unchanged. Spark conversion/reference/model tests ran only in dedicated Docker.

## Weight policy

`mulacover-quantize` writes GGUF format 1 with versioned policy
`mulacover-kquant-v1`. It follows SA3's Q4_K/Q6_K selection: ordinary block-aligned
2D weights use Q4_K; attention V projections, FFN down projections and embedding
tables use Q6_K. MuLaCover's eight output heads also use Q6_K. Norms, biases,
exact RoPE tables, vectors and rows not divisible by the target block size stay
F32. The Q4_K_M file contains 302 Q4_K, 112 Q6_K and 255 F32 tensors.

The same tool supports F16, Q8_0, Q6_K and Q5_K_M comparison mixes. It streams one
tensor at a time from the original F32 model, preserves auxiliary values and
source provenance, validates payloads and writes to a new output. Existing files,
partials, in-place conversion and reduced-precision inputs are refused. The
runtime checks every tensor against the declared precision/policy and shape.

Graph activations and KV caches are F32. F32 matmuls request F32 source and accumulator
precision; quantized matmuls request at least Q8 activation precision and F32
accumulation, using GGML's existing kernels. No new GGML changes were needed.
Quantized kernels may internally convert activations to Q8; stored graph tensors
remain F32.
The negotiated C ABI and its default configuration are unchanged.

## Measured footprint

| Mix | GGUF bytes | Approximate GiB |
| --- | ---: | ---: |
| F32 source | 17,516,780,256 | 16.31 |
| F16 | 8,768,709,920 | 8.17 |
| Q8_0 | 4,668,938,528 | 4.35 |
| Q4_K_M | 2,939,966,176 | 2.74 |

Q4 model SHA-256: `74ebd8a8537f55ea3f3566cfa17c7e28734fc555af5fbc1a254bec94a182f814`.

On the RTX 5070 Laptop, Q4_K_M with 8192 cache positions owns 2,939,920,896 weight
bytes, 1,879,343,104 conditional-cache bytes or 3,758,686,208 CFG-cache bytes, and
at most 98,073,472 graph bytes. Device usage sampled with nvidia-smi every 0.2s
peaked at **5,539 MiB conditional / 7,331 MiB CFG**, including roughly 654 MiB of
desktop usage. The device reports 8,151 MiB total; this fixture leaves about 820
MiB at its sampled CFG peak. With capacity 256, sampled peaks are 3,709 / 3,771 MiB.

These tests fully allocate the requested caches but populate only 37 prefill
positions and two incremental positions. They do not establish peaks for long
prompts, full cache occupancy or simultaneous generator/decoder residency.
Buffer totals exclude driver/backend workspaces; sampled peaks can miss brief
spikes. Debug/replay runtimes are not throughput benchmarks.

## Drift, not FP32 parity

Each configuration executes the same prepared conditional and CFG=1.5 fixtures,
three teacher-forced frames with eight heads, then reset/replay. All 600 tensors
are finite with expected shapes, replay is exact, inactive cache tails remain
zero, and the C ABI error/lifetime checks pass. The fixed F32 model still passes
all 600 original FP32 gates.

The Q4_K_M outputs fail the strict FP32 gates, as expected for lossy weights.
The measurement report keeps `passed: false`, `fp32_parity_passed: false`, and
`quantization_accuracy_accepted: false`; `execution_passed: true` only records
successful, repeatable execution. Exit code zero in measurement mode means the
measurement completed. FP32 gates and oracle payloads were not changed.

Guided-logit diagnostics on Spark, averaging 24 distributions per case:

| Mix | Conditional top-1 agreement | CFG top-1 agreement | Conditional mean KL | CFG mean KL |
| --- | ---: | ---: | ---: | ---: |
| F16 | 100% | 100% | 0.00000119 | 0.00000299 |
| Q8_0 | 87.5% | 91.7% | 0.00194 | 0.00432 |
| Q4_K_M | 75.0% | 45.8% | 0.0369 | 0.0766 |

Laptop CUDA Q4 at capacity 8192 gives 79.2% / 70.8% top-1 agreement and mean KL
0.0375 / 0.0759. Laptop CPU at capacity 256 gives 75.0% / 50.0%. Backend numerical
paths can produce different quantized logits and near-tied argmax choices;
neither bit-identical backends nor generated-code equivalence is claimed. With
only 24 heads per case, these figures are diagnostics, not a quality estimate.

Q4 is experimental. The current evidence establishes memory feasibility and
execution with prepared inputs. Longer and more varied teacher forcing,
weight-group sensitivity, free-running sampling and decoded-audio comparisons
are required before accepting its quality or starting coordinated GGML adoption.
Text/Qwen/MIDI preprocessing and complete cover generation remain future work.

## Reproduction and evidence

Use the [development profile](DEVELOPMENT_GGML.md). Model files and detailed tensor
dumps remain ignored under `models/` and `refdata/`; compact results live in
`validation/quantization-development.json`. The default Windows evaluation helper
uses Q4_K_M; `-Capacity 256` is convenient for short primitive tests and leaves
more laptop headroom. Full F32 and F16 execution remains on Spark.

Seven Python tests cover FP32/quantization metric separation, actual GGUF storage,
preserved auxiliary tensors and refusal paths. Windows CPU/CUDA development C ABI
smoke tests and the four combined Spark native tests pass. The CMake development
override rejects the shared baseline when the fixed development SHA is required.

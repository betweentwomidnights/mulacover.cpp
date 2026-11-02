# Native sampling, context tests and staged audio

November 1, 2026. Development uses isolated GGML `0f2ed045`; the shared pin stays
`b77ae79`. ggml-bump-check and all five sibling updates remain deferred.

Prepared conditioning now drives a native sampler and HeartCodec stereo decode.
The laptop Q4_K_M run generated 100 valid eight-stream RVQ frames and decoded them
to **384,000 samples/channel, 48kHz stereo (eight seconds)**. Native seeded replay
was exact. Generator weights/caches are released before loading the FP32 flow
model; HeartCodec then stages flow and SQ. The sampled device peak was **6,629
MiB**, including desktop usage, with generator cache capacity 256.

This is prepared-input generation, not a complete lyric/MIDI/Qwen frontend or
quality-approved cover pipeline. The fixture uses original short synthetic lyrics,
style tensors and MIDI conditioning. Its musical context is short; a listenable
file or finite waveform does not establish transcription fidelity or audio quality.

## Sampler and ABI

The policy follows the pinned [upstream model](https://github.com/HeartMuLa/MuLaCover/blob/e27fa7fa0da7c9817879cc6210cc5693f9327946/src/mulacover/modeling.py):
temperature scaling, top-k threshold masking with ties retained, log-softmax then
softmax, and an exponential-race categorical draw. Both CFG branches are computed
before each selection, and the selected code is fed to the next RVQ head in both
branches. Sampling eight completed teacher-forced logits afterwards is incorrect.

`include/mulacover_sampling.h` adds `mulacover_sampling_config_v1`,
`mulacover_sample_frame_v1` and `mulacover_get_sampling_api(1)`. The new capability
table points to the unchanged core v1 table. Existing layouts, entry points and
`get_api(2)==NULL` behavior are preserved. Defaults are temperature 1, top-k 250,
native seed 1701. Context calls remain serialized.

Native randomness uses SplitMix64 indexed by seed/frame/head/category, converted
to positive exponential noise. Reset rewinds the frame index. This is a native
replay contract, not equivalence to a PyTorch seed. Callers can instead supply a
borrowed `[8,8197]` F32 exponential-noise tensor for exact draw comparisons; values
must be finite and positive, with exact element count. No pointer is retained.

Any code 8193..8196 means EOS; EOS takes priority over padding 8192. Terminal
frames are returned with an explicit stop reason and must be discarded. No
subsequent frame is accepted until begin/reset. The generation tool raises on
padding, stops on EOS and never forwards special tokens to HeartCodec. Results
are owned/released through the existing frame API and survive context destruction.
Invalid arguments preserve cache position, output and stop marker; execution
failures rewind the session. The generation tool tests rejection, forced EOS and
padding, terminal rejection and reset recovery against real models.

Eight fixtures from the actual upstream sampler cover ordinary/full-vocabulary
sampling, temperature, threshold ties, flat logits, extreme logits and special
tokens. All selected IDs match with supplied noise. Probability gates are maximum
absolute error <=1e-6 and relative L2 <=1e-5, allowing F32 host-math rounding;
observed worst errors are 8.65e-7 and 1.94e-6. These new sampler gates do not alter
the model's logit/KV or activation gates. Independent fixed SplitMix64 vectors are
also checked, alongside C ABI static/DLL smoke tests.

## Expanded context evidence and remaining gap

The new upstream FP32 oracle reuses and hash-checks the existing prepared fixture.
It repeats prompt/mask tensors to 128 positions and symbolic tensors to 64 frames,
retaining the original style-prefix indices. This synthetic extension tests shapes,
positions and masks; it does not claim new upstream preprocessing parity.

| Case | Frames | Checkpoints | Compared tensors | Result |
| --- | ---: | --- | ---: | --- |
| Conditional teacher forcing | 32 | 0,1,7,31 | 400 | All logits/KV pass; 2 prefill activations fail |
| CFG=1.5 teacher forcing | 32 | 0,1,7,31 | 400 | All logits/KV pass; 2 prefill activations fail |
| CFG with supplied sampling noise | 8 | 0,1,7 | 300 | All 64 codes match; logits/KV pass; 2 prefill activations fail |

All **858 raw/guided-logit and KV tensors** pass the existing maximum-absolute
1e-3 and relative-L2 1e-4 gates. Worst public errors are 0.000337 and 0.0000113.
Reset/replay is bit exact and inactive tails stay zero. A 159-position cache is
filled to its last slot in the 32-frame cases; another frame is rejected without
state/output mutation, then replay succeeds. Laptop Q4 executes the same 32-frame
teacher cases and cache-boundary checks, with sampled CFG peak 3,719 MiB.

The complete expanded F32 suite **does not pass**. Prefill
`backbone.layers.27` and `peft_layers.27.0` exceed the existing elementwise
activation tolerance in each case. One conditional example has reference
73.202286, absolute error 0.0023346, versus an allowed 0.0010732. Global relative
L2 remains around 1.5e-6. Gates are unchanged, and reports retain `passed:false`
with `execution_passed:true`. An attention-prescaling experiment did not clear
the failures and was not adopted. This is open numerical parity work before
broader validation/adoption. The original 600-tensor suite still passes.

## Listening artifacts

Ignored local directories contain `audio.wav`, frame-major `codes.i32` and
provenance/model hashes:

- `refdata/generate-q4-laptop`: Q4_K_M on RTX 5070 Laptop, peak 0.1395, RMS 0.02995.
- `refdata/generate-f32-spark`: F32 generator on GB10, peak 0.2285, RMS 0.04664.
- `refdata/generate-q4-spark`: Q4_K_M on GB10, peak 0.3970, RMS 0.06042.

Each run generated/replayed 100 frames, passed terminal/reset tests, and produced
finite, unclipped float32 stereo WAV output. No gain normalization was applied.
Both Spark clips use the same native generator settings and decoder seed 1701,
10 Euler steps, guidance 1.25 and 372-code windows. They provide a same-platform
comparison. The laptop is a separate backend/platform run; identical numeric seeds
do not promise cross-platform token or decoder-noise equivalence. Listening and
musical-quality acceptance remain pending, and Q4 drift remains experimental.

## Reproduction

Use the [isolated development profile](DEVELOPMENT_GGML.md). On the laptop:

```powershell
..\heartcodec.cpp\.venv\Scripts\python.exe tools/generate.py --reference refdata/upstream-f32 `
  --model models/mulacover-q4_k_m.gguf --cli build-windows-development-cuda/bin/Release/mulacover-generate.exe `
  --output refdata/my-eight-second-run --ggml-source C:/dev/_ggml_reviews/cuda-f32-source-b77 `
  --frames 100 --capacity 256 --flow-model ../heartcodec.cpp/models/heartcodec-flow-f32.gguf `
  --sq-model ../heartcodec.cpp/models/heartcodec-sq-codec-f32.gguf
```

Spark Docker helpers: `sampling-oracle`, `sampling-development`, `sequence-oracle`,
`sequence-development`, and `generate-development` in `reference/run-spark.sh`.
Sequence oracle creation requires a fresh output; the current expanded parity
command exits nonzero for the recorded activation failures. Generation refuses
to overwrite a completed output. Set `MULACOVER_GENERATOR_MIX=f32` and a fresh
`MULACOVER_GENERATION_OUTPUT` for the F32 run. The helper records model, prepared
input, token and WAV hashes. Raw fixtures and audio stay outside Git; compact
evidence is `validation/sampling-and-generation.json`.

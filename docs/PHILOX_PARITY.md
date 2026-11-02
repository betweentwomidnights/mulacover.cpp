# Matched Torch CUDA RNG and full-song token parity

November 2, 2026. The user heard horns playing the guitar melody, without vocals,
in the official upstream FP32 control; earlier native F32/Q4 runs vocalized that
melody. Those runs used different random streams. This phase implements the
requested sa3.cpp-style Philox control and reruns the real source.

## Result

Native F32 on Spark now reproduces **all 6,888 upstream codes across 861 frames**
bit-for-bit, including EOS at the same position. Both produce 68.88 seconds.
The native run also passes exact reset/replay and terminal sampling checks.
The user confirms that this matched-Philox native F32 output sounds like the
upstream horns, without vocalizations. The instrumental request is listener-
accepted for this particular case; that does not accept every prompt/seed or
change any numerical waveform gate.
All 11 prepared tensors are identical to the official control. Both generator
caches are configured for 8,192 positions. The selected lane remains instrumental;
the separately confirmed vocal MIDI is not part of this case.

The decoded waveform is close but **does not pass the existing strict waveform
gate**: maximum absolute error 0.0827555 (limit 0.01), relative L2 0.00299061
(limit 0.001), cosine similarity 0.999995528. Limits come unchanged from
HeartCodec's `tools/compare-stereo.py`. Scalar-level mismatches have not been
measured for this full-song case, so no conditional gate exception is assumed.
Decoder numerical diagnostics remain open, as does the previously recorded
longer-context prefill activation gap. Do not call complete waveform parity passed.

The Q4 laptop run also completes 70 seconds with matched streams, cache capacity
1,024, reset/replay and terminal checks, and a sampled 6,865 MiB peak including
desktop usage. Q4 model drift remains separate from RNG identity; its tokens and
audio are not expected to match unquantized inference. Listening acceptance of
the Q4 clip remains pending.

## Implementation and oracle

HeartCodec provides the standalone negotiated C capability in `heartcodec_rng.h`.
It follows sa3.cpp's Philox integer rounds and CUDA grid-stride/offset policy.
CUDA transforms match Torch's F32 normal and exponential kernels bit-for-bit;
host libm approximations are not substituted. ACE-Step's BF16 policy was consulted
without copying its per-element mapping into this F32 path. No Torch dependency
is added to native binaries, and no shared GGML source/pin change is required.

The official sampler makes one exponential call of shape `[1,8197]` for each
head. Flattening an entire frame into one call would change the stream. At the
tested caps, each head consumes four offsets: an accepted 861-frame sequence
plus its EOS frame consumes 27,584 offsets. Reset/replay reuses the same noise.
Three codec windows consume consecutive CUDA normal draws of `[1,744,256]`,
ending at offset 12. CPU `randn` calls in upstream context construction populate
masked filler positions and do not advance the CUDA stream.

`reference/export-codec-rng.py` executes the real upstream codec methods with
deterministic model arithmetic stubs and verifies that the effective flow noise
is exactly three consecutive CUDA normal calls. The regular 31-case oracle covers
85,205,452 values per device, including a full 875-frame/7,000-head exponential
stream, tails, high-bit seeds/offsets and capped multi-trip tensors. Every value
and offset is bit-exact on both Spark and the laptop. The additional codec call
graph contributes 571,392 bit-exact normal values on the laptop.

Spark's target grid cap is 288; the laptop's physical cap is 216. Explicitly use
288 for cross-device comparisons with these fixtures. Zero means the execution
GPU's own cap. CUDA availability is explicit; CPU builds report this RNG mode
unavailable rather than silently selecting another stream.

One sampler correction accompanies RNG integration: with borrowed F32 noise,
probability/noise division now rounds to F32 before argmax, preserving Torch's
lowest-index tie behavior. A regression fixture exercises a tie where double
division previously selected the other category. Legacy SplitMix64 sampling and
its established reference vectors remain unchanged. All eight frozen sampler
fixtures and six native CTests pass on Spark CUDA, Windows CUDA and Windows CPU.

The frozen generator/codec core APIs and sampling v1 layouts are unchanged. The
new RNG table owns its host output; existing borrowed-noise fields carry the
values into sampling and decoding. Legacy RNG remains the default for historical
replay; use the explicit opt-in below for parity work.

## Reproduction and artifacts

Using the existing prepared full-song inputs, native development CLI and models:

```powershell
../heartcodec.cpp/.venv/Scripts/python.exe tools/generate.py `
  --reference refdata/covers/billiegasm-full-acoustic/prepared `
  --model models/mulacover-q4_k_m.gguf `
  --cli build-windows-development-cuda/bin/Release/mulacover-generate.exe `
  --output refdata/my-philox-run --frames 875 --capacity 1024 `
  --ggml-source C:/dev/_ggml_reviews/cuda-f32-source-b77 `
  --rng torch-cuda --rng-cuda-blocks 288 `
  --flow-model ../heartcodec.cpp/models/heartcodec-flow-f32.gguf `
  --sq-model ../heartcodec.cpp/models/heartcodec-sq-codec-f32.gguf
```

The transcription bridge accepts `-Rng torch-cuda -RngCudaBlocks 288`. Spark's
Docker helper accepts `MULACOVER_GENERATION_RNG=torch-cuda` and
`MULACOVER_RNG_CUDA_BLOCKS=288`; for this F32 comparison use model mix `f32`,
875 frames, capacity 8,192, and the original full-song prepared directory. All
Spark Python, inference and builds remain inside dedicated images.

Ignored run directories are `generation-f32-philox-spark` and
`generation-q4-philox-laptop` under `refdata/covers/billiegasm-full-acoustic/`.
The three comparison WAVs in `refdata/covers/billiegasm-philox-listening/` use one
constant gain of 0.79203825 (-2.02508 dB), with raw outputs preserved. Compact
evidence is `validation/philox-parity.json`; RNG oracle data remain outside Git.
Shared GGML stays at `b77ae79`, with the authorized isolated `0f2ed045` development
override. No five-sibling bump/check workflow has been initiated.

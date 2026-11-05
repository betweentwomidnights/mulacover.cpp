# Native PCM continuation job

`mulacover_get_job_api(1)` from `mulacover_job.h` accepts borrowed PCM and prepared
conditioning and returns owned **new audio only**, without Python or temporary
RVQ files. It stages HeartCodec's F32 encoder, the generator, then F32 flow and
SQ decoding. Each model is released before the next is loaded. Link through
`mulacover::mulacover`; it consumes `heartcodec::heartcodec` and one GGML target.
This requires HeartCodec's separate `heartcodec_work.h` capability. All earlier
core/sampling/tempo/continuation/encoder/RNG layouts and defaults remain frozen.

## Host contract

Initialize the request and result with their size prefixes. Keep paths, borrowed
mono/stereo F32 samples, conditioning and callback state alive until `run` returns.
PCM supports the encoder's input sample rates and explicit channel/sample strides
for interleaved or planar storage. Its count and cursor are **samples per channel
on the original input clock**, not bytes or 48kHz output samples.

- `cursor_sample` is exclusive; -1 means input end. Only earlier PCM reaches the
  encoder. `context_samples=0` selects all past audio; otherwise it selects up to
  that many samples, capped at the cursor. Context length is floored to complete
  80ms frames **backwards from the exact cursor**; any fractional frame is dropped
  from the oldest end. The encoder sees that narrowed interval before resampling,
  finite checks or feature extraction, so its noncausal frontend cannot read past
  the cursor. Encoding context separately from the guide preserves its meaning.
- `guide_samples=0` uses all encoded context; otherwise the most recent complete
  80ms frames, capped at context length. At least one guide frame is required.
  `plan` reports effective source bounds, encoded frames, guide tail and cache.
  It reads neither PCM nor conditioning arrays and loads no models.
- `natural_max_seconds` limits **new audio**, flooring `seconds*12.5` to integer
  frames. EOS is honored. It does not impose a minimum or suppress EOS. Guide,
  prompt and maximum new frames must fit 8192 positions; automatic cache planning
  rounds to 256. Practical memory/quality at the maximum remains unvalidated.
- Score `none` constructs upstream dropped-score tensors (-1, mask 0) at the
  supplied symbolic length. `provided` uses caller tensors unchanged. Prepared
  prompt/Qwen/MuQ tensors remain required in both modes. The job cannot establish
  whether caller score/style tensors contain future information: strict past-only
  transcription is the caller's responsibility. Supplied future arrangement is a
  separate supported transformation use case.
- Default Torch CUDA Philox reuses HeartCodec's validated normal/exponential
  capability, explicit positive target cap (default **288**; zero is rejected), full 64-bit seeds and multiple-of-four
  offsets. Teacher frames consume no draws; sampled EOS consumes all eight heads.
  Sampling uses eight exponential calls per frame (8197 values each). Decoder
  normal noise has its own seed and starts at offset zero. Result offsets report
  actual draws, including EOS. CPU-only builds fail this mode without fallback;
  explicit native RNG remains available for legacy experiments.

`run` is synchronous work for a DAW worker. Progress/cancel callbacks run on that
worker and must not throw or reenter; communicate cancellation with an atomic
flag. Cancellation is cooperative between generator frames, encoder graphs/chunks,
flow graphs/steps/windows and SQ graphs/channels. A running graph and initial
context/model loading are not interrupted. Progress is coarser than polling and
monotonic through completion; EOS may finish before the requested sampling total.
No realtime or deadline guarantee is implied. Status 3 means cancellation, 1
invalid arguments, 2 runtime failure. Failure/cancellation publishes no result.

Owned result codes contain guide+new frames. `samples` contains only new planar
48kHz stereo; optional `full_samples` retains reconstructed guide+new audio for
diagnostics. Results survive all model/context destruction. Use `result_free`
before unloading the DLL; never use a host allocator to release them. Models
and temporary buffers are released on all exits, so a fresh request can retry.
No input splice, normalization, time stretch or join processing is applied. Joint
noncausal decoding can change the reconstructed guide with different continuations.
Placement, device-rate conversion and join treatment belong to the downstream.

## Native tool

Build with the existing Windows script and isolated development GGML profile.
`mulacover-continue` reads PCM16/24/32 or F32 RIFF WAV natively; preset preparation
remains an offline boundary. Example from the repository root:

```powershell
$env:GGML_CUDA_TF32='0'
$env:GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
./build-windows-development-cuda/bin/Release/mulacover-continue.exe `
  --audio C:/Users/thegr/Music/billiegasm_yuey.wav `
  --input refdata/cursor-continuation/input `
  --output refdata/cursor-continuation/new-pcm-job `
  --encoder-model ../heartcodec.cpp/models/heartcodec-encoder-f32-v2.gguf `
  --model models/mulacover-q4_k_m.gguf `
  --flow-model ../heartcodec.cpp/models/heartcodec-flow-f32.gguf `
  --sq-model ../heartcodec.cpp/models/heartcodec-sq-codec-f32.gguf `
  --prompt-frames 25 --symbolic-frames 540 --qwen-index 14 --muq-index 17 `
  --cursor-sample 1312416 --natural-max-seconds 90 --score-mode none `
  --include-full-audio 1
```

At 44.1kHz that cursor is 29.76 seconds. For an eight-second tail guide, additionally
use `--guide-samples 352800`, retaining 29.76 seconds of encoder context. Default
output is `continuation.wav`, `codes.i32`, `guide.i32` and `run.json`; optional full
audio is `audio.wav`. Ctrl+C signals cooperative cancellation; cancelled jobs write
no completed output directory. The earlier prepared-code CLI remains available.

## Sibling patterns and validation

Reviewed local `sa3.cpp/src/libsa3_v1.h`, `sa3_pipeline.h`, `rng.h` and ACE-Step's
`pipeline-synth.h`, `request.h`, `philox.h`: borrowed strided inputs, owned results,
negotiated ABI, synchronous worker callbacks, cooperative cancellation and staged
residency apply directly. Philox alone is insufficient for Torch parity: normal
versus exponential transforms, dtype, draw shape/order, grid cap and offsets all
matter. Reuse the existing validated HeartCodec implementation; do not substitute
another model's distribution mapping or bf16/seed behavior. No sibling or GGML
checkout is modified. Shared pin adoption remains deferred.

`test-job` runs model-free planning/ABI/precancel checks. Its optional integration
form is `test-job SOURCE.wav PRESET_DIR ENCODER GENERATOR FLOW SQ` on CUDA. It
poisons all PCM after the cursor with NaNs, tests cancellation at encoding,
sampling, flow and waveform stages, then runs twice and compares codes/audio
exactly. It also checks equivalent dropped tensors through supplied-score mode.
HeartCodec's `test-work` separately tests context reuse after cancellation between graphs;
`test-rng` compares bulk/chunked normal and exponential streams bit-for-bit.

See [the job validation report](validation/native-audio-job.json) for actual full
run hashes, memory and checks. These are regressions against the accepted native
continuation, not a new upstream waveform-parity claim. The encoder's documented
Whisper/RVQ numerical limits and prior strict decoded-waveform failure remain.
Next frontend work is native prompt/tokenizer/Qwen and MIDI/score preparation,
consuming SheetSage2's public transcription interface rather than duplicating it.

# Native continuation and next milestones

The runtime target is a C++ DAW worker that accepts an audio guide, optional
symbolic score and style conditioning, primes the generator, and returns only
new audio. Two useful workflows are score-guided transformation/continuation
(including a supplied arrangement beyond the cursor) and audio-only continuation
from material available before the cursor. Evaluate them separately.
Python is retained for reference oracles, checkpoint conversion and analysis;
it should not be required by the shipped inference path.

## Runtime boundary today

| Stage | Current implementation | Next work |
| --- | --- | --- |
| PCM audio to eight RVQ streams | Native HeartCodec F32 encoder | Retain documented backend/numerical limits |
| Transcription | Native SheetSage2 in sibling yuey.cpp | Consume its public interface; do not duplicate the model |
| MIDI, prompt tokens and Qwen style embedding | Upstream Python reference preparation | Native conditioning preparation and Qwen/tokenizer inference |
| Guide selection, duration/cache planning | C++ continuation and raw-PCM job capabilities | Host supplies original-clock cursor/sample counts |
| Guide priming, generator, CFG, Philox, EOS/reset | Native C++ job with progress/cancellation | Integrate with the host worker |
| RVQ/flow/SQ to stereo | Native shared HeartCodec | Retain staged memory ownership |
| Generated-audio suffix | C++ continuation capability and CLI | DAW placement/resampling/join policy belongs to the host |

The [native PCM job](NATIVE_AUDIO_JOB.md) now runs from **raw PCM and prepared
conditioning** through continuation output without launching Python. The older
cached-RVQ command below remains available. Custom text/Qwen/MIDI preparation
still uses the reference bridge; the inference path accepts cached presets.

## Reusable C interface

`include/mulacover_continuation.h` adds
`mulacover_get_continuation_api(1)`. Core, sampling, tempo, HeartCodec and RNG
tables retain their layouts. Link through `mulacover::mulacover`, which uses the
existing `heartcodec::heartcodec` and single shared GGML target.

The capability is model-independent:

- `plan` validates a source interval, requested new frames, prompt length and
  cache budget. Zero guide length selects all remaining source frames; a zero
  source length permits ordinary generation without a guide. Zero cache capacity
  selects the smallest multiple of 256 that holds the run, up to 8192 positions.
  Inputs and results use integer RVQ frames (80 ms each), not floating durations.
  Invalid inputs leave the result unchanged.
- `copy_audio` takes borrowed native HeartCodec planar stereo PCM and copies
  only the suffix after the guide. The result owns its memory independently of
  the input and all inference contexts. Release it through the same capability's
  `audio_free` before unloading the library. It changes no sample values.

The host can use the existing core `teacher_frame` calls for selected guide
codes, followed by sampling calls for new frames. Guide frames consume no RNG.
Release the generator before loading/decoding the full guide-plus-new token
sequence through HeartCodec. Then obtain just the new PCM with `copy_audio`.
Inference runs on a worker, outside the realtime audio callback.

## Standalone native CLI

`mulacover-generate` now directly accepts `--guide-start-seconds`,
`--guide-seconds`, and `--continuation-seconds`. Its decimal parser rounds down
to the 80-ms grid exactly, including decimal/scientific notation, without
binary floating-point boundary errors. `--frames` still requests total frames
and is mutually exclusive with `--continuation-seconds`.

For the existing cached regression fixture, run from the repository root:

```powershell
$env:GGML_CUDA_TF32='0'
$env:GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
./build-windows-development-cuda/bin/Release/mulacover-generate.exe `
  --model models/mulacover-q4_k_m.gguf `
  --input refdata/covers/billiegasm-full-acoustic/generation-q4-philox-laptop/input `
  --output refdata/native-continuation/new-run `
  --prompt-frames 25 --symbolic-frames 540 --qwen-index 14 --muq-index 17 `
  --audio-prefix refdata/bpm-prefix/encoded-long/guide-1.i32 `
  --guide-seconds 8 --continuation-seconds 62 `
  --rng torch-cuda --rng-cuda-blocks 288 --backend cuda `
  --flow-model ../heartcodec.cpp/models/heartcodec-flow-f32.gguf `
  --sq-model ../heartcodec.cpp/models/heartcodec-sq-codec-f32.gguf
```

It selects the guide, plans capacity, primes/samples, stages the native decoder,
and writes `codes.i32`, `audio.wav`, `continuation.wav` and `run.json`. Guide
frames remain included in `codes.i32` / `audio.wav`. The new waveform contains
only generated audio. Counts, source start, cache capacity and actual EOS are
reported; a requested duration is a limit, not forced EOS suppression.

For an experimental score-free comparison, add `--score-mode none`. The default
is `provided`. The native CLI does not open `piano.f32`, `drum.f32`, `chord.f32`
or `context-mask.u8` in `none` mode. Instead it supplies -1 for all three symbolic
streams and an unpadded mask at the requested `--symbolic-frames` length. This
is upstream `modeling.py::_mask_condition`'s dropped-condition convention;
zeros would mean an empty score. Text, Qwen style, audio guide, CFG and sampling
are retained. The ordinary symbolic adaptors/cross-attention remain active.
The upstream public preprocessing frontend requires a score; this native mode
is an experiment using its existing null-condition representation, not an
upstream-supported audio-only pipeline or a claim of training-time coverage.
See [the listening comparison](SCORE_FREE_CONTINUATION.md).

`tools/generate.py` is now a reference preprocessing/provenance harness. It
passes the original guide and requested durations into the C++ executable and
independently checks the selected codes, timeline and exported waveform. It
does not perform production guide slicing or write the continuation WAV.
`tools/continuation.py` remains an independent Python reference for tests.

## Next gates, in order

1. **Retain the frozen past-only baseline and broaden musical validation.** The
   original score-guided transformation remains a useful separate workflow.
   A new score-free baseline uses only source seconds 0--29.76 and compares
   4/8/16-second tail guides ending at that cursor. The user hears a seamless,
   wilder but interesting handoff in the eight-second case. A follow-up also
   accepts a full 50-second guide and generates 6.8 new seconds before EOS.
   See [the cursor-ending guide experiment](CURSOR_CONTINUATION.md) for exact
   boundaries, native commands, frozen reports and remaining limitations.
   Expand listening across sources and seeds; these cases establish neither
   precise melody preservation nor a hard BPM lock.

2. **Maintain the native waveform-to-RVQ encoder in HeartCodec.** SQ encoding is a
   different representation and does not replace this stage. Frozen
   FP32 outputs for input resampling/chunking, the Whisper/WavLM/MuEncoder
   feature branches, feature fusion/query transformer and shared RVQ now underpin
   the native F32 GGUF implementation and separate negotiated encoder capability.
   See sibling HeartCodec's `docs/AUDIO_ENCODER.md`, `WHISPER_PARITY.md` and
   `RVQ_BOUNDARY.md` for exact code matches and still-open strict activation gates.
   The PCM job stages it separately from generator/decoder. Preserve those gates;
   use Spark Docker for reference work and keep encoder quantization deferred.

3. **Port conditioning preparation.** Native MIDI-to-pianoroll/chord assembly,
   text tokenization, prompt layout and Qwen style embedding remove the remaining
   runtime Python dependency. Reuse sibling library patterns and existing GGML
   operations. Cached prepared presets can support interim native experiments;
   they are not a native custom-style frontend.

4. **Integrate the native job into the DAW worker.** Its shared C APIs provide model/RNG lifecycle,
   progress/cancellation between inference steps, selected context ending at
   the cursor, and only-new-audio output. Measure latency and peak memory.
   Playback-rate conversion and the final join to the existing clip must be
   tested in the host's sample timeline.

The original capability described below covered cached-code control/export;
the newer PCM job also incorporates the full encoder port. Numerical gates
remain unchanged. The prior full-song strict
decoder waveform comparison with upstream still fails and requires diagnostics.
No claim of a hard BPM lock or arbitrary-song quality follows from this work.
Shared GGML stays at the agreed baseline with the already-authorized isolated
development override; no sibling bump or ggml-bump-check is initiated.

Validation is recorded in `docs/validation/native-continuation.json`:

- Eight native tests pass on Windows CUDA, Windows CPU shared/DLL, and Spark
  CUDA in the dedicated development Docker image. They include exact decimal
  boundaries, selection/cache limits and overflow, C table negotiation,
  unchanged results on invalid input, occupied outputs and owned audio lifetime.
- The direct C++ 70-second run selects eight guide seconds from the cached
  29.76-second encoding and generates 62 new seconds. All 7,000 codes,
  `audio.wav`, and `continuation.wav` match the existing Q4 native demo
  byte-for-byte. Both replay passes and EOS/padding/reset contracts pass. This
  regression launch and its hash comparison require no Python.
- The optional reference harness passes all 17 Python tests and a separate
  128-frame resume test; all 1,024 control codes match. The harness delegates
  planning/selection/export to the native executable and validates its outputs.

The regression reuses existing prepared conditioning and is not the past-only
DAW fixture described above. Its exact waveform agreement is with the earlier
native Q4 demo; it does not resolve the separate strict F32 upstream waveform
comparison failure.

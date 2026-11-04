# Native continuation and next milestones

The runtime target is a C++ DAW worker that accepts audio available before the
cursor, prepares conditioning, primes the generator, and returns only new audio.
Python is retained for reference oracles, checkpoint conversion and analysis;
it should not be required by the shipped inference path.

## Runtime boundary today

| Stage | Current implementation | Next work |
| --- | --- | --- |
| PCM audio to eight RVQ streams | Official HeartCodec encoder in Spark Docker, FP32 CPU | Port to shared `heartcodec.cpp` |
| Transcription | Native SheetSage2 in sibling yuey.cpp | Consume its public interface; do not duplicate the model |
| MIDI, prompt tokens and Qwen style embedding | Upstream Python reference preparation | Native conditioning preparation and Qwen/tokenizer inference |
| Guide selection, duration/cache planning | C++ continuation capability and CLI | Host uses integer RVQ frame counts |
| Guide priming, generator, CFG, Philox, EOS/reset | Native C++ | Host worker progress/cancellation and lifecycle integration |
| RVQ/flow/SQ to stereo | Native shared HeartCodec | Retain staged memory ownership |
| Generated-audio suffix | C++ continuation capability and CLI | DAW placement/resampling/join policy belongs to the host |

Thus the new native command runs from **prepared conditioning and cached RVQ
input** through continuation WAV output without launching Python. It does not
yet implement WAV-in continuation from arbitrary raw audio by itself.

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

`tools/generate.py` is now a reference preprocessing/provenance harness. It
passes the original guide and requested durations into the C++ executable and
independently checks the selected codes, timeline and exported waveform. It
does not perform production guide slicing or write the continuation WAV.
`tools/continuation.py` remains an independent Python reference for tests.

## Next gates, in order

1. **Freeze a past-only DAW baseline.** The existing musical demonstration uses
   the beginning of a longer encoded clip and a whole-song symbolic score.
   That is useful regression data, but not a strict cursor-only experiment.
   For the next fixture, place the cursor at the end of available audio; encode
   and transcribe only that audio, use its last selected guide frames, and
   generate beyond the cursor. No future waveform or future score is supplied.
   Compare 4/8/16-second guides and keep encoder analysis context explicit.

2. **Port the full waveform-to-RVQ encoder into HeartCodec.** SQ encoding is a
   different representation and does not replace this missing stage. Freeze
   FP32 outputs for input resampling/chunking, the Whisper/WavLM/MuEncoder
   feature branches, feature fusion/query transformer and shared RVQ. Convert
   weights to GGUF, port each stage against those fixtures, then expose the
   result through a separate negotiated HeartCodec capability. Keep existing
   representations and tables frozen. Measure laptop memory with the encoder
   staged separately from the generator and decoder; quantize only after F32
   validation. Use Spark reference Docker for oracles, never host Python.

3. **Port conditioning preparation.** Native MIDI-to-pianoroll/chord assembly,
   text tokenization, prompt layout and Qwen style embedding remove the remaining
   runtime Python dependency. Reuse sibling library patterns and existing GGML
   operations. Cached prepared presets can support interim native experiments;
   they are not a native custom-style frontend.

4. **Wire the DAW worker.** Use the shared C APIs, explicit model/RNG lifecycle,
   progress/cancellation between inference steps, selected context ending at
   the cursor, and only-new-audio output. Measure latency and peak memory.
   Playback-rate conversion and the final join to the existing clip must be
   tested in the host's sample timeline.

The first implementation here is native runtime control and export, not the
full encoder port. Numerical gates remain unchanged. The prior full-song strict
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

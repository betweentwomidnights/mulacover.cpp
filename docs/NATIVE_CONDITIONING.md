# Native conditioning preparation

`mulacover_conditioning.h` adds a separate negotiated C capability for UTF-8 text,
prompt construction, symbolic score tensors and the unquantized Qwen3-Embedding-
0.6B style encoder. `mulacover-continue` can now run text + raw PCM through stereo
continuation without a prepared tensor directory, Python, Torch or subprocesses.
Existing prepared-input workflows and all earlier ABI layouts/defaults remain.

## Validation and limits

The accepted 29.76-second guide / 57.84-second continuation reproduces **all 8,760
codes and both WAVs byte-for-byte** when prompt/style preparation is native too.
The model still emits EOS after 723 new frames despite the 90-second ceiling.
Native preparation takes 3.54 seconds in this run; the full native pipeline takes
260.04 seconds. Sampled total laptop GPU usage peaks at 7,222 MiB, including the
desktop, at 200ms sampling intervals. This is one accepted source/seed/preset,
not a guarantee for other prompts, durations or musical material.

Both tokenizers, Python-compatible lowercase/strip and special-token handling
match 262 frozen text cases exactly. Native MIDI assembly matches the synthetic
fixture and all 540 symbolic rows of each real instrumental/vocal-lane fixture.
Dedicated MIDI edge fixtures cover FIFO overlapping notes, half-grid ties,
velocity-zero note-offs, tracks/channels and tempo changes. Nine MIDI fixtures
rejected upstream are also rejected natively without publishing output.

Five unquantized style cases cover original lyrics/tags, instrumental tags,
multilingual/decomposed text, empty style and special tokens. CUDA on the laptop
and Spark passes the unchanged max-absolute 0.001 / relative-L2 0.0001 gates for
the sampled embedding, layer 1, layer 14, final hidden state and final embedding.
The final embedding also passes against actual upstream `_encode_style` using
its default SDPA implementation, independently of the eager hidden-state oracle.
Laptop final embedding maximum error against that stock control is 2.98e-7.

**CPU intermediate parity is still open.** CPU final embeddings pass all five
cases, including the stock upstream control (maximum error 4.51e-7), but four
layer-14 cases fail the absolute gate against the Spark Torch CPU oracle. Worst
max_abs is 0.01318359375 with relative L2 2.03e-6 on an activation around 6,470.
Upstream CPU and CUDA themselves differ at this stage; that observation does
not waive the native CPU failures or establish their cause. CPU comparison is
also across laptop x86 and Spark ARM hardware. Preserve those failing reports;
do not describe complete CPU activation parity as passed. No tolerance changes,
RNG changes, quantization, hidden backend fallback or GGML changes were made.

Full results and identities are in [the summary](validation/native-conditioning.json),
[laptop CUDA](validation/conditioning-cuda.json), [CPU](validation/conditioning-cpu.json),
and [Spark CUDA](validation/conditioning-spark-cuda.json). Earlier encoder
activation/RVQ limits and the separate strict upstream full-song waveform failure
remain unchanged. This work establishes exact replay of the accepted native
continuation, not a new upstream decoded-waveform parity result.

## Capability and ownership

Negotiate `mulacover_get_conditioning_api(1)` and verify its size/version. Create
a context from the tokenizer GGUF; it holds CPU maps only and can prepare multiple
requests. Request strings and note arrays are borrowed until `prepare` returns.
All text is literal UTF-8; the API never interprets strings as filenames. CLI
text files are explicit. Each text input is capped at 256 KiB; invalid UTF-8 and
empty/whitespace-only lyrics fail.

Initialize an empty result before preparation. The owned result exposes a
`mulacover_condition_v1` view that can be passed directly to the existing job or
generator API. It holds one conditional branch; the generator handles CFG.
Keep that result alive until the consuming call returns. Results survive
conditioning-context destruction, and must be released with this capability's
`result_free` before unloading the DLL. Never free them with a host allocator.
Failure/cancellation leaves output unchanged; free before reusing occupied output.

Requests and context calls belong on a serialized DAW worker. Cancellation is
cooperative before work and between Qwen layers/graphs; a running graph and
initial loading are not interrupted. The callback's completed/total values count
Qwen layers. Callbacks execute on the calling worker, cannot throw or reenter,
and are not retained after return. Another thread can signal an atomic flag that
the cancel callback reads. Status 3 means observed cancellation, 1 invalid input,
2 runtime failure. Diagnostic dumps, if requested, may remain after a failed call.

Qwen is loaded only for nonempty normalized tags and destroyed before `prepare`
returns. Then the existing job stages encoder, generator, flow and SQ. A retained
conditioning context or prepared result keeps no GPU model resident. Empty tags
produce an exact 1,024-zero style vector without loading Qwen. MuQ remains the
512-zero placeholder used by this upstream pipeline; no audio MuQ model is added.
The style GGUF is strict F32 (312 tensors including two RoPE caches). Generator
Q4_K_M remains separate; style quantization is not implemented.

## Text and score semantics

Lyrics use upstream Python lowercase. Tags use strip/lower and remove one outer
`<tag>...</tag>` wrapper. Prompt construction retains all configured delimiters,
eight empty audio columns, text-only masks and exact Qwen/MuQ replacement indices.
The pinned prompt tokenizer uses byte BPE with `ignore_merges` and three-digit
groups. The style tokenizer uses its own byte BPE, NFC, single-digit groups and
appended EOS. Added tokens are recognized before normalization. Native Unicode
tables are generated offline from the pinned Rust tokenizers' regex/NFD/NFC and
Python's lowercase behavior; builds/runtime do not require Python or a Unicode
library. YuE2's native Unicode implementation supplied the UTF-8/NFC structure;
ACE-Step's Qwen encoder supplied the graph reference. Attribution is retained.

Provided scores accept either `(onset, pitch, duration)` arrays on the integer
sixteenth grid or format 0/1 PPQ MIDI paths. Do not mix the two input forms.
Melody is required; chords and drums may contain no notes. Chords reduce pitch
modulo 12; zero-duration chords drop, while melody/drum durations clamp to at
least one sixteenth. Melody has separate onset/sustain columns; drums/chords
mark sustained occupancy. MIDI channel 9 is excluded from melody/chords, while
an explicit drum MIDI uses every channel. Matching same-pitch note-offs is FIFO
per track/channel. Quantization uses integer half-up arithmetic. Tempo remains
metadata; it does not condition or lock generated BPM. Maximum symbolic length
is 5,000 sixteenths; no silent trimming occurs. `symbolic_frames=0` derives the
provided-score length; a nonzero value requires an exact match.

Score `none` builds the upstream dropped-score representation (-1 tensors, zero
padding mask) at the explicit positive symbolic length. It is the existing
experimental score-free mode. The frontend performs no transcription or melody
lane selection. Consume SheetSage2 through yue2.cpp's public transcription/MIDI
interface and choose vocal/instrumental lanes deliberately. The caller must trim
transcription to the cursor for strict past-only continuation; an optional future
arrangement is a separate score-guided transformation workflow. The PCM job still
narrows audio before encoding and does not infer whether score/style inputs use
future information.

## Native commands

From this repository with the current development GGML build and local assets:

```powershell
$env:GGML_CUDA_TF32='0'
$env:GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
./build-windows-development-cuda/bin/Release/mulacover-continue.exe `
  --audio C:/Users/thegr/Music/billiegasm_yuey.wav `
  --tokenizers-model models/mulacover-conditioning-tokenizers.gguf `
  --style-model models/mulacover-style-f32.gguf `
  --lyrics-file refdata/native-conditioning/instrumental/lyrics.txt `
  --tags-file refdata/native-conditioning/instrumental/tags.txt `
  --encoder-model ../heartcodec.cpp/models/heartcodec-encoder-f32-v2.gguf `
  --model models/mulacover-q4_k_m.gguf `
  --flow-model ../heartcodec.cpp/models/heartcodec-flow-f32.gguf `
  --sq-model ../heartcodec.cpp/models/heartcodec-sq-codec-f32.gguf `
  --output refdata/native-conditioning/new-run --cursor-sample 1312416 `
  --symbolic-frames 540 --natural-max-seconds 90 --include-full-audio 1
```

For score-guided mode add `--score-mode provided --melody-midi ... --chord-midi ...`
and optional `--drum-midi ...`; omit `--symbolic-frames` to derive its length.
`mulacover-prepare` supports the same native conditioning inputs and writes a
standalone preset directory. `mulacover-continue --input ...` preserves the cached
path; do not mix its prompt/index options with native text arguments. Native
continuation also exports the prepared tensors under `conditioning/` for audit.

Reference preparation, model conversion, Unicode generation and comparison remain
offline tools. Run Spark work only through the dedicated Docker workflow:
`reference/run-spark.sh conditioning`, `conditioning-controls`,
`conditioning-unicode`, `conditioning-text-score`, and
`conditioning-native-development`. Conversion pins the existing generator/style
checkpoints; model/tokenizer GGUFs and raw fixtures stay outside Git. Fresh native
output directories are required. The shared pin remains b77ae79 and the authorized
isolated CUDA target remains 0f2ed045; coordinated adoption is still deferred.

# Native server and continuation studio

`mulacover-server` is a local staging service for gary4local/gary4juce integration.
It wraps the existing negotiated conditioning and job C APIs. One worker owns
inference; HTTP threads only submit, poll, cancel and retrieve results. No runtime
Python, subprocess inference, Node, CDN or web build tooling is needed. The HTML,
CSS and JavaScript are embedded into the executable with CMake.

`mulacover_runtime.h` separately negotiates model-free library/GGML metadata.
The server reads it through MuLaCover's ABI, so a DLL server does not link another
static GGML instance merely to query its version. Earlier tables stay frozen.

## Start on the laptop

Build the normal coordinated GGML target:

```powershell
./scripts/build-windows.ps1 -Backend cuda
$env:GGML_CUDA_TF32='0'
$env:GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
./build-windows-cuda/bin/Release/mulacover-server.exe `
  --model-dir C:/dev/mulacover.cpp/models `
  --codec-model-dir C:/dev/heartcodec.cpp/models
```

Open <http://127.0.0.1:8010/>. The default port is 8010, configurable with `--port`.
The server binds to loopback. Configure an exact `--allow-origin URL` for a
separate downstream browser frontend; arbitrary web origins are rejected.
Models are resolved at startup from the supplied directories:

| Directory | Files |
| --- | --- |
| `--model-dir` | `mulacover-q4_k_m.gguf`, `mulacover-conditioning-tokenizers.gguf`, `mulacover-style-f32.gguf` |
| `--codec-model-dir` | `heartcodec-encoder-f32-v2.gguf`, `heartcodec-flow-f32.gguf`, `heartcodec-sq-codec-f32.gguf` |

Both directories default to `models` relative to the launch working directory;
use absolute paths for a supervisor. `--backend` selects `cpu`, `cuda`, `vulkan`
or `metal`; it never falls back. The default follows the compiled backend.
CUDA defaults to Torch CUDA Philox (cap 288); other backends use explicit native
RNG. This is the existing capability behavior, not equivalent RNG across backends.

`--version` and `--props` exit without creating model/device contexts. Version
comes from CMake's project version. Props/health include the configure-time source
commit and dirty flag, linked GGML identity, selected backend and model-file
presence. Presence is not a model-integrity or inference-readiness check.
Disable this executable/support code with `-DMULACOVER_BUILD_SERVER=OFF`.

## HTTP v1

This is a staging contract with the sibling session/poll pattern; it is not yet a
drop-in gary4local adapter. Poll returns WAV URLs, rather than a base64 FLAC payload.

| Method | Route | Result |
| --- | --- | --- |
| GET | `/` | Embedded continuation UI |
| GET | `/health` | Model-free status, presence and `busy` |
| GET | `/props` | Version, backend, limits and capabilities |
| POST | `/continue` | Multipart submission; HTTP 202 with `session_id` |
| GET | `/poll_status/<id>` | State, monotonic fraction 0..1, stage, step counts, metadata and audio URLs |
| POST | `/cancel/<id>` | Cooperative cancellation; terminal status appears through polling |
| GET | `/audio/<id>` | New-audio-only 48 kHz stereo F32 WAV |
| GET | `/full_audio/<id>` | Optional reconstructed guide + new audio |
| POST | `/release/<id>` | Release a terminal session/result; 409 while active |
| POST | `/unload` | Clear retained results when idle; models already unload per job |

Multipart parts are `audio` (PCM WAV), `options` (a JSON object), and optional
`melody_midi`, `chord_midi`, `drum_midi` uploads. Uploaded filenames are ignored;
MIDI goes into a private temporary directory under fixed names and is removed on
all exits. Requests cannot supply model paths, output paths, executable commands
or prepared tensor directories. WAV decoding uses the same native parser as the
CLI, with 16/24/32-bit integer or F32 mono/stereo RIFF PCM support.

Example `options`:

```json
{
  "lyrics": "[Instrumental]",
  "tags": "warm acoustic pop",
  "score_mode": "none",
  "cursor_sample": 1312416,
  "context_samples": 1312416,
  "guide_samples": 1312416,
  "natural_max_seconds": 30,
  "seed": "1701",
  "include_full_audio": true
}
```

Cursor/context/guide are **original WAV sample counts per channel**, not 48 kHz
output samples or browser-resampled positions. Cursor defaults to the input end;
zero context/guide means all available past/context respectively. Context ends
at the cursor and only its tail is used as the guide. Lengths exceeding the
available past are clamped by native planning. Native planning drops
partial 80ms frames. The UI reads the WAV header to convert its seconds controls
to this clock; it does not use AudioContext's device-rate sample count.

`natural_max_seconds` is a new-audio ceiling, independent of guide/context length.
Early EOS is honored. The default is 30 seconds. Native planning and the default
1536-position server cache budget bound the combined prompt/guide/new sequence.
`--max-cache-capacity` can change that budget (256..8192); larger values are not
a claim that the laptop can allocate or use them effectively. Actual prompt
length is checked again after preparation and before encoder/generator work.

`score_mode=provided` requires melody MIDI; chord/drum MIDI may be omitted for
empty lanes. Score-none rejects any MIDI uploads. Score-none defaults to 540
symbolic rows; `symbolic_frames` can override it. Provided mode derives the score
length unless an explicit matching length is supplied. There is no transcription
inside this server. Use SheetSage2 through yue2.cpp and choose lanes explicitly.
For strict past-only continuation the caller must also trim the score before the
cursor. Future arrangements remain available for score-guided transformation.

Other options: `guidance` (1.5), `temperature` (1), `top_k` (250),
`cache_capacity` (0/automatic), `decoder_seed` (1701, independent stream),
`decoder_steps` (10), `segment_codes` (372), `rng` and `rng_cuda_blocks` (288).
Pass uint64 seeds as decimal strings to avoid JavaScript's integer precision loss.
Unknown/duplicate fields, invalid geometry, unsupported score/RNG combinations
and malformed uploads are rejected. Lyrics/tags are bounded to 16 KiB each,
options to 64 KiB, MIDI to 4 MiB per lane, and the HTTP body to 128 MiB.

## Ownership, completion and cancellation

Only one job is admitted at a time; another submission returns 409. A worker
prepares text/style/score, releases Qwen, then stages encoder → generator → flow
→ SQ through the public APIs. Inputs live through the synchronous calls; owned
results are freed with their originating capability before any library unload.
No frozen ABI layouts or legacy defaults are changed.

Poll states are `queued`, `running`, `completed`, `cancelled`, `failed`.
Completed metadata includes frames, new duration, EOS, exact cursor/context start,
seeds/Philox offsets and stage timings. Failed/cancelled jobs publish no audio.
Cancellation is checked between graphs/layers/frames/windows; loading weights and
a running backend graph cannot be interrupted. Shutdown cancels and joins the
worker. GPU contexts are not retained between jobs. Retained WAV results expire
after 30 minutes or eviction of the oldest terminal result (four sessions max).
Download before calling release/unload or starting another UI session.

Tests inject an internal runner to exercise admission, failure recovery,
cancellation and shutdown without weights. There is no fake-inference server
option. Real CUDA HTTP/CLI and browser checks are separately recorded in
`docs/validation/server.json`. Musical quality and all existing numerical parity
limits keep their earlier scope; these short checks validate service wiring.
Release packaging, installed SDK exports and the final gary4local adapter are
separate work. No release workflow is dispatched by this implementation.

# Native server and continuation studio

`mulacover-server` is a local staging service for gary4local/gary4juce integration.
It wraps the existing negotiated conditioning and job C APIs. One worker owns
inference; HTTP threads only submit, poll, cancel and retrieve results. No runtime
Python, subprocess inference, Node, CDN or web build tooling is needed. The HTML,
CSS and JavaScript are embedded into the executable with CMake.

`mulacover_runtime.h` separately negotiates model-free library/GGML metadata.
The server reads it through MuLaCover's ABI, so a DLL server does not link another
static GGML instance merely to query its version. Earlier tables stay frozen.

## Basic UI

Upload a WAV, choose a mode and **Audio context duration**:

- **Continue audio:** score-free continuation from the opening of the upload.
  Choose **Maximum new audio** for this mode.
- **Transcribe + remix:** SheetSage2 automatically transcribes the full uploaded
  song; its combined melody and chords condition the remix, while the chosen
  opening audio primes MuLaCover. No manual MIDI upload is needed.
  Duration is derived from the uploaded song; there is no selectable new-audio
  maximum. The primary playback/download is the full remix, including context.

Remix mode sends `match_source_duration=true` and omits `natural_max_seconds`.
The server targets floor(source duration / 80ms) total frames and subtracts the
opening guide frames to derive the remaining generation ceiling. It always
returns full audio for this policy. A context consuming every complete source
frame is rejected: shorten it to leave room for generated audio. Source-length
matching requires automatic full-source transcription and `audio_context_seconds`;
it cannot be mixed with an explicit new-audio duration.

This is a target, not an EOS override. The model may end early, and the codec's
80ms frame grid can omit a partial source tail. The UI reports target and actual
total duration and identifies early endings. No silence padding, time stretching,
EOS suppression or forced minimum is applied. The continuation-only API remains
available independently of the remix's full-audio download.

The default server cache budget is 2048 positions. Audio frames and text prompt
share that budget; the planner rounds allocations up to 256 positions. A
132-second source needs at least 1792 positions with a short prompt, so the
former 1536-position server limit rejected it before inference. Shortening a
remix guide does not reduce its whole-source target. Budget errors now report
the required/configured position counts. `--max-cache-capacity` configures a
server ceiling, not a request to allocate that much for every job. A long score
also consumes device memory; the 8192-position model limit is not a laptop
memory guarantee.

For example, 30 seconds of context always uses source seconds 0..30. There is no
cursor or separate tail-guide control in this basic UI. Durations are floored to
complete 80ms frames **before** choosing the opening; a 1-second request uses
0..0.96, never 0.04..1. The selected opening is shown under the duration controls.
The maximum new duration permits early EOS. Melody focus (all/singer/instruments)
is optional under generation settings. Manual MIDI and arbitrary-cursor controls
remain available to API callers for future DAW workflows.

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

Transcribe + remix reuses a local native `yue2-server`/SheetSage2 service on port
8011. Run that service separately (the future application supervisor can own it):

```powershell
yue2-server --port 8011 --device cuda --force-unload `
  --models-dir C:/dev/yue2.cpp/models `
  --transcription-model C:/dev/yue2.cpp/models/sheetsage2-mert2-f16.gguf
```

`--transcription-port N` selects another loopback provider; 0 disables it.
The bridge sends `keep_models=false`, waits for the owned transcription session
to complete and release its model, then prepares conditioning and runs MuLaCover.
It never launches a CLI/Python process or globally unloads another service's jobs.
Cancellation cancels/polls only the session it submitted. A missing provider is
an explicit job error; score-free continuation remains independent. The provider
is a separate process, not a second GGML target linked into MuLaCover. Model-free
props report configuration, not live provider readiness.
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

Health and completed-job metadata include `optimization_requests`, the three
generator cache/workspace flags requested by the process environment at context
creation. They are not hit counters; use `MULACOVER_PROFILE=1` native logs for
actual cache hits and persistent storage. The UI labels optimized/standard
requests, and its connection badge tooltip gives the build commit. Completed
metadata also preserves lyrics/style, guidance, temperature and top-k alongside
the resolved sampling/decoder seeds and timeline.

For a separate development executable while another server is running, configure
`-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=C:/dev/mulacover.cpp/refdata/verified-ui` and
build normally. Visual Studio places executables under its `Release` child.
Do not relocate a build with MSBuild `/p:OutDir`: CMake's explicit static-library
dependencies retain their original paths and can silently link stale libraries.
Check `--props`/`/health` against the intended commit after staging a build.

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
  "audio_context_seconds": 30,
  "natural_max_seconds": 30,
  "seed": "1701",
  "include_full_audio": true
}
```

The basic `audio_context_seconds` option aligns the context end down to complete
frames from source zero and sets cursor/context/guide together. It cannot be
combined with explicit cursor/context/guide options. The UI uses this option.

For advanced API callers, cursor/context/guide are **original WAV sample counts per channel**, not 48 kHz
output samples or browser-resampled positions. Cursor defaults to the input end;
zero context/guide means all available past/context respectively. Context ends
at the cursor and only its tail is used as the guide. Lengths exceeding the
available past are clamped by native planning. Native planning drops
partial 80ms frames. The UI reads the WAV header to convert its seconds controls
to this clock; it does not use AudioContext's device-rate sample count.

`natural_max_seconds` is a new-audio ceiling, independent of guide/context length.
Early EOS is honored. The default is 30 seconds. Native planning and the default
2048-position server cache budget bound the combined prompt/guide/new sequence.
`--max-cache-capacity` can change that budget (256..8192); larger values are not
a claim that the laptop can allocate or use them effectively. Actual prompt
length is checked again after preparation and before encoder/generator work.

For automatic transcription add `transcribe=true`, `score_mode=provided` and
`transcription_scope=source`. `transcription_lane` selects `combined` (default),
`vocal` or `instrumental`. This is the UI's transcribe + remix path: the full score
is an intentional future arrangement, even though audio encoding sees only the
selected opening. SheetSage2 exports no drum lane. Its native MIDI is passed to
the conditioning capability without an ABC conversion or timing rewrite.

The API also supports `transcription_scope=context` (default) for strict past-only
workflows. It sends only the exact planned context window, never future PCM.
This short score can itself encourage EOS after the guide; it is distinct from
the full-song remix workflow in the basic UI.

Manual `score_mode=provided` requires melody MIDI; chord/drum MIDI may be omitted for
empty lanes. Score-none rejects any MIDI uploads. Score-none defaults to 540
symbolic rows; `symbolic_frames` can override it. Provided mode derives the score
length unless an explicit matching length is supplied. Automatic transcription
and manual MIDI uploads cannot be mixed. The server orchestrates SheetSage2
through yue2.cpp's public HTTP interface; no transcription implementation is copied.
For strict past-only continuation the caller must also trim the score before the
cursor. Future arrangements remain available for score-guided transformation.

Other options: `guidance` (1.5), `temperature` (1), `top_k` (250),
`cache_capacity` (0/automatic), `decoder_seed` (1701, independent stream),
`decoder_steps` (10), `segment_codes` (372), `rng` and `rng_cuda_blocks` (288).
Experimental `overlap_frames` (integer 0--8192, default 0) regenerates a past
guide tail while preserving the original new-audio boundary. The Continue audio
UI offers an unchecked 0.64-second option. See `OVERLAP_CONTINUATION.md` for
ownership, RNG and preview semantics and the limited listening evidence.
Pass uint64 seeds as decimal strings to avoid JavaScript's integer precision loss.
The UI/server seed defaults to `-1` (number or string), resolved once per accepted
job to a fresh uint64 seed. Admission/poll responses expose that resolved decimal
string, including failed or cancelled jobs. Re-enter it to replay. Unless supplied
explicitly, `decoder_seed` follows the resolved sampling seed; an explicit decoder
seed overrides it. Native C ABI and CLI unsigned seeds/defaults remain unchanged;
UINT64_MAX remains a valid fixed seed and is not a random sentinel.
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
option. Initial CUDA HTTP/CLI and browser checks are recorded in
`docs/validation/server.json`; simplified UI/transcription checks are in
`docs/validation/transcribe-remix-ui.json`. Musical quality and all existing numerical parity
limits keep their earlier scope; these short checks validate service wiring.
`--diagnostic-dir DIR` and `MULACOVER_EOS_TRACE=1` enable local investigation;
both default off. See `SERVER_LIFECYCLE_EOS.md` for artifacts and privacy/lifetime.
Release packaging, installed SDK exports and the final gary4local adapter are
separate work. No release workflow is dispatched by this implementation.

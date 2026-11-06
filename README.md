# mulacover.cpp

Native MuLaCover generator work for gary4local/gary4juce. This repository owns
symbolic MIDI conditioning, prompt/style preparation, the autoregressive model,
sampling and its reference fixtures. Sibling `heartcodec.cpp` owns RVQ/flow/SQ
decoding and will be consumed as `heartcodec::heartcodec`, sharing one copy of
`betweentwomidnights/ggml` with the generator and other model libraries.

Clone the sibling development repositories together:

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/heartcodec.cpp.git
git clone https://github.com/betweentwomidnights/mulacover.cpp.git
```

Source, compact validation reports and tooling are tracked. Weights, generated
audio and raw fixtures remain outside Git. Project contributions use the MIT
license; see [third-party and model terms](THIRD_PARTY_NOTICES.md) for upstream
components and separate checkpoint/output terms. This is a development library;
Q4 quality and the remaining numerical gaps are documented below.

**Native conditioning now accepts UTF-8 lyrics/tags and MIDI or score triplets.**
The separate [conditioning capability](docs/NATIVE_CONDITIONING.md) builds prompt,
score and F32 Qwen style tensors, then releases Qwen before the audio job starts.
At the recorded earlier pin, text + raw PCM reproduces the accepted long
continuation byte-for-byte on the laptop without a cached prompt/style preset
or Python inference. CUDA passes
the sampled style gates; CPU final embeddings pass but intermediate parity remains
open. Transcription/lane selection stays with SheetSage2 and its downstream host.

**BPM is not clock-locked.** A [controlled BPM study](docs/BPM_CONTROL.md) confirms
that MIDI tempo metadata leaves every model input unchanged; 80/160-BPM text
requests also missed their targets in the tested Q4 excerpts. Plan downstream
timing separately rather than treating BPM tags as a guarantee.
The `experiment/score-clock` branch investigates inference-time score attention
and audio-token guides; see [the experiment](docs/SCORE_CLOCK_EXPERIMENT.md).
Audio priming follows the source's roughly 120 BPM in full-length F32/Q4 tests,
with eight guided seconds and 62 new seconds. It remains an experimental
continuation mode; arbitrary BPM, complete restyling and a hard lock are unproven.
The user reports that this continuation fits the intended DAW use case.
Guide start/duration and new-audio duration are independently selectable in the
prepared-input driver; it also exports a guide-free `continuation.wav`.
These controls and waveform export now run directly in the C++ CLI and a
negotiated C capability. See [native continuation and the encoder roadmap](docs/NATIVE_CONTINUATION.md)
for the native runtime boundary and offline reference tooling.
The native CLI also accepts experimental `--score-mode none` to remove symbolic
conditioning while retaining the audio guide, text and style. Both score-guided
transformation and score-free continuation are useful workflows to evaluate;
see [the controlled comparison](docs/SCORE_FREE_CONTINUATION.md).
The [cursor-ending guide baseline](docs/CURSOR_CONTINUATION.md) tests 4/8/16-second
tail guides using only available past audio, plus a 50-second context probe.
Guide length and new audio share a finite cache budget; the model may emit EOS
before the requested end.
A longer native score-free run uses 29.76 seconds of input and produces 57.84
new seconds before EOS; the user positively assesses it as a listening baseline.
The [native PCM job](docs/NATIVE_AUDIO_JOB.md) now accepts a raw audio buffer or
WAV, an original-clock cursor and prepared conditioning, with separate encoder
context/guide lengths, `natural_max_seconds`, progress and cancellation. It returns
only new audio and stages all models on the laptop. Its native conditioning path
also prepares text/Qwen/MIDI inputs. See the [extended run report](docs/validation/extended-continuation.json).

**Native FP32 teacher-forced frame/logit/cache parity passes all 600 tensors on
Spark CUDA at `e3db5210`.** Exact reset/replay also passes; see
[the new-pin validation](docs/validation/ggml-029-adoption.json).
Two CFG activations retain failed flat absolute diagnostics while passing the
existing mixed activation rule; public logits and KV retain their strict limits.
The coordinated baseline is now `e3db5210` (GGML 0.29.0), adopted through
[GGML PR #22](https://github.com/betweentwomidnights/ggml/pull/22), which subsumes
#21's CUDA F32-source fix and Vulkan/Metal precision fixes. The former `b77ae79`
failure remains recorded in [shared-pin validation](docs/GGML_B77_ALIGNMENT.md).
The isolated development override is retired; use the normal shared dependency.
See [coordinated adoption](docs/GGML_029_ADOPTION.md).
The generator implements prompt/audio embeddings, the 28-layer backbone, style
modulation, eight symbolic adaptors and gated cross-attention, the three-layer
local decoder, all eight output heads, CFG and persistent KV state. Both
conditional and CFG=1.5 cases match three frames and replay exactly after reset.
See [the parity report](docs/NATIVE_PARITY.md) and [C ABI](docs/C_ABI_V1.md).

**Experimental Q4_K_M now runs on the 8GB RTX 5070 Laptop**, including CFG and a
fully allocated 8192-position cache. The model is 2.94 GB; sampled GPU usage peaks
at 7,331 MiB including the desktop (3,771 MiB with capacity 256). Only three active
teacher-forced frames are tested. [Quantization results](docs/QUANTIZATION.md)
record substantial logit drift; execution success is not quality acceptance.

**Native sampling now produces eight-second 48kHz stereo clips on the laptop.**
The sampler selects each RVQ head before computing the next and handles CFG,
EOS/padding, deterministic native seeds and reset/replay. The generator unloads
before staged FP32 HeartCodec decoding. See [sampling/audio results](docs/SAMPLING_AND_AUDIO.md),
including the remaining longer-context hidden-activation tolerance gap.

The first [real transcription-to-cover excerpt](docs/FIRST_COVER_TEST.md) now uses
the user's `billiegasm_yuey.wav`: native SheetSage2 exports are audited, then fresh
instrumental conditioning drives native generation and stereo decode. Qwen and
prompt/MIDI preparation still use the upstream Spark Docker bridge. Listening
acceptance for vocals and musical fidelity remains pending.
The [full-song acoustic test](docs/FULL_SONG_TEST.md) now generates all 70 seconds
on the laptop, with full replay checks and a sampled 6,776 MiB peak through decode.
Follow-up [melody-source and F32/Q4 controls](docs/FULL_SONG_CONTROLS.md) verify
that the user-recognized vocal MIDI reaches prepared conditioning, while its
preservation in the generated Q4 cover remains unresolved.
The matching [official upstream FP32 control](docs/UPSTREAM_AUDIO_CONTROL.md)
now produces 68.88 seconds with all 11 freshly prepared inputs matching exactly;
the user hears horns playing the guitar melody without vocals. That original
comparison used different RNG streams.
The [opt-in Philox control](docs/PHILOX_PARITY.md) now matches Torch random tensors
bit-for-bit on both GPUs and all 6,888 native F32 codes/EOS for that full-song case.
The decoded waveform remains close but fails the unchanged strict error limits.

Prepared prompt/style and symbolic inputs can be supplied by the caller or built
through the native conditioning capability. The upstream fixture harness remains
an offline oracle. The laptop is the primary C++
workspace; full FP32 native/reference execution stays in Spark Docker containers.

The original milestone used `07b77b0`; historical reports retain that provenance.
HeartCodec now uses the coordinated `e3db5210` source pin, mounted read-only.
See the historical [pin inventory](docs/GGML_PIN_INVENTORY.md). The precision fix
is included in the coordinated fork. CMake defaults to the in-tree target and reuses
an already-provided GGML target; it never builds a second GGML instance.

`reference/export_reference.py` executes the pinned upstream preprocessing and
generator on original synthetic melody/chord/drum MIDI and original short lyrics.
It freezes prompt tokens/masks/positions, Qwen style inputs and hidden states,
symbolic tensors, selected backbone/style/adaptor states, all eight raw and guided
logits, local-decoder states, and active KV prefixes across three teacher-forced
audio frames. Both conditional and CFG=1.5 cases reset caches and replay exactly;
unused cache tails are checked for zero writes. FP32 and upstream BF16 runs are
kept separately. BF16 fixture values are stored losslessly as float32.

References are pinned to MuLaCover `e27fa7fa0da7c9817879cc6210cc5693f9327946`,
generator weights `bbbaef2b31835c2ef5172ff528dbe325230d46fb`, and Qwen
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
See `docs/validation/` for compact verification reports. Weights and raw fixture
payloads are ignored in Git and available under local `refdata/` and the Spark
mount `/home/kev/dev/mulacover.cpp`.

Reproduce on Spark, using the reference image built by sibling HeartCodec:

```bash
docker run --rm --user "$(id -u):$(id -g)" --gpus all --network none \
  -e USER=heartcodec -e LOGNAME=heartcodec -e HOME=/workspace \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "$PWD:/workspace" -v /home/kev/dev/_heartcodec_refs:/refs:ro -w /workspace \
  heartcodec-reference:20261101 /opt/sa3-venv/bin/python reference/export_reference.py \
  --upstream /refs/MuLaCover --output refdata/upstream-f32 --dtype float32
```

Use `--dtype bfloat16 --output refdata/upstream-bf16` for the BF16 oracle. Run
`reference/prepare_models.py` in the same container with network access,
`HF_HUB_OFFLINE=0`, `HF_HUB_DISABLE_XET=1`, and a persistent `HF_HOME` mount to
retrieve the pinned weights. The immutable reference image ID and build recipe
are recorded in HeartCodec's `docs/SPARK_WORKFLOW.md` and `docker/`.

Native reproduction on Spark:

```bash
bash reference/run-spark.sh convert
bash reference/run-spark.sh build
bash reference/run-spark.sh parity
```

The Windows static/DLL smoke tests build with `scripts/build-windows.ps1` and
`scripts/build-windows.ps1 -Shared`; they use the unchanged in-tree CPU backend.
Use `-Backend cuda` for the coordinated CUDA build (or `-Backend vulkan`).
Local Python checks use `requirements-dev.txt`
(NumPy/GGUF/Mido), then `python -m unittest discover -s tests -v`. Full FP32 weights
remain on Spark; the experimental Q4_K_M model is present on both machines.

Next: add the native server/web UI and stage release packaging, while retaining
the documented numerical limits. The coordinated GGML refresh was validated
across the sibling repositories. Existing yue2.cpp SheetSage2 melody/chord MIDI
exports now feed the [excerpt bridge](docs/TRANSCRIPTION_INTEGRATION.md).
Run `scripts/test-cover-windows.ps1 -CaseName my-first-cover` for the documented
local transcription -> Spark Docker preparation -> local Q4/decode workflow.

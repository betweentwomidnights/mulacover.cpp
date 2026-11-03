# Inference-time score clock experiment

This experiment lives on `experiment/score-clock`. Moving score attention did
not reliably control BPM. **Audio-token priming followed this source's roughly
120 BPM in both full-length F32 and Q4 tests**: eight source-guided seconds plus
62 newly generated seconds. A second, shorter Q4 seed also followed the tempo.
The user reports that the Q4 continuation fits the intended DAW use case and
that the source audio was encoded beautifully. This supports pursuing the
continuation workflow, without establishing a hard beat lock or general cover
quality. Keep it opt-in. Baseline findings remain in
[BPM_CONTROL.md](BPM_CONTROL.md).

## Intervention

The pinned upstream generator uses eight symbolic adaptors and sixteen score
attention modules: one content read and one gate read at each of its last eight
backbone layers. We add a Gaussian log-attention prior to these reads in the
conditional branch only. The unconditioned CFG branch is unchanged.

For prompt length `P`, absolute query position `p`, and score key `k`:

```
audio_frame = p - (P - 1)
score_center = 4 * offset_beats + audio_frame * 0.08 * bpm / 15
log_prior(k) = max(-80, -0.5 * ((k - score_center) / (4 * width_beats))^2)
```

Score units are sixteenth notes, beats are quarter notes, and each generated
RVQ frame represents 80 ms. The last prefill row predicts audio frame zero;
earlier prompt rows receive zero Gaussian bias. The center holds at the first
or last unpadded score row at the boundaries. Original padding stays masked.
The prior is added after the usual attention-score scaling, before softmax.
Backbone and score RoPE, KV addresses, sampling, Philox, audio sample rate and
codec timing are unchanged. No audio stretching is involved.

`--tempo-local-window` additionally restricts the conditional symbolic-adaptor
self-attention keys and both score reads to a moving interval of
`center +/- 4 * width_beats` score rows. It retains the nearest unpadded row if
the interval is empty. During prefill all prompt queries see the initial local
context. This stronger variant prevents adaptor keys from freely mixing distant
score material, but also removes musical context the model normally receives.

The [upstream paper](https://arxiv.org/html/2610.26684v1) explains that BPM
normalization deliberately removes frame alignment and allows the generator to
choose its pace. A moving mask is therefore an inference intervention outside
the stock generation procedure, not discovery of a hidden trained BPM input.

## Interface and ownership

`include/mulacover_tempo.h` provides a separate experimental capability table,
`mulacover_get_tempo_api(1)`. Core and sampling v1 tables and layouts remain
unchanged. The default is disabled (`bpm=0`). Configuration is copied into a
session, set before its first frame, and retained by reset/begin. A running or
terminal session rejects changes without mutating the configuration or caches.
No global model or RNG state is changed.

The native generator and prepared-input Python driver accept:

```
--experimental-bpm 120 --tempo-width-beats 1 --tempo-offset-beats 0
```

The Python driver accepts `--tempo-local-window`; the underlying native CLI
uses `--tempo-local-window 1`. Settings are recorded in `run.json` and the exact
command is retained in provenance. These options are intentionally absent from
the regular transcription-to-cover entry point until there is a useful result.

## Evaluation

Use the existing source's full instrumental score (137 melody and 160 chord
notes, 540 symbolic rows), original acoustic prompt (25 rows), seed 1701,
Torch CUDA Philox with target cap 288, top-k 250, temperature 1, and F32 codec
with 10 steps. Clips request 372 frames / 29.76 seconds. CFG is 1.5 except for
an explicitly labeled stronger-guidance probe. Q4 executes on the laptop;
F32 and the independent Torch oracle execute only in dedicated Spark Docker
containers. GGML candidate and all consumer pins stay unchanged.

`tools/tempo-analysis.py` retains all onset-periodicity candidates and a second
SheetSage2 score-clock estimate. Half-time and subdivision aliases are reported;
a weak candidate near the request is not selected to manufacture adherence.
Baseline clips are exact 372-frame crops of the existing full-song matched-RNG
outputs; their codec was run with full-song context. Fresh clip transcriptions
avoid comparing a short transcription with a full-song transcription.

`tools/score-alignment.py` measures maximum ordered one-to-one pitch-class/onset
matches against the input melody rendered at the requested BPM, with a fixed
250 ms onset tolerance. It permits octave changes, but fits no transposition,
tempo, offset or time warp. This is an automated diagnostic, not proof of
audible melody preservation. SheetSage2 may select another instrument or make
transcription errors; baseline timing differences can also reduce its score.

`reference/export-tempo.py` independently constructs the attention prior in
Torch F64, rounds it to F32, and executes the pinned upstream projections,
RoPE, backbone, symbolic adaptors, gates and local decoder. Only score attention
SDPA receives the additive prior; optional local masking changes the masks
passed into the original adaptor and cross-attention functions. Source and
checkpoints remain unchanged. A zero-BPM case executes the original model.
Teacher-forced public logits and all backbone/local KV tensors are compared
with existing strict gates, including reset/replay and untouched cache tails.
This checks the intervention's implementation, not whether it controls tempo.

Results and hashes are recorded in `docs/validation/score-clock-experiment.json`.
Generated audio, raw tensors and models remain under ignored `refdata/bpm-clock`.
No shared-GGML adoption or sibling regression/bump process is part of this work.

## Audio-token lead-in

A separate intervention uses the existing teacher-forcing API to prime the
backbone and local decoder with a real audio guide before sampling new frames.
The score and style stay unchanged. No attention prior is enabled for these
tests. This is an experimental continuation task, not a trained BPM input.

`reference/encode-prefix.py` uses the clean official HeartLib source at
`5858ca8f1ffe58d62be7c007ea55f1033c5ca456`, the existing decoder/RVQ revision,
and the separate encoder checkpoint at
`e6020510612d6653b73d6cd54393ef80929e3c81`. The checkpoint's SHA256SUMS is
verified. Encoding runs in FP32 on Spark CPU inside the dedicated reference
Docker image: its Torch/CUDA NVRTC cannot compile one complex-spectrogram
operation for GB10. CPU is a supported encoder device; no dependency, CUDA,
GGML or production-container changes are made to work around this issue.

Only private source excerpts (initially eight seconds, then 29.76 seconds for
encoder-context testing) were transferred to the user's Spark. The encoder
processes 29.76-second chunks and pads short inputs. Whether padding contributes
to the observed early EOS is a hypothesis, not an established cause. The full
source stays unchanged on the laptop. Encoder caches, source excerpts, tokens
and generated audio remain outside Git.

The native generator accepts `--audio-prefix FILE` as frame-major
`int32[frames,8]`, with normal audio codes only. The total requested frame count
includes the guide. The output WAV and codes include the guide, so reported
continuation tempo and listening copies **exclude** those forced frames.
The Python provenance hashes the guide and verifies it is copied into output
codes exactly. A guide must leave room for sampling; immediate EOS with no new
audio is an error. An early EOS after only a few new frames is still not a
usable tempo test. Replay and terminal/reset checks also run with priming.

Teacher-forced frames draw no random numbers. New sampling starts at the
requested Philox offset (`--sampler-offset`, default zero); the decoder seed is
still reset separately after generator release. A resume control takes the
first 99 frames of the existing Q4 output and starts sampling at offset
`99 * 8 * 4 = 3168`. All 1,024 codes across 128 frames reproduce the baseline
exactly, including all 232 newly sampled codes. This validates the mixed
teacher-forcing/sampling path and RNG resumption, not source-tempo adherence.

Counterfactual short-guide probes encode the input excerpt at source speed,
0.75x and 1.25x using pitch-preserving `ffmpeg atempo`. Only the **input guide**
is stretched; generated continuations are not. The short lead-ins cover about
four source beats: 25 frames at source speed, 33 at 0.75x, and 20 at 1.25x.
They share prompt, score, seed, sampling and codec settings. The source-speed
Q4 continuation measured 119.8 BPM for 11.92 new seconds, but ended early.
With the longer encoder input, both 70-second requests completed all 875 frames
without EOS. Their eight-second guide uses the first 100 encoded frames; only
the remaining 775 frames count as generated continuation.

| Guide / generator | Newly generated audio | Onset quarter-note candidate | SheetSage2 score clock |
| --- | ---: | ---: | ---: |
| Source speed, Q4, seed 1701, long encoder context | 62.00 s | 119.65 BPM | 118.66 BPM |
| Source speed, F32, seed 1701, long encoder context | 62.00 s | 119.79 BPM | 119.59 BPM |
| Source speed, Q4, seed 1702, long encoder context | 21.76 s | 119.62 BPM | 119.55 BPM |
| 0.75x guide, Q4, short context | 15.84 s | Weak/diffuse periodicity; no confident quarter-note estimate | 85.16 BPM, versus nominal 90 |
| 1.25x guide, Q4, short context | 12.24 s | 148.93 BPM, with a strong 74.26 half-time alias | 74.32 BPM, half-time ambiguity |

The full Q4 onset windows give quarter-note candidates from 119.32 to 120.06
BPM; the F32 windows give 119.63 to 119.86 BPM. These are overlapping 15-second
windows every ten seconds plus the final 15 seconds. Q4's late SheetSage2 clock
fits fall to 116--117 BPM even though onset periodicity stays near 120. Report
that estimator disagreement; do not silently choose whichever supports a lock.
The slower-guide miss prevents claiming reliable control across requested BPMs.
Source-speed replication uses just one song and two seeds, with target style
similar to the source. New styles and songs remain untested.

The full-length Q4 run peaks at 7,540 MiB on the 8,151-MiB laptop, including
desktop usage. It executes both full replay passes and terminal/reset checks,
then staged F32 codec decode. Its generation/checks take 317.53 s and decoding
38.61 s. F32 and the encoder remain Spark reference work; the laptop remains
the Q4 development environment. This does not require a shared-GGML bump.

The first eight seconds are real source-guide tokens, and the saved
`continuation.wav` excludes them. Continuing after selected source audio is the
user's intended DAW use case. Full-song restyling from bar one is a different,
untested task. The positive listening feedback is specific to this continuation;
exact melody/timing agreement and decoder waveform parity remain unaccepted.

## Choosing input and continuation lengths

Guide length and requested new-audio length are independent. The Python driver
now accepts, for example:

```
--audio-prefix encoded/guide-1.i32 --guide-seconds 8 --continuation-seconds 30
```

These options are added to the existing prepared-input generation command.
This requests eight input seconds followed by thirty new seconds (38 total).
`--frames` still means total guide-plus-new frames and is mutually exclusive
with `--continuation-seconds`. The model may emit EOS before the requested end.
Omitting `--capacity` automatically rounds the required cache positions up to
a multiple of 256, within the model's 8192-position context. An explicit
insufficient capacity is rejected before loading the generator.

`--guide-start-seconds` selects a region inside cached encoded audio. For example,
`--guide-start-seconds 12 --guide-seconds 4` uses the four encoded seconds
beginning at twelve seconds. Start and duration are independently rounded down
to complete 80-ms RVQ frames. Out-of-range selections are rejected; the cached
source tokens are never modified. The selected tokens are copied to the run's
input directory, hashed and recorded with the requested and actual selection.
Omitting `--guide-seconds` retains all encoded frames after the selected start.
Selection changes can reuse the same encoding without rerunning the encoder.

Encoder analysis context is separate from the guide fed to the generator. The
successful case encodes 29.76 seconds but feeds only eight. Shorter, silence-padded
encoder inputs produced early EOS in previous probes, so selecting a shorter
guide from an existing encoding is different from encoding only that short
waveform. The reference helper now accepts longer source audio through the
official chunked encoder; source speed is its default, with speed-altered
research controls requested explicitly through `--factors`.

When codec models are supplied, the driver writes both `audio.wav` (guide plus
new audio) and `continuation.wav` (new audio only), without changing samples.
The latter is the clip to place after the guide in the DAW. Encoding remains
the Spark Docker reference bridge; native Q4 continuation runs on the laptop.

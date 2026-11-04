# Continuation from audio ending at the cursor

The cursor is the point in the DAW where new audio should begin. A cursor-ending
guide is the tail of audio available before that point. If the cursor is at
29.76 seconds, an eight-second guide is seconds 21.76--29.76, not seconds 0--8.
The encoder may analyze earlier available audio but must not receive audio
after the cursor. A score-free run also receives no future symbolic notes.

The earlier score-guided transformation remains a useful separate mode: a
provided arrangement may intentionally describe music beyond the cursor.
This experiment evaluates the score-free case using only past input.

## Frozen source and boundaries

The official FP32 encoder previously received only the first 29.76 seconds of
the selected source. That private excerpt is 1,312,416 stereo samples at
44,100 Hz, with SHA256
`ff700ca0aa7a9bca414e4a6bb0877a5f4eb47d0912232d563be9bd6e50311a0c`.
The cached encoding has 372 complete RVQ frames, SHA256
`bdbeab789d35496a3744f41410ab4ffeaecf91e7a0a9759dd1a9e7ae075ab342`.
Its manifest pins the official encoder/checkpoint and source excerpt. No audio
from the remainder of the original song was passed to that encoder invocation.
This run reuses that encoding; it does not launch Python encoding or inference.

All guides end at frame 372. The native CLI selects their codes directly from
the cached file, with each complete frame representing exactly 80 ms:

| Guide | Source interval | First frame | Guide frames | Requested new frames |
| --- | --- | ---: | ---: | ---: |
| 4 seconds | 25.76--29.76 s | 322 | 50 | 400 |
| 8 seconds | 21.76--29.76 s | 272 | 100 | 400 |
| 16 seconds | 13.76--29.76 s | 172 | 200 | 400 |

The 400 new frames request 32 seconds beyond the cursor, to source timeline
61.76 seconds. EOS may end a run earlier. The generator's local timeline starts
at the selected guide; only its guide-free output is placed at the DAW cursor.
It does not receive the original waveform beyond the cursor, future score,
or a transcription of the remaining original song.

All three native runs pass exact replay, guide-token selection and bit-exact
suffix export checks. Results for this one source/seed are:

| Guide | Actual new audio | EOS | Onset-periodicity candidates near the source tempo |
| --- | ---: | --- | --- |
| 4 seconds | 32.00 s | No | 119.21 BPM; 59.72 half-time alias |
| 8 seconds | 32.00 s | No | 116.94 BPM; 58.46 half-time alias |
| 16 seconds | 31.84 s | Yes | 118.22 BPM; stronger 59.37 half-time candidate |

The original eight-second source tail gives 120.10 BPM. These are estimates
from generated audio only, not a clock-lock or melody-preservation test. More
guide context is not consistently closer to the source tempo in these cases.
The user hears a seamless handoff in the eight-second guide-plus-new preview,
with a wilder but interesting continuation. Four- and sixteen-second variants
have not yet received a listening verdict.

Generation including two replay passes and contract checks takes respectively
167.874, 189.832 and 208.732 seconds; staged decoding takes 25.5235, 32.7378 and
27.7064 seconds. These validation runs are not single-pass DAW latency benchmarks.

## Native command and fixed controls

Use `mulacover-generate` directly with the existing Q4_K_M generator, F32 staged
codec and original cached text/style preset. Its four-file input directory
contains only prompt tokens/mask, Qwen and MuQ buffers. There are no symbolic
files. `--score-mode none` supplies the upstream -1 dropped-score convention.
The 540 symbolic positions are a fixed null-condition shape retained from the
prior comparison; no notes or chords are loaded. The preset is instrumental
acoustic pop and contains no song-specific future lyrics.

Common settings are seed 1701, CFG 1.5, temperature 1, top-k 250, Torch CUDA
Philox cap 288, zero sampler offset, cache capacity 1024, ten decoder flow
steps and decoder guidance 1.25. Teacher-forced guide frames consume no sampling
noise. Each run performs exact replay and terminal/reset contract checks.

For the eight-second case, substitute these options into the native command
in `NATIVE_CONTINUATION.md` and use a fresh output directory:

```text
--score-mode none --capacity 1024
--audio-prefix refdata/bpm-prefix/encoded-long/guide-1.i32
--guide-start-seconds 21.76 --guide-seconds 8 --continuation-seconds 32
```

This validates a boundary and continuation behavior using already prepared
inputs. It does not implement a raw-PCM native encoder or custom-style frontend.
The entire encoding context is 29.76 seconds even when the selected guide is
shorter; guide duration and encoder analysis context remain separate.

## Listening and interpretation

Listening previews must say whether they include the reconstructed guide.
`audio.wav` includes it; `continuation.wav` starts at the cursor and contains
only generated audio. Do not present a continuation-only clip as though its
opening were a source reconstruction. Both must remain available for review.

The flow decoder processes noncausal 29.76-second windows. Newly generated
codes can affect the waveform reconstructed for the guide despite unchanged
guide tokens. A reconstructed guide is not an exact copy of host PCM. No raw
source samples are spliced into experimental outputs; final DAW source placement
and any join/crossfade remain explicit host work.

An input boundary that excludes future audio does not prove melody preservation,
instrumental behavior, a clean audio join, streaming inference, arbitrary-source
quality or a hard BPM lock. This is one source and seed with an experimental
score-free condition and Q4 generator. Existing expanded activation failures
and strict upstream decoder waveform failures remain open.

Private runs stay in `refdata/cursor-continuation/`. Keep only compact manifests,
boundaries, hashes and comparison results in `docs/validation/`.

## Longer guides and the context budget

Guide duration is not capped at the codec's 29.76-second processing window.
The pinned official encoder resamples available audio, splits it into chunks,
processes those chunks in batches, and concatenates their RVQ codes. The native
generator can teacher-force the resulting longer guide before sampling.

The native limit is:

```text
prompt_frames + guide_frames + new_frames - 1 <= cache_capacity <= 8192
```

With this 25-frame prompt, a 1024-position cache holds at most 1000 audio frames
(80 seconds total guide plus requested continuation). Larger caches extend that
budget within the 8192-position API limit, but their memory and latency require
measurement. This is finite context, not unlimited history. More guide frames
also add processing time before the first newly sampled frame.

After hearing the eight-second tail test, the user reports a seamless transition
with a wilder but interesting continuation and asks about a 50-second context.
The follow-up probe encodes only source seconds 0--50, primes all 625 complete
frames and requests 200 new frames (16 seconds), for 66 seconds total. It uses
two encoder chunks and a 1024-position generator cache. It has a different
cursor and requested length from the 4/8/16-second tail study, so it is a
feasibility/listening probe, not a controlled guide-length comparison.

Its one remaining reference encoder step runs in the dedicated Spark Docker
workflow, using the existing pinned FP32 CPU encoder; it installs nothing:

```sh
MULACOVER_ENCODER_AUDIO=refdata/cursor-continuation/source-first-50.wav \
MULACOVER_ENCODER_OUTPUT=refdata/cursor-continuation/encoded50 \
bash reference/run-spark.sh encode-prefix
```

The reference script uses read-only pinned source and decoder mounts and no
network. Generation, guide selection, sampling, decoding and suffix export then
run through the native laptop CLI. No new Python runtime stage is introduced.

The 50-second probe accepts all 625 guide frames (5,000 codes) exactly and
generates 85 new frames, or 6.80 seconds, before EOS. The request was 16 seconds;
early EOS is respected. Exact replay, terminal/reset checks and bit-exact suffix
export pass. Generation with two passes/checks takes 240.612 seconds and staged
decoding 35.8384 seconds. The context budget is sufficient: 849 positions were
reserved for the requested run in a 1024-position cache. EOS is not a cache limit.

Its preview starts 42 seconds into the full decoded output, showing the final
eight guide seconds and all 6.8 new seconds. All 50 guide seconds were supplied
to the model. The raw 56.8-second full decode and 6.8-second new-only output are
retained. Listening copies apply only constant -3 dB gain. The user hears the
model produce an ending and finds that result interesting. This is a listening
assessment of this short completion, not acceptance of guaranteed length or
general musical fidelity. The short generated section gives an onset candidate
near 117.64 BPM, with strong subdivision aliases and limited measurement length.

The first 372 frames of the 50-second source encoding differ in five code
positions from the earlier encoding ending at 29.76 seconds. Their input
contexts differ. Preserve each frozen encoder input independently when porting
resampling/chunk preparation rather than assuming prefix-invariant RVQ codes;
the cause of those differences is not isolated here.

The small reports are `docs/validation/cursor-continuation.json` and
`docs/validation/long-guide-continuation.json`. Larger context succeeds as an
input capability in this case, while continuation duration and musical quality
remain separate model behavior to evaluate.

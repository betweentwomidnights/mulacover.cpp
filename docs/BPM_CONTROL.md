# BPM control investigation

November 2, 2026. **The current MuLaCover path does not provide hard BPM control.**
MIDI tempo metadata is not passed to the generator, and explicit BPM text tags
did not achieve their requested tempos in the two controlled Q4 excerpts below.

## What reaches the model

The pinned upstream `SymbolicCondition` describes BPM as source metadata, not a
separate generation condition. MIDI notes are converted from ticks to a shared
sixteenth-note grid. `to_tensors()` supplies melody onset/sustain, chord chroma,
drums and a context mask, without BPM or seconds-per-beat. In the reference-audio
path, `bpm` affects transcription quantization; it is not a rendering clock.
Passing a separate `bpm` with MIDI is explicitly rejected upstream.

The runtime ablation retains all 137 instrumental melody notes and 160 voiced
chord notes, changing only MIDI tempo events to 80 or 160 BPM. Actual upstream
preprocessing freshly regenerates all inputs. **All 11 tensors are byte-identical
to the original conditioning in both cases.** A deterministic generator with the
same RNG therefore cannot distinguish those MIDI-tempo requests. Re-rendering
identical tensors twice is not necessary to establish this invariance.

Relevant pinned source:

- [Symbolic condition / BPM semantics](https://github.com/HeartMuLa/MuLaCover/blob/e27fa7fa0da7c9817879cc6210cc5693f9327946/src/mulacover/symbolic.py)
- [Pipeline MIDI/ref-audio handling](https://github.com/HeartMuLa/MuLaCover/blob/e27fa7fa0da7c9817879cc6210cc5693f9327946/src/mulacover/pipeline.py)

## Audio measurements

| Audio | Requested BPM text | Onset-period candidate, quarter-note interpretation | SheetSage2 score-clock estimate |
| --- | ---: | ---: | ---: |
| Original 69.985s source | none | 120.04 | 119.18 |
| Official upstream F32, 68.88s | none | 130.82 | not re-transcribed |
| Matched native F32, 68.88s | none | 130.82 | 131.06 |
| Q4 excerpt, 29.76s | 80 | 128.39 | 127.54 |
| Q4 excerpt, 29.76s | 160 | 126.82 | 121.85 |

These are estimates, not human-annotated ground truth. Onset analysis uses
positive log-spectral flux and normalized lag autocorrelation. Its strongest
global peaks for source/F32 are half-time aliases near 60 / 65.45 BPM; the table
shows their double-time candidates consistent with the score's quarter-note
clock. The raw candidate lists, correlations and windowed estimates are retained.
No estimator is forced to select a candidate close to the requested BPM. The
160-tag estimates disagree and warrant caution; neither demonstrates 160 BPM.

The default upstream and matched native F32 cover is about 9% faster than the
source by the onset estimates. This is present upstream as well as natively.
HeartCodec still returns exactly 3,840 samples per code frame at 48kHz (80ms);
this observation is not an accidental native sample-rate change.

## Text-tag experiment and limits

Both Q4 requests append `80 bpm` or `160 bpm` to the same acoustic instrumental
tags, with `[Instrumental]` lyrics. Fresh prompt/style tensors change; melody,
chords and zero drum tensors remain identical. Both use seed 1701, matched Torch
CUDA Philox with target grid cap 288, CFG 1.5, temperature 1, top-k 250, cache 512,
372 requested frames, and the same FP32 codec settings. Both reach 372 frames
without EOS and pass exact replay and terminal/reset checks.

This is one source and one seed, using Q4_K_M. It is not an unquantized, multi-seed
benchmark of every possible tag format. The measured samples do not establish
reliable text-tag BPM control. Even if another prompt/seed steers tempo better,
the architecture still supplies no explicit clock lock through MIDI metadata.
For downstream design, do not expose BPM as a guaranteed native generation
constraint. Strict timing would need a separate alignment/time-warp strategy or
a change in conditioning/model behavior; neither is implemented by this study.

## Reproduction and artifacts

`reference/probe-bpm.py` runs inside the dedicated Spark reference Docker image
with the usual `/workspace` and read-only `/refs` mounts:

```bash
/opt/sa3-venv/bin/python reference/probe-bpm.py --upstream /refs/MuLaCover \
  --prepared refdata/covers/billiegasm-full-acoustic/prepared --output refdata/bpm-study
```

Use a fresh output directory. Copy its small prepared directories to the laptop
and run `tools/generate.py` on `tags-80/prepared` and `tags-160/prepared` using
`--frames 372 --capacity 512 --rng torch-cuda --rng-cuda-blocks 288` and the
existing Q4/codec models. Tempo analysis uses only NumPy; optional SheetSage2
score-clock data come from the unchanged native Yuey binary. Calibration tests
check known 80/120/160-BPM clicks and silence; all 12 Python tests pass.

Compact evidence is `validation/bpm-adherence.json`. Raw audio and fixtures stay
outside Git. Local listening copies are `refdata/bpm-study/listening/request-80.wav`
and `request-160.wav`, both using the same constant gain, with raw WAVs preserved.
This investigation changes no model code, GGML pin or existing parity gates.

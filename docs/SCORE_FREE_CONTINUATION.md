# Score-guided and score-free continuation

The user identifies two useful workflows: the existing score-guided continuation
adheres to the song and transforms it, while an audio-only continuation could
extend material in the DAW without needing future notes. Neither replaces the
other. Listener assessment of the first does not establish the second.

## Controlled native comparison

Use the accepted Q4 continuation's original prepared text/Qwen style, seed 1701,
CFG 1.5, temperature 1, top-k 250, Torch CUDA Philox cap 288, eight-second
100-frame guide and 62-second 775-frame request. Keep 540 symbolic positions and
automatic 1024-position cache capacity. Change only symbolic conditioning:

- Control: original whole-song instrumental melody and chords, zero drums,
  original all-unpadded mask.
- Score-free: -1 throughout melody, chords and drums, all-unpadded mask.
  This matches pinned upstream `modeling.py::_mask_condition` for dropped
  conditioning. The conditional branch retains its original text and style;
  CFG's unconditional branch and the model graph remain unchanged.

`--score-mode none` implements this directly in `mulacover-generate`. It does
not read symbolic files. The experiment's input directory contains only
`tokens.i32`, `mask.u8`, `qwen.f32` and `muq.f32`, copied byte-for-byte from the
control. No Python process launches for generation, guide selection or decoding.
No core C ABI, GGML code or consumer pins change.

Append this option to the native command in `NATIVE_CONTINUATION.md` and use a
fresh output directory:

```text
--score-mode none
```

The default is `--score-mode provided`. `run.json` records the mode. The separate
`continuation.wav` contains only generated audio; `audio.wav` includes the guide.

## Interpretation

The upstream public frontend requires reference audio for transcription or
melody/chord MIDI. It does not expose an audio-only mode. Using its existing
dropped-score representation while retaining text/style is an inference
experiment, not evidence of a documented audio-only training mode. Silence/zero
notes and bypassing score cross-attention would be different interventions.

This first A/B intentionally retains the old guide to isolate score removal.
That guide consists of the first eight seconds of a 29.76-second encoding;
encoder features may contain later context. A strict DAW test must instead use
a tail guide ending at the end of available encoder input. The no-score mode
does remove all future symbolic content, but this A/B alone is not that strict
past-only baseline. Quantization, hard BPM lock, and upstream waveform parity
remain separate open questions.

## First result

The native Q4 run completes all 875 frames: 100 guide frames and 775 generated
frames (62 new seconds), without EOS. The 800 guide codes remain exact; 6,122
of 6,200 new codes differ from the score-guided control, starting at the first
sampled frame. Exact replay and terminal/reset contracts pass. Native generation
with checks takes 335.104 seconds and staged decoding 43.1978 seconds.

Onset periodicity's strongest quarter-note candidate is 118.09 BPM without the
score, versus 119.65 with the score and roughly 120 in the source. Half/double-time
aliases remain visible in the report. These estimates do not establish a hard
clock lock: 15-second windows at starts 0 and 10 seconds give about 119.5 BPM,
falling to about 118.0 and 117.7 at starts 30 and 40 seconds. Melody preservation
and musical usefulness require listening. Listener assessment of
the score-free version is pending.

The original float WAV is preserved. Its full render peak is 1.07852; a separate
`continuation-listen.wav` applies only constant -1 dB gain via native FFmpeg,
without stretching, trimming, normalization or sample-rate conversion. The
62-second raw suffix is a bit-exact sample copy of the full decoded waveform
after the guide. Both Windows CUDA and CPU shared/DLL builds pass eight native
tests; invalid score modes are rejected before model loading.

Private ignored audio/fixtures stay under `refdata/native-continuation/`.
The compact report is `docs/validation/score-free-continuation.json`. No new
Python files are introduced; the existing NumPy tempo analysis is an offline
measurement, separate from the native inference command.

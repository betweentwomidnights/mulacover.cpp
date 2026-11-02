# Full-song instrumental request

The user heard instruments only in the first acoustic Q4 eight-second excerpt
and requested a whole-song test before judging instrumental control.

On November 2, 2026, this run completed all 875 frames without early EOS and
produced 70 seconds of finite 48kHz float32 stereo. Exact replay and terminal/reset
checks passed. Generation with checks took 384.94 seconds and three-window codec
decoding took 34.46 seconds. Sampled GPU usage peaked at 6,776 MiB including the
desktop; generation used a 470,351,872-byte cache. Full-song listening is pending.

The raw WAV has peak 1.01176 and RMS 0.14925, with one individual channel sample
above full scale. A separate `generation/audio-listening.wav` applies one constant
gain of 0.88088865 (-1.10158 dB), giving a -1 dBFS peak for playback. There is no
limiter or edit, and the raw generated WAV is retained unchanged. This listening
copy was made after the validation run; the command below produces the raw WAV.

Reproduce from the laptop with the existing builds/models and Spark reference
image available:

```powershell
./scripts/test-cover-windows.ps1 -CaseName billiegasm-full-acoustic -FullSong `
  -Tags 'warm acoustic pop, soft drums, instrumental, no vocals'
```

Use a fresh case name for another run. `-FullSong` preserves the complete native
instrumental/chord MIDI exports and the full source reference PCM. It derives
the requested duration from the source, rounded up to 80ms generator frames,
and automatically sizes the cache unless `-CacheCapacity` is supplied. Do not
combine it with `-Frames` or `-Sixteenths`. EOS can still end generation early.

For this source, 69.9851 seconds requests 875 frames (70 seconds), with a
25-position prompt and 1,024-position cache. Full transcription has 209 events,
137 instrumental melody notes, 44 vocal notes and 160 voiced chord notes. The
chosen instrumental/chord MIDI exports are retained byte-for-byte. Upstream
symbolic preparation produces 540 sixteenth-note positions, with zero drum
conditioning. Prompt/style tensors match the acoustic excerpt; the full symbolic
context differs, so the beginning need not reproduce that excerpt's token stream.

All Spark model execution remains inside the dedicated reference Docker image.
Only request text and MIDI travel to Spark. Q4 generation/replay and staged FP32
HeartCodec decoding run on the laptop using isolated GGML `0f2ed045`. Shared
consumer pins remain at `b77ae79`; no coordinated bump/check is part of this test.

Outputs are ignored under `refdata/covers/billiegasm-full-acoustic/`: full
transcription and reference audio in `transcription/`, prepared inputs in
`prepared/`, and generated `audio.wav`, `codes.i32` and `provenance.json` in
`generation/`. Compact results are recorded in `validation/full-song-test.json`.

Listening must assess singing/vocalizations throughout the track, fidelity to
the source music, continuity across codec windows and overall audio quality.
Successful execution does not settle those questions. Generated tempo is not
locked to the source or MIDI rendering tempo. This run does not resolve the
previously recorded FP32 longer-context prefill activation gap or establish Q4
quality acceptance.

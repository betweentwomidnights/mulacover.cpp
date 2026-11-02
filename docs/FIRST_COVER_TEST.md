# First real-audio transcription-to-cover excerpt

November 2, 2026. Source: user-selected
`C:/Users/thegr/Music/billiegasm_yuey.wav`, unchanged SHA-256
`140f18292989b1026113f02e0b6e2c9c1eaea29df05965f9404e40d76f74af9a`.

The first short real-input pipeline now executes. This is an excerpt integration
test; instrumental control and musical fidelity are not yet listener-approved.
The source audio, transcriptions, tensors and WAV files remain outside Git.

## Stages and provenance

1. Existing native `yue2-transcribe 0.3.0` processes the full 69.985-second file
   on the laptop with `sheetsage2-mert2-f16.gguf`, full score mode and pickup kept.
   Its linked GGML reports `b77ae790`; no sibling files or pins were changed.
2. `tools/transcribe-cover.py` checks the input hash, audits every exported melody
   note against decoded events, and selects the explicit instrumental role. The
   MIDI audit checks beat onset, pitch and duration, not ground-truth musical
   correctness. Output is 209 events and 181 melody notes: 137 instrumental and
   44 vocal, with no transcription warnings.
3. `tools/midi_bridge.py` crops the first 64 sixteenth-note positions without
   an ABC roundtrip. PPQ/conductor information is retained and crossing notes are
   closed. The excerpt has 23 instrumental melody notes and 19 voiced chord notes.
   Source timestamps span 0..8 seconds. Inferred MIDI tempo is 118 BPM, giving
   8.1356 seconds of MIDI rendering. MuLaCover consumes beat-space conditioning;
   its output tempo is not fixed by this MIDI metadata.
4. `reference/prepare-conditioning.py` runs the pinned upstream prompt tokenizer,
   MIDI-to-tensor conversion and Qwen encoder in Spark Docker. Every request gets
   fresh prompt/style tensors. Re-exporting/reloading the exact symbolic MIDI
   reproduces all conditioning tensors. Repeating preparation from the committed
   Alex example reproduces all prepared tensors bit-for-bit. Qwen is released
   after preparation. The original audio never needs to be transferred to Spark.
5. Native MuLaCover sampling and staged FP32 HeartCodec decoding use the isolated
   GGML fix `0f2ed045`. Transcription, generator and decoder weights are released
   between stages. The laptop Q4 runs use cache capacity 256 and peak at a sampled
   **6,776 MiB** through generation/decode; native transcription peaks at 3,134 MiB,
   both including desktop usage. These are sampled validation peaks, not guarantees.

The fully native prompt/tokenizer/Qwen/symbolic frontend is still future work.
Separate processes currently keep Yuey's unchanged shared baseline distinct from
MuLaCover's development dependency. ggml-bump-check and all coordinated sibling
bumps remain deferred. The earlier longer-context activation gap is unchanged.

## Requests tested

All requests use `[Instrumental]` in the required nonempty lyric field. Native
generation uses temperature 1, top-k 250, CFG 1.5, seed 1701 and 100 frames. The
decoder uses seed 1701, 10 steps, guidance 1.25 and 372-code windows. Each output is
eight seconds of 48kHz float32 stereo, with no gain normalization.

The free-form request uses `warm acoustic pop, soft drums, instrumental, no vocals`.
It is tested first with the old synthetic MIDI (F32/Q4 on Spark), then with the
actual transcribed instrumental/chord excerpt (F32 on Spark, Q4 on the laptop).
The real test has no drum MIDI and its drum tensor is entirely zero; drums are
optional conditioning, and the style request can still influence orchestration.

The second real-input request reproduces
[Alex's reported setup](https://github.com/HeartMuLa/MuLaCover/issues/1#issuecomment-6251041772):

```text
topic:[Instrumental]; genre:[Classical]; instrument:[Violin, Piano]; mood:[dramatic]
```

This is his reported recipe, not a guarantee of instrumental output. The request
gets its own style embedding and is tested with F32 on Spark and Q4 on the laptop.
Its raw tag string is lowercased by the actual upstream preparation, as are lyrics.
Requests are committed in `examples/`; their fresh embeddings differ from the
previous demo's explicitly vocal style. No cached "clear voice" embedding is used.

All six runs complete 100 generated frames with exact seeded replay, terminal
token/reset checks, valid RVQ codes and finite stereo output. This demonstrates
execution, not absence of singing or accuracy of the cover. F32/Q4 free-running
samples can diverge. Cross-platform numeric seeds do not imply identical decoder
noise; the synthetic F32/Q4 Spark pair is the same-platform comparison.

## Listening files (local, ignored)

| Case | Audio path | Peak | RMS |
| --- | --- | ---: | ---: |
| Original source excerpt | `refdata/billiegasm-cover/transcription/source-excerpt.wav` | original PCM | original PCM |
| Real, acoustic, Q4 laptop | `refdata/billiegasm-cover/q4-laptop/audio.wav` | 0.6110 | 0.05881 |
| Real, acoustic, F32 Spark | `refdata/billiegasm-cover/f32-spark/audio.wav` | 0.3655 | 0.04693 |
| Real, Alex recipe, Q4 laptop | `refdata/billiegasm-cover/alex-q4-laptop/audio.wav` | 0.2010 | 0.03631 |
| Real, Alex recipe, F32 Spark | `refdata/billiegasm-cover/alex-f32-spark/audio.wav` | 0.6900 | 0.09591 |
| Synthetic instrumental control, F32 | `refdata/instrumental-control/f32-spark/audio.wav` | 0.3611 | 0.09815 |
| Synthetic instrumental control, Q4 | `refdata/instrumental-control/q4-spark/audio.wav` | 0.9576 | 0.11165 |

Every run retains its `provenance.json`, token file and model/input/audio hashes.
Compact checked-in evidence is `validation/first-cover-test.json`. Listening must
separately assess vocalizations, recognizability of the source melody/chords and
musical/audio quality. No automatic vocal classifier was used. Full-length cover
generation remains untested; the complete 70-second source transcription is saved.

## Reproduction

Install the pinned local development requirements (now including Mido). Run the
complete bridge from the laptop, with the existing models/builds and Spark
reference image available:

```powershell
./scripts/test-cover-windows.ps1 -CaseName my-first-cover
```

This defaults to the selected source, instrumental melody role, 64 sixteenth-note
positions, Alex's structured tags and 100 generated frames. It performs the native
transcription/audit locally, sends only request text and cropped MIDI to Spark,
prepares conditioning in Docker, then returns to the laptop for Q4 generation and
decoding. The original audio stays on the laptop. Use a fresh `CaseName` each time;
existing outputs are preserved. `-Lyrics`, `-Tags`, `-Frames`, `-CacheCapacity`,
`-MelodyRole` and model/CLI paths can be overridden. The cache must accommodate
the prompt and generated frames. Outputs live under `refdata/covers/<CaseName>`.

The integration run `billiegasm-reproduce` regenerated all 11 prepared tensors,
the 100-frame token stream and final WAV bit-for-bit against the earlier Alex
recipe laptop run. Its sampled device peak was again 6,776 MiB. This establishes
repeatability on this setup, not cross-platform RNG equivalence or audio quality.

For individual stages, run the native transcription wrapper with the
source/model/CLI paths recorded in its provenance; `--melody-role instrumental
--sixteenths 64` selects this excerpt.

On Spark, after transferring only the excerpt MIDI and committed example, run
the following **inside the dedicated reference Docker image** with the established
`/workspace` and read-only `/refs` mounts:

```bash
/opt/sa3-venv/bin/python reference/prepare-conditioning.py --upstream /refs/MuLaCover \
  --request examples/billiegasm-alex-instrumental.json --output refdata/conditioning/my-alex-request
```

Copy that small prepared directory back to the laptop. Use `tools/generate.py`
with it as `--reference`, the development CUDA CLI, both HeartCodec models and a
fresh output directory, as in `SAMPLING_AND_AUDIO.md`. No source or submodule bump
is necessary. Tests cover tempo-change preservation, same-pitch overlap/clipping,
note closure, role/event mismatches and the existing codec/metric guards: 10 Python
tests pass. The selected original file's hash was checked again after the run.

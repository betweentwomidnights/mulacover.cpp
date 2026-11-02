# Full-song melody-source and precision controls

The user reported that the first 70-second acoustic Q4 output vocalized the guitar
melody and did not audibly represent the singer's melody. That run explicitly
selected the instrumental transcription lane. These controls separate melody
source selection from generator precision.

| Run | Melody lane | Generator | Hardware | Comparison |
| --- | --- | --- | --- | --- |
| Original | Instrumental, 137 notes | Q4_K_M | Laptop | User-reported guitar vocalization |
| Vocal source | Vocal, 44 notes | Q4_K_M | Laptop | Change melody source only |
| F32 control | Instrumental, 137 notes | F32 | Spark | Compare with Q4 on Spark |
| Q4 platform control | Instrumental, 137 notes | Q4_K_M | Spark | Keep hardware fixed for precision comparison |

All requests retain `[Instrumental]`, tags `warm acoustic pop, soft drums,
instrumental, no vocals`, CFG 1.5, temperature 1, top-k 250 and native seed 1701.
Each requests 875 frames, with cache capacity 1,024. HeartCodec stays FP32, using
seed 1701, 10 steps, guidance 1.25 and 372-code windows. Generator weights are
released before codec loading; the raw WAVs are preserved.

The vocal-source Q4 run sampled EOS after 791 accepted frames, producing 63.28
seconds. It passed exact replay and terminal/reset checks; no tokens or audio were
added to fill the requested 70-second limit. Its sampled GPU peak was 6,865 MiB
including desktop usage. Generation with checks took 267.10 seconds and codec
decoding took 33.46 seconds. Its individual listening copy applies a constant
-2.27592 dB gain, with the raw 1.15823-peak WAV retained.

The user did not recognize the original singer's melody in that generated Q4
clip. A direct fixed-tone render of the 44-note vocal MIDI was then recognized
as correct by the user. This supports this source's vocal transcription and
leaves preservation of that melody through conditioning/generation unresolved.
The user has not yet reported vocal presence for the vocal-source Q4 clip.

For the lane definitions, the [official SheetSage2 model card](https://huggingface.co/m-a-p/SheetSage2)
and Yuey's native API expose two melodic lanes (vocal and instrumental), combined
melody, and symbolic chords. Combined melody is the union of the two lanes, not
a third independently transcribed melody or a complete per-instrument score.
`tools/preview-midi.py` renders each component directly as a fixed harmonic tone
with equal note velocities and its MIDI tempo map. It adds no MuLaCover model
inference and does not reproduce the original instrumentation or wall-clock timing.

Additional artifact checks verify that all 44 vocal-note onsets and sustain spans
are present in the prepared pianoroll, with exactly 44 onset cells. The original
instrumental preparation similarly contains all 137 note onsets and sustain spans.
This establishes the selected notes at the prepared-input boundary; it does not
establish that the generator follows them faithfully.

The Spark F32 control completed all 875 frames (70 seconds), with no early EOS;
generation/replay took 540.43 seconds and decoding took 29.54 seconds. The matching
Q4 Spark control sampled EOS after 831 frames (66.48 seconds); generation/replay
took 197.98 seconds and decoding took 24.48 seconds. Both pass exact replay and
terminal/reset checks. These are individual validation timings, not controlled
performance benchmarks. No output is padded to the requested limit.

The four comparison WAVs in `refdata/covers/billiegasm-controls-listening/` all use
the same constant gain, 0.76949173 (-2.27592 dB). Their relative output levels are
preserved and their maximum peak is -1 dBFS. Raw outputs and individual listening
copies remain unchanged. The Spark F32/Q4 pair uses identical prepared-manifest
hashes, model-independent settings, codec weights and executable.

The F32 run tests instrumental-request behavior on the original instrumental-lane
case. It is not an F32 vocal-lane control; preservation of the listener-confirmed
vocal melody through F32 generation would be a separate comparison. Do not assign
the Q4 vocal-lane melody failure to transcription or quantization from these
observations alone.

The user subsequently confirmed vocalizations following the guitar melody in the
native FP32 instrumental-lane control as well. This demonstrates that the reported
behavior is not exclusive to Q4 quantization. Instrumental control is not accepted
for that F32 run either. Q4 Spark listening remains pending. No matching official
upstream full-song generation has been auditioned, so this finding does not yet
distinguish the underlying model/prompt behavior from native implementation
differences. The separate vocal-lane melody-preservation question remains open.

All 11 prepared tensors are hash-checked. Between the original and vocal-source
runs, only `input.pianoroll` differs; the other ten tensors are byte-identical.
Both symbolic contexts have 540 sixteenth-note positions, including identical
chords and zero drum conditioning. Native melody/event audits pass, but do not
establish musical transcription accuracy. The complete original vocal MIDI is
retained; it contains only the 44 notes transcribed into that lane.

The Spark F32/Q4 controls use the exact same prepared directory and executable.
Cross-platform numerical seeds do not guarantee identical codec noise or tokens;
use the same-platform Spark pair to compare precision. A free-running sample
comparison is not a tensor-parity test, and a single seed is not a quality survey.
These are native generator controls, not new official upstream generation runs.
The previously recorded longer-context F32 prefill activation gap remains open.

## Reproduction

From the laptop, with the existing models/builds and Spark reference image:

```powershell
./scripts/test-cover-windows.ps1 -CaseName billiegasm-full-vocal-acoustic -FullSong `
  -MelodyRole vocal -Tags 'warm acoustic pop, soft drums, instrumental, no vocals'
```

For each Spark precision control, invoke the helper on the Spark host. It runs all
Python/model execution in the dedicated Docker image. After the original full-song
conditioning has been prepared, run this from `/home/kev/dev/mulacover.cpp`:

```bash
MULACOVER_GENERATION_REFERENCE=refdata/covers/billiegasm-full-acoustic/prepared \
MULACOVER_GENERATOR_MIX=f32 MULACOVER_GENERATION_FRAMES=875 \
MULACOVER_GENERATION_CAPACITY=1024 \
MULACOVER_GENERATION_OUTPUT=refdata/covers/billiegasm-full-acoustic/generation-f32-spark \
bash reference/run-spark.sh generate-development
```

For the same-platform Q4 control, change `MULACOVER_GENERATOR_MIX` to `q4_k_m` and
the output suffix to `generation-q4-spark`. Use fresh output/case names for repeat
runs. The optional `MULACOVER_GENERATION_REFERENCE` override leaves the helper's
original synthetic reference default intact.

Compact results and listening-file hashes are recorded in
`validation/full-song-controls.json`. Full-song vocal presence, source-melody
recognizability and musical quality require listening evidence. Shared GGML pins
and production services are unchanged; coordinated bump/check work remains deferred.

# SheetSage2 integration seam

The existing sibling `C:/dev/yue2.cpp` at
`c36baa91c594d84b646e73013b1fa81a3f3e39f4` implements native MERT2/SheetSage2
transcription and exposes independent melody, vocal, instrumental and chord MIDI
exports in `yue2::TranscriptionMidiExports` (`include/yue2/transcription.h`).

Alex reports improved instrumental MuLaCover results using SheetSage2 melody/chord
MIDI in [upstream issue #1](https://github.com/HeartMuLa/MuLaCover/issues/1#issuecomment-6251041772).
This is his reported experience; it is not a benchmark reproduced in this port.

The intended reusable path is yue2's transcription result -> melody/chord MIDI ->
MuLaCover's existing symbolic-conditioning boundary. Retain score beat timing,
pickup handling and aligned melody/chord tracks; do not convert through ABC text
when structured MIDI/events already exist. SheetSage2's current exported score
does not provide a drum lane, so drum conditioning must remain optional or use a
separate source. Native generator parity initially uses frozen synthetic MIDI
fixtures, allowing transcription integration to be validated separately later.

Model libraries should share one GGML target and release transcription model
weights before loading the generator on an 8GB GPU. Full FP32 generator inference
stays in Spark Docker. Q4_K_M generation and staged HeartCodec decoding now run
on the laptop; see `SAMPLING_AND_AUDIO.md`. Native prompt/Qwen/MIDI preparation
remain outstanding as native primitives. A working reference bridge now joins
the native transcription exports to native generation; see `FIRST_COVER_TEST.md`.

## User-selected real-audio test

Use `C:/Users/thegr/Music/billiegasm_yuey.wav` when the transcription-to-cover path
is ready. The user selected this source on November 1, 2026. It is approximately
69.985 seconds, 44.1kHz stereo, signed 16-bit PCM. The file is referenced in
`validation/transcription-cover-input.json` by path and SHA-256; the audio remains
unchanged outside the repository. Do not substitute an unrelated source or infer
its musical content from its filename.

The first eight-second generator demo was explicitly vocal: lyrics were
`[Verse]` followed by "A lantern swings beside the rain / We count the lights
along the train", with tags `warm acoustic pop, soft drums, clear voice`. It is
not evidence for or against instrumental controllability.

For the real test, retain SheetSage2's aligned melody/chord MIDI and expose which
melody role is used. Keep drum conditioning optional. First validate the MIDI
timing/contents against the source, then generate a short excerpt before a full
length attempt. Release transcription weights before generator loading, and
generator weights before codec loading.

Instrumental control is an explicit pending acceptance test. A candidate request
uses nonempty `[Instrumental]` text and instrumental style tags without "clear
voice"; this is a test input, not a demonstrated recipe. The pinned upstream
preprocessor rejects empty lyrics. Rebuild both prompt tokens and Qwen/style
embeddings for the new request; reusing the demo's vocal style embedding would
confound the test. Use the same transcribed MIDI and controlled sampling settings
for F32/Q4 comparisons. Record audible singing/vocalizations separately from MIDI
fidelity, audio quality and numerical execution success. A successful WAV decode
does not establish that an instrumental request was respected.

On November 2, 2026, the first excerpt test completed. SheetSage2 transcribed the
full source into 209 events / 181 notes, with no warnings. All 181 melody MIDI
notes (44 vocal, 137 instrumental) match the decoded events in pitch, beat onset
and duration. The first 64 sixteenth-note positions of the instrumental lane and
chords supply the test: 23 melody notes and 19 voiced chord notes. Their source
timestamp span is 0..8 seconds; their MIDI rendering lasts 8.1356 seconds at its
inferred 118 BPM. MuLaCover conditioning uses beat positions, so this is not a
promise that generated tempo or waveform time will follow the source exactly.

Both the free-form instrumental request and Alex's exact structured-tag recipe
have fresh Qwen/style conditioning and F32/Q4 listening clips. Their numeric
execution/replay checks pass; instrumental controllability and musical fidelity
are still listener acceptance checks. The integration uses the existing Yuey
binary on `b77ae79` in a separate process, without changing that sibling or its
pin. The native MuLaCover process uses the authorized isolated CUDA-fix profile.

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
weights before loading the generator on an 8GB GPU. The present FP32 generator
parity runtime is Spark Docker; a laptop GPU deployment depends on later generator
quantization and cache measurements.

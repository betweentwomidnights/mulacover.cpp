# Project instructions

- Follow sibling `sa3.cpp` and `heartcodec.cpp` library, GGUF, parity and ownership patterns.
- Depend on `heartcodec::heartcodec`; share one `betweentwomidnights/ggml` instance.
- Spark Python, inference, conversion, dependency installs, builds and tests run only
  inside dedicated Docker images. Do not change production containers.
- Match unquantized reference tensors before quantizing the generator.
- Keep weights, caches and large fixtures out of Git; keep compact manifests and reports.
- Native FP32 teacher-forced frame/logit/cache parity passed 600 tensors on
  Spark CUDA at historical `07b77b0`; see `docs/NATIVE_PARITY.md`. The coordinated
  `b77ae79` baseline currently fails CUDA precision parity; see
  `docs/GGML_B77_ALIGNMENT.md`. Windows static/DLL smoke tests pass.
  Full F32 weights exceed laptop VRAM; F32 native/reference GPU execution stays in
  Spark Docker. Q4_K_M native execution now runs on the laptop. See
  `docs/QUANTIZATION.md` for experimental drift and memory results.
- Use the FP32 oracle for F32 math; BF16 reference fixtures are a separate suite.
  Preserve the strict logit/cache gates and the recorded flat-absolute activation
  diagnostics. Q4_K_M is an experimental measured policy, not quality-approved.
  Native top-k sampling and staged HeartCodec decoding now work from prepared
  conditioning through a separate sampling capability API. Native preprocessing
  is still absent. See `docs/SAMPLING_AND_AUDIO.md` for eight-second clips and
  long-context tests. The original 600-tensor suite passes; the 128-prompt suite
  passes public logits/KV but retains two prefill activation gate failures per
  case. Do not relax those gates or call the complete expanded suite passed.
- GGML is shared across repositories. Keep the agreed pin and working tree intact.
  Try the existing in-tree target/API before proposing infrastructure changes.
  An essential GGML change belongs in an isolated worktree/branch and a focused
  draft PR for other agents to review. Do not advance consumer pins as part of
  that proposal; adoption must be coordinated after review.
- The coordinated GGML pin is `b77ae790e628270af46a8af1eca38441c23f1252`,
  branch `feature/shared-sa3-acestep-v0.17` (historical name). Any necessary PR
  must be based on that baseline and target that branch. Do not patch shared
  checkouts or coordinate adoption of an unreviewed fix. Earlier `07b77b0` reports are historical.
  Do not automatically repair another repository's working checkout.
- User explicitly authorized continued MuLaCover development on CUDA fix
  `0f2ed045` before coordinated sibling bumps. Use the isolated source override
  and `dependencies/ggml-development.json`; leave HeartCodec's recorded GGML pin
  at `b77ae79`. Do not run ggml-bump-check or update sa3.cpp, yuey.cpp, acestep.cpp,
  stems.cpp or audiocraft.cpp until the user starts that process after MuLaCover works.
- User selected `C:/Users/thegr/Music/billiegasm_yuey.wav` for the first real
  transcription-to-cover test. See `docs/TRANSCRIPTION_INTEGRATION.md` and the
  input manifest; keep the audio outside Git and unchanged. Instrumental control
  must be tested explicitly with regenerated prompt/style conditioning. The
  initial eight-second demo used lyrics and a "clear voice" tag, so it was vocal.
- The first real excerpt bridge now works: native SheetSage2 on the laptop ->
  audited instrument/chord MIDI -> upstream conditioning/Qwen in Spark Docker ->
  native F32/Q4 generator and staged codec. See `docs/FIRST_COVER_TEST.md`.
  Do not call the frontend fully native or mark instrumental/musical quality
  accepted without listening evidence. The source audio is unchanged; the first
  cover uses only its first four bars/eight-second excerpt.
- User heard instruments only in the first acoustic Q4 excerpt and requested the
  whole song. The full 70-second Q4 acoustic run now executes on the laptop;
  see `docs/FULL_SONG_TEST.md`. All 875 frames, exact replay and terminal/reset
  checks pass; full-song vocals/fidelity remain listener checks. Preserve the raw
  WAV; the separate listening copy uses only a constant -1.10158 dB gain.

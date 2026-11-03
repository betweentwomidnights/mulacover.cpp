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
  checks pass. User reports the full-song output vocalizes the guitar melody and
  does not represent the vocalist's melody. Instrumental control is not accepted
  for this run. We selected 137 instrumental notes and excluded 44 vocal notes;
  melody source selection is separate from desired output instrumentation.
  Full-song controls now execute: Q4 vocal lane on laptop (63.28s, EOS), F32
  instrumental lane on Spark (70s), and Q4 instrumental lane on Spark (66.48s,
  EOS). See `docs/FULL_SONG_CONTROLS.md`. User confirms the direct vocal MIDI
  preview is correct but does not recognize that melody in the Q4 vocal-lane
  generated clip. All 44 vocal onsets/sustains reach prepared conditioning;
  preservation through generation remains unresolved. The F32 control uses the
  instrumental lane; no F32 vocal-lane run or combined-lane run has been made.
  User also hears guitar-melody vocalizations in the native F32 instrumental-lane
  control. This behavior is not exclusive to Q4. Q4 Spark listening remains
  pending. An official upstream full-song FP32 control now executes (861 frames,
  68.88s, EOS), with all 11 fresh prepared tensors matching the native control.
  See `docs/UPSTREAM_AUDIO_CONTROL.md`. User hears the guitar melody played by
  horns with no vocals upstream. The original native control used different RNG.
  A new opt-in Torch CUDA Philox capability in HeartCodec now matches all 31
  oracles (85,205,452 values) bit-for-bit on Spark and laptop with target cap 288.
  The matched-RNG native F32 full-song run reproduces all 6,888 upstream codes
  across 861 frames and EOS exactly. See `docs/PHILOX_PARITY.md`.
  User confirms this matched-Philox native F32 clip matches the upstream horns
  with no vocalizations. Listener acceptance applies to this F32 case only.
  Decoded audio cosine is 0.999995528, but strict waveform gates still FAIL
  (max_abs 0.0827555 vs 0.01; relative L2 0.00299061 vs 0.001). Do not relax them,
  infer an SQ-level exception without diagnostics, or claim waveform parity.
  Q4 Philox runs 70s on the laptop (6,865 MiB peak); Q4 listening is pending.
  Use `--rng torch-cuda --rng-cuda-blocks 288` for new Spark/laptop parity
  comparisons; the legacy default exists for historical replay.
  Do not blame transcription or quantization
  without the relevant evidence, or relax the recorded numerical gates.
  Preserve the raw
  WAV; the separate listening copy uses only a constant -1.10158 dB gain.
- BPM study: `docs/BPM_CONTROL.md` and `docs/validation/bpm-adherence.json`.
  Changing only MIDI tempo to 80/160 produces all 11 identical model tensors.
  Source audio estimates near 120 BPM; official/matched-native F32 near 131.
  Tested Q4 text tags 80/160 give roughly 128 / 122-127 BPM estimates, not reliable
  adherence. Do not promise clock-locked BPM or invent a model BPM input; the
  codec still maps each frame to exactly 80ms. Tag study is one source/seed.
- Repositories are public under betweentwomidnights: heartcodec.cpp and
  mulacover.cpp. Publish source/docs/compact reports only; retain model, audio,
  fixture and build exclusions. HeartCodec's GitHub-initialized MIT LICENSE
  commit is preserved in local history; MulaCover uses matching own-source MIT
  with separate upstream source/model notices. Shared-GGML adoption stays deferred.
- BPM intervention work is isolated on `experiment/score-clock`.
  `docs/SCORE_CLOCK_EXPERIMENT.md` records conditional Gaussian score-attention
  bias and optional local symbolic masks. The initial 1,404 public logit/KV
  oracle comparisons pass unchanged gates; attention did not reliably control
  tempo. Audio priming follows the source's ~120 BPM in full F32/Q4 runs (eight
  guided seconds + 62 new seconds) and one shorter Q4 seed-1702 replication.
  Slow-guide control misses its target; no arbitrary-BPM or hard-lock claim.
  Do not enable these controls by default. Audio-token priming is experimental:
  output includes teacher-forced guide frames, which must be excluded from
  continuation tempo/quality measurements. Teacher-forced frames draw no RNG;
  an explicit sampler offset can resume an existing Philox stream.

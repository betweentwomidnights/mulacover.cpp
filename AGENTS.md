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
  Full F32 weights exceed laptop VRAM; native/reference GPU execution stays in
  Spark Docker. Routine C++ editing/building happens on the laptop.
- Use the FP32 oracle for F32 math; BF16 reference fixtures are a separate suite.
  Preserve the strict logit/cache gates and the recorded flat-absolute activation
  diagnostics. Q4_K_M, free-running sampling and native preprocessing are not yet
  implemented. The C ABI currently accepts prepared conditioning and teacher codes.
- GGML is shared across repositories. Keep the agreed pin and working tree intact.
  Try the existing in-tree target/API before proposing infrastructure changes.
  An essential GGML change belongs in an isolated worktree/branch and a focused
  draft PR for other agents to review. Do not advance consumer pins as part of
  that proposal; adoption must be coordinated after review.
- The coordinated GGML pin is `b77ae790e628270af46a8af1eca38441c23f1252`,
  branch `feature/shared-sa3-acestep-v0.17` (historical name). Any necessary PR
  must be based on that baseline and target that branch. Do not patch shared
  checkouts or adopt an unreviewed fix. Earlier `07b77b0` reports are historical.
  Do not automatically repair another repository's working checkout.

# Project instructions

- Follow sibling `sa3.cpp` and `heartcodec.cpp` library, GGUF, parity and ownership patterns.
- Depend on `heartcodec::heartcodec`; share one `betweentwomidnights/ggml` instance.
- Spark Python, inference, conversion, dependency installs, builds and tests run only
  inside dedicated Docker images. Do not change production containers.
- Match unquantized reference tensors before quantizing the generator.
- Keep weights, caches and large fixtures out of Git; keep compact manifests and reports.
- Native FP32 teacher-forced frame/logit/cache parity now passes 600 tensors on
  Spark CUDA; see `docs/NATIVE_PARITY.md`. Windows static/DLL smoke tests pass.
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
- `docs/GGML_PIN_INVENTORY.md` records current consumer divergence. The validated
  native baseline is HeartCodec's clean `07b77b0`, not a claim of alignment with
  sa3.cpp/yue2.cpp. Do not automatically repair another repository's checkout.

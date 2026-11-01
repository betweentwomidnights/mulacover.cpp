# Project instructions

- Follow sibling `sa3.cpp` and `heartcodec.cpp` library, GGUF, parity and ownership patterns.
- Depend on `heartcodec::heartcodec`; share one `betweentwomidnights/ggml` instance.
- Spark Python, inference, conversion, dependency installs, builds and tests run only
  inside dedicated Docker images. Do not change production containers.
- Match unquantized reference tensors before quantizing the generator.
- Keep weights, caches and large fixtures out of Git; keep compact manifests and reports.
- The first milestone is frozen symbolic/prompt/style/logit/KV-cache reference data.

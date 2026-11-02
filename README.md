# mulacover.cpp

Native MuLaCover generator work for gary4local/gary4juce. This repository owns
symbolic MIDI conditioning, prompt/style preparation, the autoregressive model,
sampling and its reference fixtures. Sibling `heartcodec.cpp` owns RVQ/flow/SQ
decoding and will be consumed as `heartcodec::heartcodec`, sharing one copy of
`betweentwomidnights/ggml` with the generator and other model libraries.

**Native FP32 teacher-forced frame/logit/cache parity now passes on Spark CUDA.**
The generator implements prompt/audio embeddings, the 28-layer backbone, style
modulation, eight symbolic adaptors and gated cross-attention, the three-layer
local decoder, all eight output heads, CFG and persistent KV state. Both
conditional and CFG=1.5 cases match three frames and replay exactly after reset.
See [the parity report](docs/NATIVE_PARITY.md) and [C ABI](docs/C_ABI_V1.md).

Q4_K_M and free-running sampling are not implemented yet. Prepared prompt/style
and symbolic inputs are supplied by the caller; text/Qwen/MIDI preprocessing
currently comes from the upstream fixture harness. The laptop is the primary C++
workspace; full FP32 native/reference execution stays in Spark Docker containers.

GGML was not changed for this milestone. This baseline uses HeartCodec's existing
`07b77b0` source checkout, mounted read-only. Other inspected consumers record
`b77ae79`; see the [pin inventory](docs/GGML_PIN_INVENTORY.md). Selecting/adopting a
common pin requires coordination. CMake defaults to the in-tree target and reuses
an already-provided GGML target; it never builds a second GGML instance.

`reference/export_reference.py` executes the pinned upstream preprocessing and
generator on original synthetic melody/chord/drum MIDI and original short lyrics.
It freezes prompt tokens/masks/positions, Qwen style inputs and hidden states,
symbolic tensors, selected backbone/style/adaptor states, all eight raw and guided
logits, local-decoder states, and active KV prefixes across three teacher-forced
audio frames. Both conditional and CFG=1.5 cases reset caches and replay exactly;
unused cache tails are checked for zero writes. FP32 and upstream BF16 runs are
kept separately. BF16 fixture values are stored losslessly as float32.

References are pinned to MuLaCover `e27fa7fa0da7c9817879cc6210cc5693f9327946`,
generator weights `bbbaef2b31835c2ef5172ff528dbe325230d46fb`, and Qwen
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
See `docs/validation/` for compact verification reports. Weights and raw fixture
payloads are ignored in Git and available under local `refdata/` and the Spark
mount `/home/kev/dev/mulacover.cpp`.

Reproduce on Spark, using the reference image built by sibling HeartCodec:

```bash
docker run --rm --user "$(id -u):$(id -g)" --gpus all --network none \
  -e USER=heartcodec -e LOGNAME=heartcodec -e HOME=/workspace \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "$PWD:/workspace" -v /home/kev/dev/_heartcodec_refs:/refs:ro -w /workspace \
  heartcodec-reference:20261101 /opt/sa3-venv/bin/python reference/export_reference.py \
  --upstream /refs/MuLaCover --output refdata/upstream-f32 --dtype float32
```

Use `--dtype bfloat16 --output refdata/upstream-bf16` for the BF16 oracle. Run
`reference/prepare_models.py` in the same container with network access,
`HF_HUB_OFFLINE=0`, `HF_HUB_DISABLE_XET=1`, and a persistent `HF_HOME` mount to
retrieve the pinned weights. The immutable reference image ID and build recipe
are recorded in HeartCodec's `docs/SPARK_WORKFLOW.md` and `docker/`.

Native reproduction on Spark:

```bash
bash reference/run-spark.sh convert
bash reference/run-spark.sh build
bash reference/run-spark.sh parity
```

The Windows static/DLL smoke tests build with `scripts/build-windows.ps1` and
`scripts/build-windows.ps1 -Shared`; they use the unchanged in-tree CPU backend.
Local Python checks require NumPy only, using
`python -m unittest discover -s tests -v`. Full FP32 weights remain on Spark.

Next: coordinate the cross-repository GGML baseline, then add a controlled
Q4_K_M policy and measure logit/cache drift and laptop memory. Native sampling and
prompt/style preprocessing follow. Existing yue2.cpp SheetSage2 melody/chord MIDI
exports provide a later [transcription integration seam](docs/TRANSCRIPTION_INTEGRATION.md).

# mulacover.cpp

Native MuLaCover generator work for gary4local/gary4juce. This repository owns
symbolic MIDI conditioning, prompt/style preparation, the autoregressive model,
sampling and its reference fixtures. Sibling `heartcodec.cpp` owns RVQ/flow/SQ
decoding and will be consumed as `heartcodec::heartcodec`, sharing one copy of
`betweentwomidnights/ggml` with the generator and other model libraries.

The first milestone is frozen upstream reference data. **No native generator or
Q4_K_M generator is implemented yet.** The laptop is the primary C++ workspace;
full upstream reference execution stays in dedicated Spark Docker containers.

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

Next: audit generator tensor mapping and tokenizer metadata, implement one native
teacher-forced frame against the FP32 fixtures, then validate incremental cache
updates and sampling before a Q4_K_M weight policy. Audio transcription and the
full HeartCodec audio tokenizer are later components.

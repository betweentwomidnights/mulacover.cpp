# MuLaCover development dependency

The user authorized continued MuLaCover work with the CUDA fix before a coordinated
GGML bump. Use clean commit `0f2ed045eaff6bf01bed074878b6988248efbc22`, recorded in
`dependencies/ggml-development.json`, from draft
[GGML PR #21](https://github.com/betweentwomidnights/ggml/pull/21). Its baseline is
`b77ae790e628270af46a8af1eca38441c23f1252` and PR target is
`feature/shared-sa3-acestep-v0.17`.

HeartCodec's recorded submodule and shared working checkouts stay at `b77ae79`.
The MuLaCover development build supplies one GGML target to HeartCodec and itself.
No project-local kernels, second GGML instance, PR merge or sibling pin updates
are involved. Do not initiate ggml-bump-check or change sa3.cpp, yuey.cpp,
acestep.cpp, stems.cpp or audiocraft.cpp until the user starts that process after
MuLaCover is working.

## Laptop

The isolated checkout already exists at `C:/dev/_ggml_reviews/cuda-f32-source-b77`.
To recreate it elsewhere, clone the fork separately and check out the full SHA.
The CMake override verifies exact HEAD and clean status before configuring.

```powershell
.\scripts\build-windows.ps1 -Backend cuda -DevelopmentGgml C:/dev/_ggml_reviews/cuda-f32-source-b77
.\scripts\build-windows.ps1 -Backend cpu -DevelopmentGgml C:/dev/_ggml_reviews/cuda-f32-source-b77
.\scripts\evaluate-windows.ps1 -Backend cuda -Capacity 256
```

Development artifacts live in `build-windows-development-{cpu,cuda}`. CPU/DLL
baseline builds remain available through the default script options. The default
evaluation model is ignored `models/mulacover-q4_k_m.gguf`, already copied from
Spark and verified by SHA-256. Evaluation requires NumPy; the GGUF quantizer tests
also need the versions in `requirements-dev.txt`. The helper uses the sibling
HeartCodec virtualenv when available.

## Spark (Docker only)

The isolated clean checkout is `/home/kev/dev/ggml-review-b77`. Override its path
with `MULACOVER_DEVELOPMENT_GGML_DIR` if necessary. Host operations are Git checks,
file operations and Docker orchestration. All builds, Python, conversion and
inference run in dedicated HeartCodec images, without production-container changes.

```bash
bash reference/run-spark.sh build-development
bash reference/run-spark.sh parity-development
# Produces a new model; refuses an existing output or partial file.
bash reference/run-spark.sh quantize-development
MULACOVER_CACHE_CAPACITY=256 bash reference/run-spark.sh measure-development
```

The Docker profile mounts the isolated checkout read-only over `/codec/ggml`,
reusing the existing `build-review-b77-cuda` directory and one in-tree target.
No shared on-disk checkout is altered. The parity harness checks source SHA/clean
status and linked binary SHA, explicitly marks the development revision as an
unadopted review candidate, and records cache capacity/model SHA. Default
`build|parity` still exercise the unchanged `b77ae79` baseline (whose CUDA precision
test is expected to fail). Candidate execution uses TF32 disabled and cuBLAS
compute type `auto`.

FP32 parity passes all 600 frozen tensors on this profile. Quantized evaluation is
separate: unchanged FP32 gates remain diagnostics; successful execution does not
accept quantization quality. See [QUANTIZATION.md](QUANTIZATION.md).

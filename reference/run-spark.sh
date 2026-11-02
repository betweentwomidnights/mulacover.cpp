#!/usr/bin/env bash
# Invoke on Spark; Python/model execution stays inside the dedicated image.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
refs="$(dirname "$project_root")/_heartcodec_refs"
codec="${MULACOVER_HEARTCODEC_DIR:-$(dirname "$project_root")/heartcodec.cpp}"
image=heartcodec-reference:20261101
common=(--rm --user "$(id -u):$(id -g)" -e USER=heartcodec -e LOGNAME=heartcodec
  -e HOME=/workspace -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v "$project_root:/workspace" -w /workspace)
case "${1:-}" in
  convert)
    docker run "${common[@]}" --network none -e PYTHONPATH=/gguf-deps \
      -v "$refs:/refs:ro" -v "$codec/.python-deps:/gguf-deps:ro" "$image" \
      /opt/sa3-venv/bin/python tools/convert-to-gguf.py --upstream /refs/MuLaCover \
      --source models/generator.json --output models/mulacover-f32.gguf
    ;;
  build)
    docker run "${common[@]}" --gpus all --network none -v "$codec:/codec:ro" \
      heartcodec-dev:20261101 /bin/bash -lc \
      'cmake -S . -B build-shared-b77-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DMULACOVER_HEARTCODEC_DIR=/codec -DMULACOVER_USE_INSTALLED_GGML=OFF \
        -DHEARTCODEC_BUILD_TOOLS=ON -DHEARTCODEC_BUILD_TESTS=ON \
        -DMULACOVER_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=121a-real \
        > build-native-configure.log 2>&1 &&
       cmake --build build-shared-b77-cuda -j4 > build-native.log 2>&1 &&
       ctest --test-dir build-shared-b77-cuda --output-on-failure'
    ;;
  parity)
    docker run "${common[@]}" --gpus all --network none -e GGML_CUDA_TF32=0 \
      -v "$codec:/codec:ro" "$image" /opt/sa3-venv/bin/python tools/run-parity.py \
      --reference refdata/upstream-f32 --model models/mulacover-f32.gguf \
      --cli build-shared-b77-cuda/bin/mulacover-parity --output refdata/b77-native-baseline \
      --backend cuda --ggml-source /codec/ggml
    ;;
  prepare)
    docker run "${common[@]}" -e HF_HUB_OFFLINE=0 -e HF_HUB_DISABLE_XET=1 \
      -e HF_HOME=/workspace/.cache/huggingface "$image" /opt/sa3-venv/bin/python reference/prepare_models.py
    ;;
  float32|bfloat16)
    suffix=f32
    if [[ "$1" == bfloat16 ]]; then suffix=bf16; fi
    docker run "${common[@]}" --gpus all --network none -v "$refs:/refs:ro" \
      "$image" /opt/sa3-venv/bin/python reference/export_reference.py --upstream /refs/MuLaCover \
      --output "refdata/upstream-$suffix" --dtype "$1"
    ;;
  *) echo 'Usage: bash reference/run-spark.sh prepare|float32|bfloat16|convert|build|parity' >&2; exit 2 ;;
esac

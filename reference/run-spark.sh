#!/usr/bin/env bash
# Invoke on Spark; Python/model execution stays inside the dedicated image.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
refs="$(dirname "$project_root")/_heartcodec_refs"
codec="${MULACOVER_HEARTCODEC_DIR:-$(dirname "$project_root")/heartcodec.cpp}"
image=heartcodec-reference:20261101
command="${1:-}"
build=build-shared-b77-cuda
suffix=b77-native-baseline
development_mount=()
revision_args=()
if [[ "$command" == *-development ]]; then
  command="${command%-development}"
  development="${MULACOVER_DEVELOPMENT_GGML_DIR:-$(dirname "$project_root")/ggml-review-b77}"
  revision="$(sed -n 's/^  "revision": "\([a-f0-9]*\)",[[:space:]]*$/\1/p' "$project_root/dependencies/ggml-development.json")"
  [[ ${#revision} == 40 && "$(git -C "$development" rev-parse HEAD)" == "$revision" && -z "$(git -C "$development" status --porcelain)" ]] || {
    echo 'Development GGML checkout does not match the clean pinned profile' >&2; exit 1;
  }
  development_mount=(-v "$development:/codec/ggml:ro")
  revision_args=(--review-ggml-revision "$revision")
  build=build-review-b77-cuda
  suffix=development-native-cuda
fi
mix="${MULACOVER_QUANTIZATION_MIX:-q4_k_m}"
capacity="${MULACOVER_CACHE_CAPACITY:-8192}"
common=(--rm --user "$(id -u):$(id -g)" -e USER=heartcodec -e LOGNAME=heartcodec
  -e HOME=/workspace -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e GGML_CUDA_TF32=0 -e GGML_CUDA_CUBLAS_COMPUTE_TYPE=auto
  -v "$project_root:/workspace" -w /workspace)
case "$command" in
  encode-prefix)
    # Pinned CPU reference avoids the image's unsupported GB10 NVRTC STFT path.
    # The host only orchestrates Docker; input may span multiple encoder chunks.
    encoder_audio="${MULACOVER_ENCODER_AUDIO:?Set the container-relative input WAV path}"
    encoder_output="${MULACOVER_ENCODER_OUTPUT:?Set a fresh container-relative output directory}"
    codec_snapshot=/codec/.cache/huggingface/hub/models--HeartMuLa--HeartCodec-oss-20260123/snapshots/f889dab0532cfa4bf459f2a3367eb6d346b8eeda
    docker run "${common[@]}" --network none \
      -e HF_HOME=/workspace/.cache/huggingface -v "$codec:/codec:ro" -v "$refs:/refs:ro" \
      "$image" /opt/sa3-venv/bin/python reference/encode-prefix.py \
      --upstream /refs/heartlib --codec-snapshot "$codec_snapshot" \
      --audio "$encoder_audio" --output "$encoder_output" --device cpu
    ;;
  convert)
    docker run "${common[@]}" --network none -e PYTHONPATH=/gguf-deps \
      -v "$refs:/refs:ro" -v "$codec/.python-deps:/gguf-deps:ro" "$image" \
      /opt/sa3-venv/bin/python tools/convert-to-gguf.py --upstream /refs/MuLaCover \
      --source models/generator.json --output models/mulacover-f32.gguf
    ;;
  build)
    docker run "${common[@]}" --gpus all --network none -v "$codec:/codec:ro" "${development_mount[@]}" \
      -e MULACOVER_BUILD_DIR="$build" \
      heartcodec-dev:20261101 /bin/bash -lc \
      'cmake -S . -B "$MULACOVER_BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DMULACOVER_HEARTCODEC_DIR=/codec -DMULACOVER_USE_INSTALLED_GGML=OFF \
        -DMULACOVER_GGML_SOURCE_DIR= -DGGML_BUILD_TESTS=OFF \
        -DHEARTCODEC_BUILD_TOOLS=ON -DHEARTCODEC_BUILD_TESTS=ON \
        -DMULACOVER_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=121a-real \
        > build-native-configure.log 2>&1 &&
       cmake --build "$MULACOVER_BUILD_DIR" -j4 > build-native.log 2>&1 &&
       ctest --test-dir "$MULACOVER_BUILD_DIR" --output-on-failure'
    ;;
  parity)
    docker run "${common[@]}" --gpus all --network none -e GGML_CUDA_TF32=0 \
      -v "$codec:/codec:ro" "${development_mount[@]}" "$image" /opt/sa3-venv/bin/python tools/run-parity.py \
      --reference refdata/upstream-f32 --model models/mulacover-f32.gguf \
      --cli "$build/bin/mulacover-parity" --output "refdata/$suffix" \
      --backend cuda --ggml-source /codec/ggml "${revision_args[@]}" --capacity "$capacity"
    ;;
  quantize)
    docker run "${common[@]}" --network none -v "$codec:/codec:ro" "${development_mount[@]}" \
      heartcodec-dev:20261101 "$build/bin/mulacover-quantize" --input models/mulacover-f32.gguf \
      --output "models/mulacover-$mix.gguf" --mix "$mix"
    ;;
  measure)
    docker run "${common[@]}" --gpus all --network none -v "$codec:/codec:ro" "${development_mount[@]}" \
      "$image" /opt/sa3-venv/bin/python tools/run-parity.py \
      --reference refdata/upstream-f32 --model "models/mulacover-$mix.gguf" \
      --cli "$build/bin/mulacover-parity" --output "refdata/development-$mix-cuda-cap$capacity" \
      --backend cuda --ggml-source /codec/ggml "${revision_args[@]}" --capacity "$capacity" --measure-quantization
    ;;
  sampling-oracle|sequence-oracle)
    exporter=export-sampling.py
    oracle_args=(--output refdata/sampling-upstream)
    if [[ "$command" == sequence-oracle ]]; then
      exporter=export-sequence.py
      oracle_args=(--prepared refdata/upstream-f32 --output refdata/sequence-f32)
    fi
    docker run "${common[@]}" --gpus all --network none -v "$refs:/refs:ro" "$image" \
      /opt/sa3-venv/bin/python "reference/$exporter" --upstream /refs/MuLaCover "${oracle_args[@]}"
    ;;
  sampling)
    docker run "${common[@]}" --network none "$image" /opt/sa3-venv/bin/python tools/compare-sampling.py \
      --reference refdata/sampling-upstream --cli "$build/bin/mulacover-sampling-check" --output refdata/sampling-native
    ;;
  sequence)
    docker run "${common[@]}" --gpus all --network none -v "$codec:/codec:ro" "${development_mount[@]}" "$image" \
      /opt/sa3-venv/bin/python tools/run-parity.py --reference refdata/sequence-f32 --model models/mulacover-f32.gguf \
      --cli "$build/bin/mulacover-parity" --output refdata/sequence-native-f32 --backend cuda --ggml-source /codec/ggml \
      "${revision_args[@]}" --capacity 159 --cases conditional,cfg-1.5,sample-cfg
    ;;
  generate)
    generator_mix="${MULACOVER_GENERATOR_MIX:-q4_k_m}"
    docker run "${common[@]}" --gpus all --network none -v "$codec:/codec:ro" "${development_mount[@]}" "$image" \
      /opt/sa3-venv/bin/python tools/generate.py --reference "${MULACOVER_GENERATION_REFERENCE:-refdata/upstream-f32}" --model "models/mulacover-$generator_mix.gguf" \
      --cli "$build/bin/mulacover-generate" --ggml-source /codec/ggml --frames "${MULACOVER_GENERATION_FRAMES:-100}" \
      --capacity "${MULACOVER_GENERATION_CAPACITY:-256}" --output "${MULACOVER_GENERATION_OUTPUT:-refdata/generate-$generator_mix-spark}" \
      --rng "${MULACOVER_GENERATION_RNG:-native}" --rng-cuda-blocks "${MULACOVER_RNG_CUDA_BLOCKS:-0}" \
      --flow-model /codec/models/heartcodec-flow-f32.gguf --sq-model /codec/models/heartcodec-sq-codec-f32.gguf
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
  *) echo 'Usage: bash reference/run-spark.sh encode-prefix|prepare|float32|bfloat16|convert|sampling-oracle|sequence-oracle|build-development|parity-development|quantize-development|measure-development|sampling-development|sequence-development|generate-development (build/parity also support the shared baseline)' >&2; exit 2 ;;
esac

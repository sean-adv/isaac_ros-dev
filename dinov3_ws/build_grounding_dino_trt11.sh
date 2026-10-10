#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$script_dir/../dinov3_trt11_venv/bin/python"
artifact_dir="$script_dir/artifacts/tensorrt11"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH=""
export LD_LIBRARY_PATH="$script_dir/../dinov3_trt11_venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
if [[ ! -f "$artifact_dir/calibration.npz" ]]; then
    echo "Missing real-image calibration.npz; see README.md." >&2
    exit 1
fi
"$python_bin" -m modelopt.onnx.autocast \
  --onnx_path "$script_dir/artifacts/tensorrt/grounding_dino_wingdzero.onnx" \
  --output_path "$artifact_dir/grounding_dino_mixed_fp16.onnx" \
  --calibration_data "$artifact_dir/calibration.npz" --keep_io_types --providers cpu \
  --op_types_to_exclude LayerNormalization Softmax ReduceMean ReduceSum ReduceMax \
  --log_level INFO > "$artifact_dir/autocast.log" 2>&1
"$python_bin" "$script_dir/build_grounding_dino_trt11.py" \
  --onnx "$artifact_dir/grounding_dino_mixed_fp16.onnx" \
  --output "$artifact_dir/grounding_dino_mixed_fp16.engine" \
  > "$artifact_dir/build-grounding.log" 2>&1

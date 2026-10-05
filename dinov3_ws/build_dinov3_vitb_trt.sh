#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/../dinov3_venv/bin/activate"
artifact_dir="$script_dir/artifacts/tensorrt"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
rope_precision="$(python "$script_dir/trt_rope_precision.py" "$artifact_dir/dinov3_vitb16_dynamic.onnx")"
/usr/src/tensorrt/bin/trtexec \
    --onnx="$artifact_dir/dinov3_vitb16_dynamic.onnx" \
    --saveEngine="$artifact_dir/dinov3_vitb16_dynamic_bf16.engine" \
    --minShapes=pixel_values:1x3x32x32 \
    --optShapes=pixel_values:1x3x720x1280 \
    --maxShapes=pixel_values:1x3x2560x2560 \
    --shapes=pixel_values:1x3x720x1280 \
    --bf16 --noTF32 --memPoolSize=workspace:4096 \
    --precisionConstraints=obey \
    --layerPrecisions="$rope_precision" --layerOutputTypes="$rope_precision" \
    --timingCacheFile="$artifact_dir/timing_vitb16_dynamic_bf16.cache" \
    --profilingVerbosity=detailed --warmUp=500 --duration=3 \
    > "$artifact_dir/build_vitb16_dynamic_bf16.log" 2>&1
cp "$artifact_dir/dinov3_vitb16_dynamic.json" \
   "$artifact_dir/dinov3_vitb16_dynamic_bf16.json"
echo "Built $artifact_dir/dinov3_vitb16_dynamic_bf16.engine"

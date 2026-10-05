#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/../dinov3_venv/bin/activate"
artifact_dir="$script_dir/artifacts/tensorrt"
trtexec=/usr/src/tensorrt/bin/trtexec

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
test -f "$artifact_dir/dinov3_convnext_large.onnx"

for precision in fp32 bf16; do
    flags=(--noTF32)
    if [[ "$precision" == bf16 ]]; then
        flags+=(--bf16)
    fi
    echo "Building $precision engine; log: $artifact_dir/build_$precision.log"
    "$trtexec" \
        --onnx="$artifact_dir/dinov3_convnext_large.onnx" \
        --saveEngine="$artifact_dir/dinov3_convnext_large_$precision.engine" \
        --timingCacheFile="$artifact_dir/timing_convnext_large_$precision.cache" \
        --profilingVerbosity=detailed \
        --warmUp=500 --duration=3 \
        "${flags[@]}" > "$artifact_dir/build_$precision.log" 2>&1
    echo "Completed $precision engine"
done

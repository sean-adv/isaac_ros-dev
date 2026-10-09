#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
artifact_dir="$script_dir/artifacts/tensorrt"
"$script_dir/../dinov3_venv/bin/python" "$script_dir/export_grounding_dino.py"
/usr/src/tensorrt/bin/trtexec \
    --onnx="$artifact_dir/grounding_dino_community.onnx" \
    --saveEngine="$artifact_dir/grounding_dino_community_fp32.engine" \
    --minShapes=input_ids:1x3,token_type_ids:1x3,attention_mask:1x3 \
    --optShapes=input_ids:1x16,token_type_ids:1x16,attention_mask:1x16 \
    --maxShapes=input_ids:1x256,token_type_ids:1x256,attention_mask:1x256 \
    --noTF32 --memPoolSize=workspace:4096 \
    --timingCacheFile="$artifact_dir/timing_grounding_dino_community_fp32.cache" \
    --profilingVerbosity=detailed --skipInference \
    > "$artifact_dir/build_grounding_dino_community_fp32.log" 2>&1
cp "$artifact_dir/grounding_dino_community.source.json" \
   "$artifact_dir/grounding_dino_community_fp32.json"
echo "Built $artifact_dir/grounding_dino_community_fp32.engine"

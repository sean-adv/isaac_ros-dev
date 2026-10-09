#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$script_dir")"
set +u
source /opt/ros/jazzy/setup.bash
source "$workspace_dir/dinov3_trt11_venv/bin/activate"
source "$script_dir/install/setup.bash"
set -u
export PYTHONPATH="$script_dir/src/dinov3_cases:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
if [[ "${1:-}" == "--ros-args" ]]; then shift; fi
exec python "$script_dir/launch_pipeline.py" --model vits16 --ros-args \
  -p "engine_path:=$script_dir/artifacts/tensorrt11/dinov3_vits16_fp16_224.engine" \
  -p "grounding_engine_path:=$script_dir/artifacts/tensorrt11/grounding_dino_mixed_fp16.engine" \
  -p object_max_side:=224 -p object_square_input:=true "$@"

#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$script_dir")"
set +u
source /opt/ros/jazzy/setup.bash
source "$workspace_dir/dinov3_trt11_venv/bin/activate"
set -u
export PYTHONPATH="$script_dir/src/dinov3_cases:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec python "$script_dir/launch_pipeline.py" "$@"

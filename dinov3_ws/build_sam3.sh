#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$script_dir/../dinov3_trt11_venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec "$script_dir/../dinov3_trt11_venv/bin/python" "$script_dir/build_sam3.py" "$@"

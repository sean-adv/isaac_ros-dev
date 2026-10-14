#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="$script_dir/../dinov3_trt11_venv/bin/python"
[[ -x "$python" ]] || { echo "Missing environment: $python" >&2; exit 1; }
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$script_dir/../TensorRT-Model-Connect/core/builder:$script_dir/../TensorRT-Model-Connect:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$script_dir/../dinov3_trt11_venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec "$python" "$script_dir/build_dinov3.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${DINOV3_VENV:-$project_dir/.venv}"
python_bin="${PYTHON_BIN:-python3.12}"
if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi
# CUDA wheels are selected explicitly, so pip does not silently choose CPU torch.
"$venv_dir/bin/python" -m pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
"$venv_dir/bin/python" -m pip install -c "$project_dir/constraints-thor.txt" \
  -r "$project_dir/requirements-build.txt"
"$venv_dir/bin/python" -m pip check

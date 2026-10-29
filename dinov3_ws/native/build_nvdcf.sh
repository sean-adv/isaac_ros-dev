#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ds_dir="${DEEPSTREAM_DIR:-/opt/nvidia/deepstream/deepstream-9.1}"
mkdir -p "$project_dir/artifacts/nvdcf"
g++ -std=c++17 -O2 -Wall -Wextra "$project_dir/native/nvdcf_bridge.cpp" \
  -I"$ds_dir/sources/includes" -I/usr/local/cuda/include \
  -L"$ds_dir/lib" -L/usr/local/cuda/lib64 \
  -Wl,-rpath,"$ds_dir/lib" -Wl,-rpath,/usr/local/cuda/lib64 \
  -lnvbufsurface -lcudart -ldl -o "$project_dir/artifacts/nvdcf/nvdcf_bridge"

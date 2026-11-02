#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_dir/artifacts/cuda"
/usr/local/cuda/bin/nvcc -std=c++17 -O3 --shared -Xcompiler -fPIC -arch="${CUDA_ARCH:-sm_110}" \
  "$project_dir/native/gpu_geometry.cu" -o "$project_dir/artifacts/cuda/libgpu_geometry.so"

#!/usr/bin/env bash
# Run inside the CUDA/DeepStream development container, without sudo.
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${NATIVE_BUILD_DIR:-$native_dir/../build/native}"
target="${1:-all}"
case "$target" in
  all|nvdcf_bridge) bridges=ON ;;
  gpu_geometry) bridges=OFF ;;
  *) echo "Usage: bash $0 [all|gpu_geometry|nvdcf_bridge]" >&2; exit 2 ;;
esac
# Accept the old CUDA_ARCH=sm_110 spelling as well as CMake's numeric spelling.
architecture="${CUDA_ARCH:-110}"
cmake -S "$native_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${architecture#sm_}" \
  -DDEEPSTREAM_DIR="${DEEPSTREAM_DIR:-/opt/nvidia/deepstream/deepstream-9.1}" \
  -DNATIVE_BUILD_BRIDGES="$bridges"
cmake --build "$build_dir" --parallel --target "$target"

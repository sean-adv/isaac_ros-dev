#!/usr/bin/env bash
# Requires DART's CUTLASS checkout, CUDA 13 and matching TensorRT 11.1 headers.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$script_dir")"
output="$script_dir/artifacts/sam3/trt11"
plugin_src="$workspace_dir/DART/dartf/plugins"
cutlass="$workspace_dir/DART/dartf/third_party/cutlass"
trt_lib="$workspace_dir/dinov3_trt11_venv/lib/python3.12/site-packages/tensorrt_libs"
trt_inc="${TRT_INC:-$output/include}"
mkdir -p "$output/plugins"
"${NVCC:-/usr/local/cuda/bin/nvcc}" -O3 -std=c++17 -arch="sm_${ARCH:-110}" \
  --use_fast_math --expt-relaxed-constexpr -DLQ_STAGES=5 -Xcompiler -fPIC -shared \
  -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$trt_inc" \
  "$plugin_src/lq_plugins.cu" "$plugin_src/lq_attn_plugin.cu" \
  "$plugin_src/lq_block_plugins.cu" "$plugin_src/lq_ground_plugins.cu" \
  "$plugin_src/lq_memattn_plugin.cu" -o "$output/plugins/lq_plugins.so" \
  -L "$trt_lib" -L /usr/local/cuda/lib64 -l:libnvinfer.so.11 -lcublasLt -lcublas

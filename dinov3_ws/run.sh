#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Use the same large-image transport in camera and inference processes.
# Keep explicit caller-provided DDS profiles.
if [[ -z "${FASTDDS_DEFAULT_PROFILES_FILE:-}" && -z "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]]; then
  export FASTRTPS_DEFAULT_PROFILES_FILE="$script_dir/config/fastdds_rgbd.xml"
fi

# Background 3D/BLAS pools otherwise compete with per-frame NvDCF tracking.
# These defaults apply before NumPy/Open3D/Torch imports and to the SAM3 child.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
venv_dir="${DINOV3_VENV:-$script_dir/.venv}"
set +u
source /opt/ros/jazzy/setup.bash
set -u
if [[ ! -x "$venv_dir/bin/python" ]]; then
  printf '%s\n' 'Python environment missing. Run setup_env.sh first.' >&2
  exit 1
fi
export VIRTUAL_ENV="$venv_dir"
export PATH="$venv_dir/bin:$PATH"
export PYTHONPATH="$script_dir/src/dinov3_cases:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec "$VIRTUAL_ENV/bin/python" "$script_dir/launch_pipeline.py" "$@"

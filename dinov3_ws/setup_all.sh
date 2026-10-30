#!/usr/bin/env bash
# Run as the workspace user inside the Thor/Isaac ROS container.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $(uname -m) != aarch64 ]]; then
  echo 'This setup targets Jetson Thor ARM64 inside the Isaac ROS container.' >&2
  exit 1
fi

# Match the runtime preference: workspace bridge first, image bridge second.
bridge="$project_dir/artifacts/nvdcf/nvdcf_bridge"
if [[ ! -f "$bridge" ]]; then
  bridge=/usr/local/bin/dinov3-nvdcf-bridge
fi
nvdcf_ready=false
if [[ -x "$bridge" && -f /opt/nvidia/deepstream/deepstream-9.1/lib/libnvds_nvmultiobjecttracker.so ]]; then
  nvdcf_ready=true
fi

# Check privileges before spending time installing Python dependencies.
if [[ "$nvdcf_ready" == false && $EUID -ne 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo 'NvDCF installation requires sudo; ask the container administrator to install it.' >&2
    exit 1
  fi
  sudo -v
fi

echo '[1/2] Setting up the Python environment...'
bash "$project_dir/setup_env.sh"

echo '[2/2] Setting up NvDCF...'
if [[ "$nvdcf_ready" == true ]]; then
  echo "Using existing NvDCF bridge: $bridge"
elif [[ $EUID -eq 0 ]]; then
  bash "$project_dir/native/install_nvdcf.sh"
else
  sudo bash "$project_dir/native/install_nvdcf.sh"
fi

echo 'Environment setup complete. Model download/build is a separate step; see BUILD_GUIDE.md.'

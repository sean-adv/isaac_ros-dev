#!/usr/bin/env bash
# Install inside the Thor/Isaac ROS container; does not modify the model venv.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
package="${DEEPSTREAM_DEB:-/tmp/deepstream-9.1-arm64.deb}"
if [[ $(uname -m) != aarch64 ]]; then
  echo 'This installer targets Jetson Thor ARM64.' >&2; exit 1
fi
if [[ ! -f /opt/nvidia/deepstream/deepstream-9.1/lib/libnvds_nvmultiobjecttracker.so ]]; then
  if ! dpkg-deb --info "$package" >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$package.download" \
      https://github.com/NVIDIA/DeepStream/releases/download/v9.1.0/deepstream-9.1_9.1.0-1_arm64.deb
    mv -- "$package.download" "$package"
  fi
  apt-get update
  apt-get install -y "$package"
fi
apt-get install -y g++ libmosquitto1 libyaml-cpp0.8
ldconfig
bash "$project_dir/native/build_nvdcf.sh"

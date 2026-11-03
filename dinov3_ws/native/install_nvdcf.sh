#!/usr/bin/env bash
# Compatibility installer. New callers should install dependencies, then build as the workspace user.
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$native_dir/install_deps.sh"
bash "$native_dir/build.sh" nvdcf_bridge

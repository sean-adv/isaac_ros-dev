#!/usr/bin/env bash
# Compatibility entry point; dependency tracking lives in CMake.
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$native_dir/build.sh" gpu_geometry

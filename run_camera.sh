#!/usr/bin/env bash
# Keep the root entry point compatible; camera settings live in one place.
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/dinov3_ws/run_camera.sh" "$@"

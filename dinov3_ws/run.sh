#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$script_dir")"
source /opt/ros/jazzy/setup.bash
source "$workspace_dir/dinov3_venv/bin/activate"
source "$script_dir/install/setup.bash"
exec python "$script_dir/launch_pipeline.py" "$@"

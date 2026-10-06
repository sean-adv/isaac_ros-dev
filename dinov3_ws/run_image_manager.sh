#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$script_dir")"
export PYTHONPATH="$script_dir/src/dinov3_cases${PYTHONPATH:+:$PYTHONPATH}"
exec "$workspace_dir/dinov3_venv/bin/python" -m dinov3_cases.image_manager "$@"

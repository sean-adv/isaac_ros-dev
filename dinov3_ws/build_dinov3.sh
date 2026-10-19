#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${DINOV3_VENV:-$project_dir/.venv}/bin/python" "$project_dir/build_models.py" --target cat2real "$@"

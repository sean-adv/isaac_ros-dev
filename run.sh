#!/usr/bin/env bash
# Run from the host or from inside the Isaac ROS development container.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f /.dockerenv ]]; then
  exec bash "$script_dir/dinov3_ws/run.sh" "$@"
fi

container="${DINOV3_CONTAINER:-money_arm_medicine}"
container_workspace="${DINOV3_CONTAINER_WORKSPACE:-/workspaces/medicine_recog/isaac_ros-dev}"
container_user="${DINOV3_CONTAINER_USER:-admin}"
if ! command -v docker >/dev/null 2>&1; then
  printf '%s\n' 'Docker is required on the host. Run this script inside the development container instead.' >&2
  exit 1
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]]; then
  printf 'Container %s is not running. Start it before running inference.\n' "$container" >&2
  exit 1
fi
# Allocate a terminal only for interactive callers; preserve ROS arguments verbatim.
terminal_args=()
if [[ -t 0 && -t 1 ]]; then
  terminal_args=(-it)
fi
exec docker exec "${terminal_args[@]}" --user "$container_user" \
  --workdir "$container_workspace" "$container" bash ./dinov3_ws/run.sh "$@"

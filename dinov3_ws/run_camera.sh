#!/usr/bin/env bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$script_dir")"
source /opt/ros/jazzy/setup.bash
source "$workspace_dir/orbbec_sdk/install/setup.bash"
exec ros2 launch orbbec_camera gemini_330_series.launch.py \
  enable_color:=true enable_depth:=true \
  enable_left_ir:=false enable_right_ir:=false \
  color_width:=1920 color_height:=1080 color_fps:=30 \
  depth_width:=1280 depth_height:=720 depth_fps:=30 \
  depth_registration:=true \
  "$@"

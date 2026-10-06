import sys
if sys.prefix == '/workspaces/isaac_ros-dev/dinov3_venv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/workspaces/isaac_ros-dev/dinov3_ws/install/dinov3_cases'

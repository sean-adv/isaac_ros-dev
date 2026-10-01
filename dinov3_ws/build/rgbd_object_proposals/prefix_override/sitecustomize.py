import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/workspaces/isaac_ros-dev/dinov3_ws/install/rgbd_object_proposals'

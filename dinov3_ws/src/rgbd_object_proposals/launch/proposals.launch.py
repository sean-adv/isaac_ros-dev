from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = str(Path(get_package_share_directory('rgbd_object_proposals')) / 'config' / 'proposals.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=params),
        Node(package='rgbd_object_proposals', executable='rgbd_object_proposal_node',
             name='rgbd_object_proposal_node', output='screen',
             parameters=[LaunchConfiguration('params_file')]),
    ])

"""T21 local elevation mapping: elevation_mapping_cupy on lidar + D435i (merged by cloud_merger).

ros2 launch hill_patrol_bringup elevation.launch.py
Needs TF map->base_link (localization.launch.py) and a CUDA GPU with cupy + torch.
Output: /elevation_mapping_node/elevation_map_raw (grid_map_msgs/GridMap, layers
elevation, traversability, variance) in frame map.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile


def generate_launch_description():
    upstream = Path(get_package_share_directory('elevation_mapping_cupy')) / 'config/core/core_param.yaml'
    overlay = Path(get_package_share_directory('hill_patrol_bringup')) / 'config/elevation_sim.yaml'
    return LaunchDescription([Node(
        package='hill_patrol_sim', executable='cloud_merger', name='elevation_cloud_merger',
        parameters=[{'use_sim_time': True}], output='screen'), Node(
        package='elevation_mapping_cupy', executable='elevation_mapping_node',
        name='elevation_mapping_node', output='screen',
        parameters=[ParameterFile(str(upstream), allow_substs=True),
                    ParameterFile(str(overlay), allow_substs=True),
                    {'use_sim_time': True}])])

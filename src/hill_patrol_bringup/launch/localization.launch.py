"""T23 localization bringup: sim (navigation mode) + cloud_adapter + Lightning-LM localization
+ localization monitor.

ros2 launch hill_patrol_bringup localization.launch.py world:=small
    map_dir:=~/hill_patrol_ws/maps/data/hill_world_small x:=0 y:=0 yaw:=0
    global_init:=false sim:=true

Lightning-LM publishes TF map->base_link and /lightning/pose; its "base_link" is
the IMU/lidar body it estimates, i.e. our mid360_frame. A static transform
lightning_body -> base_link (inverse lidar mount) is therefore inserted by
relaying the published transform under the correct child frame.
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLDS = {'small': 'hill_world_small', 'full': 'hill_world'}


def _setup(context):
    world = LaunchConfiguration('world').perform(context)
    map_dir = Path(os.path.expanduser(LaunchConfiguration('map_dir').perform(context)))
    workdir = map_dir.parent.parent  # Lightning-LM writes ./log relative to its cwd
    bringup = Path(get_package_share_directory('hill_patrol_bringup'))
    config = bringup / 'config'
    loc = Path(get_package_prefix('lightning')) / 'lib/lightning/run_loc_online'
    command = [str(loc), f'--config={config / "lightning_sim.yaml"}', f'--map_path={map_dir}/']
    if LaunchConfiguration('global_init').perform(context) == 'true':
        command.append('--global_init=true')
    command += ['--', '--ros-args', '-p', 'use_sim_time:=true',
                '-r', '/tf:=/lightning/loc_tf']
    actions = []
    if LaunchConfiguration('sim').perform(context) == 'true':
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(bringup / 'launch/sim.launch.py')),
            launch_arguments={'world': world, 'mode': 'navigation',
                              'x': LaunchConfiguration('x').perform(context),
                              'y': LaunchConfiguration('y').perform(context),
                              'yaw': LaunchConfiguration('yaw').perform(context)}.items()))
    actions += [
        Node(package='hill_patrol_sim', executable='cloud_adapter', name='cloud_adapter',
             parameters=[{'use_sim_time': True, 'min_range': 0.1}], output='screen'),
        ExecuteProcess(cmd=command, cwd=str(workdir), output='screen', name='lightning_loc'),
        Node(package='hill_patrol_bringup', executable='localization_monitor.py',
             name='localization_monitor',
             parameters=[{'use_sim_time': True, 'safety_path': str(config / 'm20_safety.yaml'),
                          'sensors_path': str(Path(get_package_share_directory('hill_patrol_sim'))
                                              / 'config/sensors.yaml')}],
             output='screen'),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='small', choices=list(WORLDS)),
        DeclareLaunchArgument('map_dir', default_value='~/hill_patrol_ws/maps/data/hill_world_small'),
        DeclareLaunchArgument('global_init', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('sim', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        OpaqueFunction(function=_setup),
    ])

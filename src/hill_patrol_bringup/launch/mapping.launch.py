"""T20 mapping bringup: sim (teleop mode) + cloud_adapter + Lightning-LM online SLAM.

ros2 launch hill_patrol_bringup mapping.launch.py world:=small workdir:=~/hill_patrol_ws/maps
    drive:=true laps:=1

Lightning-LM resolves ./data/<map_id>/ and ./log/ against its working directory,
so it runs inside `workdir`. Save the map with
    ros2 service call /lightning/save_map lightning/srv/SaveMap "{map_id: hill_world_small}"
drive:=true starts the test-only route driver (stand-in for the operator's joystick).
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLDS = {'small': 'hill_world_small', 'full': 'hill_world'}


def _setup(context):
    world = LaunchConfiguration('world').perform(context)
    workdir = Path(os.path.expanduser(LaunchConfiguration('workdir').perform(context)))
    workdir.mkdir(parents=True, exist_ok=True)
    bringup = Path(get_package_share_directory('hill_patrol_bringup'))
    sim_share = Path(get_package_share_directory('hill_patrol_sim'))
    slam = Path(get_package_prefix('lightning')) / 'lib/lightning/run_slam_online'
    actions = [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(bringup / 'launch/sim.launch.py')),
                                 launch_arguments={'world': world, 'mode': 'teleop'}.items()),
        Node(package='hill_patrol_sim', executable='cloud_adapter', name='cloud_adapter',
             parameters=[{'use_sim_time': True, 'min_range': 0.1}], output='screen'),
        # Lightning-LM initialises gravity from the first IMU samples: start it while still.
        ExecuteProcess(cmd=[str(slam), f'--config={bringup / "config/lightning_sim.yaml"}',
                            f'--map_name={WORLDS[world]}', '--rviz=true',
                            '--', '--ros-args', '-p', 'use_sim_time:=true'],
                       cwd=str(workdir), output='screen', name='lightning_slam'),
    ]
    if LaunchConfiguration('drive').perform(context) == 'true':
        route = sim_share / 'worlds' / f'{WORLDS[world]}_route.yaml'
        actions.append(TimerAction(period=float(LaunchConfiguration('drive_delay').perform(context)),
                                   actions=[Node(
            package='hill_patrol_bringup', executable='drive_route.py', name='drive_route',
            parameters=[{'use_sim_time': True, 'route_file': str(route),
                         'laps': int(LaunchConfiguration('laps').perform(context)),
                         'speed': float(LaunchConfiguration('speed').perform(context)),
                         'truth_tf': LaunchConfiguration('truth_tf').perform(context) == 'true'}],
            output='screen')]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='small', choices=list(WORLDS)),
        DeclareLaunchArgument('workdir', default_value='~/hill_patrol_ws/maps'),
        DeclareLaunchArgument('drive', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('drive_delay', default_value='15.0'),
        DeclareLaunchArgument('laps', default_value='1'),
        DeclareLaunchArgument('speed', default_value='0.5'),
        DeclareLaunchArgument('truth_tf', default_value='false', choices=['true', 'false']),
        OpaqueFunction(function=_setup),
    ])

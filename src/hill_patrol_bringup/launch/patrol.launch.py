"""T28/T29 patrol bringup: navigation stack + patrol_manager (+ gait rule monitor in sim).

ros2 launch hill_patrol_bringup patrol.launch.py world:=small laps:=1
    patrol_dir:=/tmp/patrol_small   (stair_zones.yaml + waypoints.yaml, map frame)

If patrol_dir lacks the files they are generated from the world truth by
make_patrol_config.py (stand-in for operator annotation, plan T29).
Start the patrol with: ros2 service call /patrol/start std_srvs/srv/Trigger
"""
import os
from pathlib import Path
import subprocess

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLDS = {'small': 'hill_world_small', 'full': 'hill_world'}


def _setup(context):
    world = LaunchConfiguration('world').perform(context)
    patrol_dir = Path(os.path.expanduser(LaunchConfiguration('patrol_dir').perform(context)))
    if not (patrol_dir / 'waypoints.yaml').is_file():
        subprocess.run(['ros2', 'run', 'hill_patrol_bringup', 'make_patrol_config.py',
                        WORLDS[world], str(patrol_dir)], check=True)
    bringup = Path(get_package_share_directory('hill_patrol_bringup'))
    sim_share = Path(get_package_share_directory('hill_patrol_sim'))
    sensors = yaml.safe_load((sim_share / 'config/sensors.yaml').read_text(encoding='utf-8'))
    length, width, _ = sensors['body']['size']['value']
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(bringup / 'launch/navigation.launch.py')),
            launch_arguments={'world': world, 'map_dir': LaunchConfiguration('map_dir').perform(context),
                              'zones_file': str(patrol_dir / 'stair_zones.yaml')}.items()),
        Node(package='patrol_manager', executable='patrol_node', name='patrol_manager',
             parameters=[{'use_sim_time': True, 'waypoints_file': str(patrol_dir / 'waypoints.yaml'),
                          'laps': int(LaunchConfiguration('laps').perform(context)),
                          'max_attempts': 2}],
             output='screen'),
        Node(package='hill_patrol_sim', executable='gait_rule_monitor', name='gait_rule_monitor',
             parameters=[{'use_sim_time': True, 'body_half_length': length / 2,
                          'body_half_width': width / 2,
                          'zones_file': str(sim_share / 'worlds' / f'{WORLDS[world]}_truth_zones.yaml'),
                          'treads_file': str(sim_share / 'worlds' / f'{WORLDS[world]}_truth_treads.yaml')}],
             output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='small', choices=list(WORLDS)),
        DeclareLaunchArgument('map_dir', default_value='~/hill_patrol_ws/maps/data/hill_world_small'),
        DeclareLaunchArgument('patrol_dir', default_value='/tmp/hill_patrol_patrol'),
        DeclareLaunchArgument('laps', default_value='1'),
        OpaqueFunction(function=_setup),
    ])

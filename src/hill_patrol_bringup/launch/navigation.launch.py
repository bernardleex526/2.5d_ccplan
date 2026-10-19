"""T27 navigation bringup: localization + elevation + traversability + Nav2 + gait_manager.

ros2 launch hill_patrol_bringup navigation.launch.py world:=small
    map_dir:=~/hill_patrol_ws/maps/data/hill_world_small zones_file:=<stair_zones.yaml>

zones_file defaults to the world's truth zones, standing in for the operator's
annotation (plan T29). Nav2's smoothed output goes to /cmd_vel, which m20_bridge
(navigation mode) converts to M20 axis commands.
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

WORLDS = {'small': 'hill_world_small', 'full': 'hill_world'}


def _setup(context):
    world = LaunchConfiguration('world').perform(context)
    map_dir = Path(os.path.expanduser(LaunchConfiguration('map_dir').perform(context)))
    bringup = Path(get_package_share_directory('hill_patrol_bringup'))
    config = bringup / 'config'
    zones = LaunchConfiguration('zones_file').perform(context) or str(
        Path(get_package_share_directory('hill_patrol_sim')) / 'worlds' / f'{WORLDS[world]}_truth_zones.yaml')
    nav2 = Path(get_package_share_directory('nav2_bringup')) / 'launch/navigation_launch.py'
    # The global costmap is static: size it to the global elevation map (T22).
    meta = yaml.safe_load((map_dir / 'elevation/elevation.yaml').read_text(encoding='utf-8'))
    res = meta['resolution']
    params = yaml.safe_load((config / 'nav2_sim.yaml').read_text(encoding='utf-8'))
    global_params = params['global_costmap']['global_costmap']['ros__parameters']
    global_params.update(width=int(meta['cols'] * res) + 1, height=int(meta['rows'] * res) + 1,
                         origin_x=meta['origin'][0] - res / 2, origin_y=meta['origin'][1] - res / 2)
    params_file = Path('/tmp') / f'hill_patrol_nav2_{WORLDS[world]}.yaml'
    params_file.write_text(yaml.safe_dump(params, sort_keys=False), encoding='utf-8')
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(bringup / 'launch/localization.launch.py')),
            launch_arguments={'world': world, 'map_dir': str(map_dir),
                              'x': LaunchConfiguration('x').perform(context),
                              'y': LaunchConfiguration('y').perform(context),
                              'yaw': LaunchConfiguration('yaw').perform(context)}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(bringup / 'launch/elevation.launch.py'))),
        Node(package='traversability_maps', executable='traversability_maps', name='traversability_maps',
             parameters=[{'use_sim_time': True, 'zones_file': zones,
                          'elevation_yaml': str(map_dir / 'elevation/elevation.yaml')}],
             output='screen'),
        Node(package='gait_manager', executable='gait_node', name='gait_manager',
             parameters=[{'use_sim_time': True, 'zones_file': zones,
                          'safety_path': str(config / 'm20_safety.yaml')}],
             output='screen'),
        # Nav2 waits for localization TF before activating its lifecycle nodes.
        TimerAction(period=25.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(nav2)),
            launch_arguments={'use_sim_time': 'true', 'params_file': str(params_file),
                              'autostart': 'true'}.items())]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='small', choices=list(WORLDS)),
        DeclareLaunchArgument('map_dir', default_value='~/hill_patrol_ws/maps/data/hill_world_small'),
        DeclareLaunchArgument('zones_file', default_value=''),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        OpaqueFunction(function=_setup),
    ])

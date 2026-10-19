"""T30 data recording: ros2 bag record of config/record_topics.yaml.

ros2 launch hill_patrol_bringup record.launch.py output:=~/bags/run1 [use_sim_time:=true]
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import yaml


def _setup(context):
    config = Path(get_package_share_directory('hill_patrol_bringup')) / 'config/record_topics.yaml'
    topics = yaml.safe_load(config.read_text(encoding='utf-8'))['topics']
    command = ['ros2', 'bag', 'record', '-o',
               os.path.expanduser(LaunchConfiguration('output').perform(context))]
    if LaunchConfiguration('use_sim_time').perform(context) == 'true':
        command.append('--use-sim-time')
    return [ExecuteProcess(cmd=command + topics, output='screen', name='bag_record')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('output'),
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        OpaqueFunction(function=_setup),
    ])

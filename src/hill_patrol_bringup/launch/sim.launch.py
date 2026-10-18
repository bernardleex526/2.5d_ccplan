"""T16 simulation bringup: Gazebo hill world + M20 proxy + mock M20 + m20_bridge.

ros2 launch hill_patrol_bringup sim.launch.py world:=small|full mode:=teleop|navigation
    headless:=true|false software_gl:=auto|true|false teleop:=false|true

ROS topics: /points_raw, /imu, /d435i/depth/points, /gnss/fix, /clock,
/sim/ground_truth/pose (world frame). Command path: /cmd_vel -> m20_bridge ->
UDP -> mock_m20_server -> /sim/m20_axis -> follower plugin, identical to the
hardware protocol path. software_gl=auto selects Mesa llvmpipe under WSL2, whose
D3D12 OpenGL (4.2) is below the Ogre2 requirement.
"""
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

WORLDS = {'small': 'hill_world_small', 'full': 'hill_world'}
BRIDGE_TOPICS = [
    '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
    '/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
    '/sim/ground_truth/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose',
    '/mid360/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
    '/d435i/depth/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
    '/gnss/fix@sensor_msgs/msg/NavSatFix[ignition.msgs.NavSat',
    '/sim/m20_axis@geometry_msgs/msg/Twist]ignition.msgs.Twist',
    '/sim/m20_gait@std_msgs/msg/Int32]ignition.msgs.Int32',
]


def compose(world_name, output, x=0.0, y=0.0, yaw=0.0):
    """Generated hill world + M20 proxy + follower plugin, with absolute mesh URIs."""
    share = Path(get_package_share_directory('hill_patrol_sim'))
    config = Path(get_package_share_directory('hill_patrol_bringup')) / 'config'
    world_file = share / 'worlds' / f'{world_name}.sdf'
    tree = ET.parse(world_file)
    world = tree.getroot().find('world')
    for uri in world.iter('uri'):
        uri.text = str((world_file.parent / uri.text).resolve())
    model = copy.deepcopy(ET.parse(share / 'models/m20_proxy/model.sdf').getroot().find('model'))
    ET.SubElement(model, 'pose').text = f'{x} {y} 0 0 0 {yaw}'
    plugin = ET.SubElement(model, 'plugin', name='hill_patrol_sim::KinematicFollower',
                           filename=str(Path(get_package_prefix('hill_patrol_sim'))
                                        / 'lib/libkinematic_follower.so'))
    for key, value in {'height_grid': world_file.parent / f'{world_name}_body_height.csv',
                       'limits_path': config / 'm20_limits.yaml',
                       'sensors_path': share / 'config/sensors.yaml'}.items():
        ET.SubElement(plugin, key).text = str(value)
    world.append(model)
    tree.write(output, encoding='utf-8', xml_declaration=True)
    return output


def _is_wsl():
    try:
        return 'microsoft' in Path('/proc/version').read_text().lower()
    except OSError:
        return False


def static_frames(sensors):
    """base_link -> sensor frames from sensors.yaml mounts (x y z roll pitch yaw)."""
    frames = [('mid360', 'mid360_frame'), ('d435i', 'd435i_frame'), ('gnss', 'gnss_frame')]
    nodes = []
    for section, frame in frames:
        x, y, z, roll, pitch, yaw = sensors[section]['mount']['value']
        nodes.append(Node(package='tf2_ros', executable='static_transform_publisher',
                          name=f'tf_{frame}', parameters=[{'use_sim_time': True}],
                          arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                                     '--roll', str(roll), '--pitch', str(pitch), '--yaw', str(yaw),
                                     '--frame-id', 'base_link', '--child-frame-id', frame]))
    nodes.append(Node(package='tf2_ros', executable='static_transform_publisher',
                      name='tf_d435i_optical', parameters=[{'use_sim_time': True}],
                      arguments=['--roll', str(-math.pi / 2), '--yaw', str(-math.pi / 2),
                                 '--frame-id', 'd435i_frame',
                                 '--child-frame-id', 'd435i_optical_frame']))
    return nodes


def _setup(context):
    world = WORLDS[LaunchConfiguration('world').perform(context)]
    mode = LaunchConfiguration('mode').perform(context)
    headless = LaunchConfiguration('headless').perform(context) == 'true'
    software = LaunchConfiguration('software_gl').perform(context)
    port = LaunchConfiguration('robot_port').perform(context)
    sim_share = Path(get_package_share_directory('hill_patrol_sim'))
    config = Path(get_package_share_directory('hill_patrol_bringup')) / 'config'
    composed = compose(world, f'/tmp/hill_patrol_{world}.sdf',
                       float(LaunchConfiguration('x').perform(context)),
                       float(LaunchConfiguration('y').perform(context)),
                       float(LaunchConfiguration('yaw').perform(context)))
    actions = []
    if software == 'true' or (software == 'auto' and _is_wsl()):
        actions.append(SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'))
    gz = ['ign', 'gazebo', '-r', str(composed)]
    if headless:
        gz[2:2] = ['-s', '--headless-rendering']
    actions.append(ExecuteProcess(cmd=gz, output='screen', name='gazebo'))
    actions.append(Node(package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge',
                        arguments=BRIDGE_TOPICS, parameters=[{'use_sim_time': True}],
                        remappings=[('/mid360/points', '/points_raw')], output='screen'))
    sensors = yaml.safe_load((sim_share / 'config/sensors.yaml').read_text(encoding='utf-8'))
    actions += static_frames(sensors)
    actions.append(Node(package='m20_bridge', executable='mock_m20_server', name='mock_m20',
                        arguments=['--port', port, '--heartbeat-timeout', '2', '--ros'],
                        output='screen'))
    actions.append(Node(package='m20_bridge', executable='bridge_node', name='m20_bridge',
                        parameters=[{'robot_ip': '127.0.0.1', 'robot_port': int(port), 'mode': mode,
                                     'safety_path': str(config / 'm20_safety.yaml'),
                                     'limits_path': str(config / 'm20_limits.yaml')}],
                        output='screen'))
    if LaunchConfiguration('teleop').perform(context) == 'true':
        actions.append(Node(package='teleop_twist_keyboard', executable='teleop_twist_keyboard',
                            prefix='xterm -e', output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='small', choices=list(WORLDS)),
        DeclareLaunchArgument('mode', default_value='teleop', choices=['teleop', 'navigation']),
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('software_gl', default_value='auto', choices=['auto', 'true', 'false']),
        DeclareLaunchArgument('teleop', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('robot_port', default_value='30000'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        OpaqueFunction(function=_setup),
    ])

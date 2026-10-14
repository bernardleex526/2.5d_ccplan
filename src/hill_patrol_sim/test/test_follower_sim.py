"""Installed Fortress follower acceptance on the accepted nonuniform small grid.

T15 isolates the follower by omitting rendering sensors in the temporary composed
model (the accepted model file is untouched). Box centre height is support height
plus the rotated box's vertical half-extent. This is a kinematic envelope, not leg
contact mechanics. IMU is an ideal colocated base_link simulation choice at 100 Hz,
not calibrated hardware. Errors and frequencies are report-only diagnostics.
"""

import bisect
import copy
import csv
import json
import math
import os
from pathlib import Path
import time
import uuid
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import launch_pytest
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
import yaml

from hill_patrol_testing import thresholds

IMU_HZ = 100.0  # Explicit ideal SIM sampling choice, no sensors.yaml calibration claim.
ALT_GRAVITY = 4.7  # Alternate scene gravity used to expose any hardcoded value.


@pytest.fixture
def resources(tmp_path, monkeypatch):
    domain = str(100 + os.getpid() % 100)
    partition = 't15_' + uuid.uuid4().hex
    monkeypatch.setenv('ROS_DOMAIN_ID', domain)
    monkeypatch.setenv('IGN_IP', '127.0.0.1')
    monkeypatch.setenv('IGN_PARTITION', partition)
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    sim = Path(get_package_share_directory('hill_patrol_sim'))
    profile = str(sim / 'config/fastdds_large_sensor.xml')
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', profile)
    env = {name: os.environ[name] for name in (
        'ROS_DOMAIN_ID', 'IGN_IP', 'IGN_PARTITION', 'RMW_IMPLEMENTATION',
        'FASTRTPS_DEFAULT_PROFILES_FILE')}
    yield {'tmp': tmp_path, 'sim': sim, 'env': env}


@launch_pytest.fixture(params=['ramp', 'stairs', 'turn', 'gravity'])
def follower_launch(request, resources):
    sim = resources['sim']
    scenario = request.param
    world_file = sim / 'worlds/hill_world_small.sdf'
    tree = ET.parse(world_file)
    world = tree.getroot().find('world')
    if scenario == 'gravity':
        # Lifecycle probe: a scene gravity that differs from the usual 9.8 makes
        # any hardcoded or fallback gravity value directly observable in the IMU.
        world.find('gravity').text = f'0 0 {-ALT_GRAVITY}'
    for uri in world.iter('uri'):
        uri.text = str((world_file.parent / uri.text).resolve())
    model = copy.deepcopy(ET.parse(sim / 'models/m20_proxy/model.sdf').getroot().find('model'))
    for link in model.findall('link'):
        for sensor in link.findall('sensor'):
            link.remove(sensor)
    start = {'ramp': 3.2, 'stairs': 9.2, 'turn': 4.0, 'gravity': 3.2}[scenario]
    ET.SubElement(model, 'pose').text = f'{start} 0 0 0 0 0'
    library = Path(get_package_prefix('hill_patrol_sim')) / 'lib/libkinematic_follower.so'
    assert library.is_file(), 'Acceptance must load the installed plugin'
    plugin = ET.SubElement(model, 'plugin', filename=str(library),
                           name='hill_patrol_sim::KinematicFollower')
    limits = Path(get_package_share_directory('hill_patrol_bringup')) / 'config/m20_limits.yaml'
    for key, value in {'height_grid': sim / 'worlds/hill_world_small_body_height.csv',
                       'limits_path': limits, 'imu_hz': IMU_HZ}.items():
        ET.SubElement(plugin, key).text = str(value)
    world.append(model)
    composed = resources['tmp'] / 'follower.sdf'
    tree.write(composed, encoding='utf-8', xml_declaration=True)
    gz = ExecuteProcess(cmd=['ign', 'gazebo', '-s', '-r', str(composed)],
                        additional_env=resources['env'], output='screen')
    bridge = ExecuteProcess(cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        '/sim/m20_axis@geometry_msgs/msg/Twist]ignition.msgs.Twist',
        '/sim/m20_gait@std_msgs/msg/Int32]ignition.msgs.Int32',
        '/sim/ground_truth/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose',
        '/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
        '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        '/world/hill_world_small/dynamic_pose/info@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V'],
        additional_env=resources['env'], output='screen')
    resources.update(scenario=scenario, limits=yaml.safe_load(limits.read_text()),
                     gravity=ALT_GRAVITY if scenario == 'gravity' else 9.8,
                     size=[float(v) for v in model.find('link/collision/geometry/box/size').text.split()])
    return LaunchDescription([gz, bridge]), resources


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def body_z_axis(q):
    """World +z expressed in the body frame: the third row of R_body->world.

    Rotating a world vector into the body frame is R^T v, so component i of the
    world vertical is R_3i: the third row, not the third column. At rest an
    ideal colocated accelerometer reads |g| times exactly this direction, and a
    world-z angular rate appears as the same direction scaled by the rate.
    """
    return rotation(q)[2]


def grid_sampler(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    xs = sorted({float(row['x']) for row in rows})
    ys = sorted({float(row['y']) for row in rows})
    values = {(float(row['x']), float(row['y'])): float(row['z']) for row in rows}
    def sample(x, y):
        i = min(len(xs)-2, max(0, bisect.bisect_right(xs, x)-1))
        j = min(len(ys)-2, max(0, bisect.bisect_right(ys, y)-1))
        tx, ty = (x-xs[i])/(xs[i+1]-xs[i]), (y-ys[j])/(ys[j+1]-ys[j])
        return sum(values[(xs[i+a], ys[j+b])] * (tx if a else 1-tx) *
                   (ty if b else 1-ty) for a in (0, 1) for b in (0, 1))
    return sample


@pytest.mark.sim
@pytest.mark.launch(fixture=follower_launch)
def test_installed_follower(follower_launch):
    _, data = follower_launch
    context = Context()
    rclpy.init(args=[], context=context)
    node = rclpy.create_node('follower_acceptance', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    poses, imus, physics_poses = [], [], []
    node.create_subscription(PoseStamped, '/sim/ground_truth/pose', poses.append, 1000)
    node.create_subscription(Imu, '/imu', imus.append, qos_profile_sensor_data)
    node.create_subscription(PoseArray, '/world/hill_world_small/dynamic_pose/info',
                             physics_poses.append, 100)
    axis = node.create_publisher(Twist, '/sim/m20_axis', 10)
    gait = node.create_publisher(Int32, '/sim/m20_gait', 10)
    def stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    def observe(predicate, command=None, timeout=20):
        deadline, next_send = time.monotonic()+timeout, 0
        while time.monotonic() < deadline:
            if command is not None and time.monotonic() >= next_send:
                axis.publish(command)
                next_send = time.monotonic()+0.05
            executor.spin_once(timeout_sec=0.01)
            if predicate():
                return
        raise AssertionError('Real follower observation deadline expired')
    try:
        observe(lambda: len(poses)>100 and len(imus)>20)
        stationary = imus[-20:]
        magnitudes = [math.sqrt(sum(getattr(m.linear_acceleration, k)**2
                                    for k in ('x', 'y', 'z'))) for m in stationary]
        gravity_error = max(abs(value-data['gravity']) for value in magnitudes)
        assert all(m.header.frame_id == 'base_link' for m in stationary)
        assert math.isfinite(gravity_error)
        # Lifecycle: the plugin takes gravity from the world ECS at first
        # PreUpdate, so at rest an ideal colocated accelerometer must read the
        # scene gravity vector along body +z. A hardcoded or fallback gravity
        # (e.g. 9.8 while the scene is 4.7) fails these two assertions.
        stationary_vector_error = max(math.sqrt(sum((getattr(m.linear_acceleration, k)-
            data['gravity']*body_z_axis(m.orientation)[i])**2
            for i, k in enumerate(('x', 'y', 'z')))) for m in stationary)
        assert gravity_error < 1e-3, 'stationary magnitude must equal scene gravity'
        assert stationary_vector_error < 1e-3, 'stationary axis must be body +z'
        start_index = len(poses)
        command = Twist()
        command.linear.x = 0.5
        if data['scenario'] == 'turn':
            command.linear.y = 0.2
            command.angular.z = 0.5
        gait.publish(Int32(data=14))
        start_time = stamp(poses[-1])
        observe(lambda: poses and stamp(poses[-1])-start_time>2, command)
        axis.publish(Twist())
        movement = poses[start_index:]
        assert movement[-1].pose.position.x > movement[0].pose.position.x
        sample = grid_sampler(data['sim'] / 'worlds/hill_world_small_body_height.csv')

        def surface_residual(message):
            p, q = message.pose.position, message.pose.orientation
            r = rotation(q)
            offset = sum(abs(r[2][i])*data['size'][i]/2 for i in range(3))
            assert all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            return abs(p.z - offset - sample(p.x, p.y))

        # Lifecycle: initialization completes before the first publication, so
        # every published truth pose, the stationary prefix included, already
        # lies on the supported envelope surface. A publication before readiness
        # would surface the raw model pose instead (z 0 at these starts).
        assert max(surface_residual(m) for m in poses) < 1e-6
        errors = [surface_residual(m) for m in movement]
        gyro_error = 0.0
        if data['scenario'] != 'turn':
            assert all(b.pose.position.z >= a.pose.position.z for a, b in zip(movement, movement[1:]))
        else:
            assert movement[-1].pose.position.y > movement[0].pose.position.y
            assert abs(movement[-1].pose.orientation.z) > abs(movement[0].pose.orientation.z)
            moving_imus = [m for m in imus if stamp(m) > start_time+0.5]
            assert moving_imus and any(abs(m.angular_velocity.z)>0 for m in moving_imus)
            expected_yaw_rate = command.angular.z*data['limits'][14]['yaw']
            gyro_error = max(math.sqrt(sum((getattr(m.angular_velocity, k)-
                expected_yaw_rate*body_z_axis(m.orientation)[i])**2
                for i, k in enumerate(('x', 'y', 'z')))) for m in moving_imus)
        # SceneBroadcaster observes actual ECM/physics independently of truth.
        observe(lambda: physics_poses and any(p.position.x > movement[0].pose.position.x
                                             for p in physics_poses[-1].poses))
        by_stamp = {(m.header.stamp.sec, m.header.stamp.nanosec): m for m in movement}
        orientation_errors = [max(abs(getattr(m.orientation, k)-
            getattr(by_stamp[(m.header.stamp.sec, m.header.stamp.nanosec)].pose.orientation, k))
            for k in ('x', 'y', 'z', 'w')) for m in imus
            if (m.header.stamp.sec, m.header.stamp.nanosec) in by_stamp]
        assert orientation_errors and all(error == 0 for error in orientation_errors)
        times = [stamp(m) for m in imus[-100:]]
        periods = [b-a for a, b in zip(times, times[1:])]
        assert periods and all(p>0 and math.isfinite(p) for p in periods)
        rate = (len(times)-1)/(times[-1]-times[0])
        ns_times = [m.header.stamp.sec*1000000000+m.header.stamp.nanosec for m in imus[-100:]]
        configured_ns = round(1000000000/IMU_HZ)
        assert all((b-a) % configured_ns == 0 for a, b in zip(ns_times, ns_times[1:]))
        assert all(math.isfinite(getattr(m.angular_velocity, k)) and
                   math.isfinite(getattr(m.linear_acceleration, k))
                   for m in imus for k in ('x', 'y', 'z'))
        thresholds.report('elev_err_m', max(errors))
        diagnostics = {'scenario': data['scenario'], 'support_error_m': max(errors),
                       'scene_gravity_m_s2': data['gravity'],
                       'imu_gravity_magnitude_m_s2': max(magnitudes),
                       'imu_gravity_err_m_s2': gravity_error, 'imu_rate_hz': rate,
                       'configured_imu_hz': IMU_HZ, 'imu_rate_err_hz': abs(rate-IMU_HZ),
                       'pose_count': len(movement), 'imu_count': len(imus),
                       'stationary_vector_err_m_s2': stationary_vector_error,
                       'turn_gyro_err_rad_s': gyro_error,
                       'imu_pose_orientation_error': max(orientation_errors),
                       'actual_ecm_pose_count': len(physics_poses),
                       'start': [movement[0].pose.position.x, movement[0].pose.position.y, movement[0].pose.position.z],
                       'end': [movement[-1].pose.position.x, movement[-1].pose.position.y, movement[-1].pose.position.z]}
        metrics = json.loads(thresholds.METRICS_PATH.read_text())
        metrics.setdefault('t15_diagnostics', {})[data['scenario']] = diagnostics
        thresholds.METRICS_PATH.write_text(json.dumps(metrics, indent=2)+'\n')
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        context.shutdown()

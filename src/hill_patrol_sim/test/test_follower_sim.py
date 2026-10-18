"""Installed Fortress follower acceptance on the generated small hill world.

The rendering sensors are removed from a temporary composed model so the
follower runs alone. Expectations are analytic consequences of the follower's
configured filters (sensors.yaml), not invented thresholds:
* height/attitude follow the support plane after the filter settles;
* IMU specific force is bounded by gravity plus the filters' peak response,
  so no millisecond-scale spikes appear at stair/landing kinks;
* stationary specific force equals the scene gravity (hardcoded value probe).
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
from geometry_msgs.msg import PoseStamped, Twist
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

ALT_GRAVITY = 4.7  # Scene gravity differing from 9.8 exposes any hardcoded value.
SCENARIOS = {'stairs': (6.5, 0.0), 'turn': (4.0, 0.0), 'gravity': (1.5, 0.0)}


@pytest.fixture
def resources(tmp_path, monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', str(100 + os.getpid() % 100))
    monkeypatch.setenv('IGN_IP', '127.0.0.1')
    monkeypatch.setenv('IGN_PARTITION', 'follower_' + uuid.uuid4().hex)
    env = {name: os.environ[name] for name in ('ROS_DOMAIN_ID', 'IGN_IP', 'IGN_PARTITION')}
    if 'RMW_IMPLEMENTATION' in os.environ:
        env['RMW_IMPLEMENTATION'] = os.environ['RMW_IMPLEMENTATION']
    yield {'tmp': tmp_path, 'sim': Path(get_package_share_directory('hill_patrol_sim')), 'env': env}


@launch_pytest.fixture(params=list(SCENARIOS))
def follower_launch(request, resources):
    sim, scenario = resources['sim'], request.param
    world_file = sim / 'worlds/hill_world_small.sdf'
    tree = ET.parse(world_file)
    world = tree.getroot().find('world')
    for plugin in world.findall('plugin'):
        if plugin.get('name').endswith(('Sensors', 'NavSat')):
            world.remove(plugin)
    if scenario == 'gravity':
        world.find('gravity').text = f'0 0 {-ALT_GRAVITY}'
    for uri in world.iter('uri'):
        uri.text = str((world_file.parent / uri.text).resolve())
    model = copy.deepcopy(ET.parse(sim / 'models/m20_proxy/model.sdf').getroot().find('model'))
    for link in model.findall('link'):
        for sensor in link.findall('sensor'):
            link.remove(sensor)
    x0, y0 = SCENARIOS[scenario]
    ET.SubElement(model, 'pose').text = f'{x0} {y0} 0 0 0 0'
    library = Path(get_package_prefix('hill_patrol_sim')) / 'lib/libkinematic_follower.so'
    assert library.is_file(), 'acceptance must load the installed plugin'
    plugin = ET.SubElement(model, 'plugin', filename=str(library),
                           name='hill_patrol_sim::KinematicFollower')
    limits = Path(__file__).resolve().parents[2] / 'hill_patrol_bringup/config/m20_limits.yaml'
    sensors_path = sim / 'config/sensors.yaml'
    for key, value in {'height_grid': world_file.parent / 'hill_world_small_body_height.csv',
                       'limits_path': limits, 'sensors_path': sensors_path}.items():
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
        '/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU'],
        additional_env=resources['env'], output='screen')
    sensors = yaml.safe_load(sensors_path.read_text(encoding='utf-8'))
    resources.update(scenario=scenario, limits=yaml.safe_load(limits.read_text()),
                     gravity=ALT_GRAVITY if scenario == 'gravity' else 9.8, sensors=sensors,
                     grid=world_file.parent / 'hill_world_small_body_height.csv')
    return LaunchDescription([gz, bridge]), resources


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


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
        return sum(values[(xs[i+a], ys[j+b])] * (tx if a else 1-tx) * (ty if b else 1-ty)
                   for a in (0, 1) for b in (0, 1))
    return sample


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


@pytest.mark.sim
@pytest.mark.launch(fixture=follower_launch)
def test_installed_follower(follower_launch):
    _, data = follower_launch
    sensors, scenario = data['sensors'], data['scenario']
    imu_hz = sensors['imu']['update_rate']['value']
    tau_v = sensors['follower']['velocity_time_constant_s']['value']
    half = [v / 2 for v in sensors['body']['size']['value']]
    context = Context()
    rclpy.init(args=[], context=context)
    node = rclpy.create_node('follower_acceptance', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    poses, imus = [], []
    node.create_subscription(PoseStamped, '/sim/ground_truth/pose', poses.append, 2000)
    node.create_subscription(Imu, '/imu', imus.append, qos_profile_sensor_data)
    axis = node.create_publisher(Twist, '/sim/m20_axis', 10)
    gait = node.create_publisher(Int32, '/sim/m20_gait', 10)

    def observe(predicate, command=None, timeout=60):
        deadline, next_send = time.monotonic()+timeout, 0
        while time.monotonic() < deadline:
            if command is not None and time.monotonic() >= next_send:
                axis.publish(command)
                next_send = time.monotonic()+0.05
            executor.spin_once(timeout_sec=0.01)
            if predicate():
                return
        raise AssertionError('real follower observation deadline expired')
    try:
        observe(lambda: len(poses) > 200 and len(imus) > 40)
        stationary = imus[-20:]
        magnitudes = [math.sqrt(sum(getattr(m.linear_acceleration, k)**2 for k in 'xyz'))
                      for m in stationary]
        gravity_error = max(abs(v-data['gravity']) for v in magnitudes)
        assert all(m.header.frame_id == 'mid360_frame' for m in stationary)
        assert gravity_error < 1e-6, 'stationary specific force must equal scene gravity'
        command = Twist()
        command.linear.x = 1.0  # full-scale ratio: worst-case filter response
        if scenario == 'turn':
            command.linear.y = 0.4
            command.angular.z = 0.3
        gait.publish(Int32(data=14 if scenario == 'stairs' else 1))
        start_index, start_time = len(poses), stamp(poses[-1])
        duration = 12.0 if scenario == 'stairs' else 4.0
        observe(lambda: poses and stamp(poses[-1])-start_time > duration, command)
        axis.publish(Twist())
        movement = poses[start_index:]
        moving_imus = [m for m in imus if stamp(m) > start_time]
        assert movement[-1].pose.position.x > movement[0].pose.position.x + 1

        # Support tracking after the height filter settles (5 tau), on the support plane.
        sample = grid_sampler(data['grid'])
        settled = [m for m in movement if stamp(m) > start_time + 1.0]
        errors = []
        for m in settled:
            p, r = m.pose.position, rotation(m.pose.orientation)
            yaw = math.atan2(r[1][0], r[0][0])
            c, s = math.cos(yaw), math.sin(yaw)
            corners = [sample(p.x+c*bx-s*by, p.y+s*bx+c*by)
                       for bx, by in ((half[0], half[1]), (half[0], -half[1]),
                                      (-half[0], half[1]), (-half[0], -half[1]))]
            offset = sum(abs(r[2][i])*half[i] for i in range(3))
            errors.append(abs(p.z - offset - sum(corners)/4))
        support_error = max(errors)

        # IMU smoothness: specific force bounded by gravity + filter-limited dynamics.
        peak = max(math.sqrt(sum(getattr(m.linear_acceleration, k)**2 for k in 'xyz'))
                   for m in moving_imus)
        gyro = max(math.sqrt(sum(getattr(m.angular_velocity, k)**2 for k in 'xyz'))
                   for m in moving_imus)
        limits = data['limits'][14 if scenario == 'stairs' else 1]
        planar = math.hypot(limits['x'], limits['y']) / (math.e * tau_v)
        # Generous analytic envelope: planar lag peak, centripetal term, 3 g vertical
        # budget for filtered stair kinks; a 1 ms second difference gave >30 g.
        bound = data['gravity'] + planar + limits['x']*limits['yaw'] + 3*9.8
        assert peak < bound, f'IMU spike {peak:.1f} m/s2 exceeds analytic envelope {bound:.1f}'
        if scenario == 'turn':
            assert any(abs(m.angular_velocity.z) > 0.5*command.angular.z*limits['yaw']
                       for m in moving_imus)
        if scenario != 'turn':
            assert movement[-1].pose.position.z > movement[0].pose.position.z

        times = [stamp(m) for m in imus[-100:]]
        rate = (len(times)-1)/(times[-1]-times[0])
        ns = [m.header.stamp.sec*1000000000+m.header.stamp.nanosec for m in imus[-100:]]
        configured = round(1e9/imu_hz)
        assert all((b-a) % configured == 0 and b > a for a, b in zip(ns, ns[1:]))
        thresholds.report('elev_err_m', support_error)
        diagnostics = {'scenario': scenario, 'support_error_m': support_error,
                       'imu_peak_specific_force_m_s2': peak, 'imu_bound_m_s2': bound,
                       'imu_peak_gyro_rad_s': gyro, 'imu_rate_hz': rate,
                       'imu_gravity_err_m_s2': gravity_error,
                       'start': [movement[0].pose.position.x, movement[0].pose.position.z],
                       'end': [movement[-1].pose.position.x, movement[-1].pose.position.z]}
        metrics = json.loads(thresholds.METRICS_PATH.read_text())
        metrics.setdefault('follower_diagnostics', {})[scenario] = diagnostics
        thresholds.METRICS_PATH.write_text(json.dumps(metrics, indent=2)+'\n')
        print('FOLLOWER_DIAGNOSTICS', json.dumps(diagnostics))
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        context.shutdown()

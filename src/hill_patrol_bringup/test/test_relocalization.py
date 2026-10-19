"""T23: relocalize on the saved small-world map from 3 different start poses on the loop.

Each start publishes /initialpose (map frame) computed from ground truth via the
known map<-world offset (the map origin is the lidar pose at the first mapping
keyframe, i.e. the route start). Pass: /localization/ok becomes true and the
pose stays fresh; position/yaw errors vs truth are report-only.
Requires the map produced by test_mapping_loop (HILL_PATROL_MAPS).
"""
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import numpy as np
import pytest
import yaml

from ament_index_python.packages import get_package_share_directory
from hill_patrol_testing import eval_traj, thresholds

MAPS = Path(os.environ.get('HILL_PATROL_MAPS', Path.home() / 'hill_patrol_ws/maps'))
MAP_DIR = MAPS / 'data/hill_world_small'
# Start poses on the small loop (x, y, yaw): start pad, top of ramp, return lane.
STARTS = {'start_pad': (1.0, 0.0, 0.0), 'ramp_top': (8.4, 0.0, 0.0), 'return_lane': (20.0, -14.84, math.pi)}


def sensors():
    path = Path(get_package_share_directory('hill_patrol_sim')) / 'config/sensors.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def support_height(x, y):
    from hill_patrol_testing.eval_elevation import TruthGrid
    grid = TruthGrid(Path(get_package_share_directory('hill_patrol_sim')) / 'worlds/hill_world_small_body_height.csv')
    return grid.sample(x, y)


def ros(env, *args, timeout=30):
    return subprocess.run(['ros2', *args], env=env, capture_output=True, text=True, timeout=timeout)


def load_csv(path):
    return [list(map(float, l.split(','))) for l in Path(path).read_text().splitlines() if l]


@pytest.mark.sim
@pytest.mark.parametrize('start', list(STARTS))
def test_relocalization(start, tmp_path):
    if not (MAP_DIR / 'global.pcd').is_file():
        pytest.skip('run test_mapping_loop first to produce the map')
    x, y, yaw = STARTS[start]
    half_height = sensors()['body']['size']['value'][2] / 2
    lidar = sensors()['mid360']['mount']['value']
    # World -> map: the map origin is the lidar at the route start (0,0, half_height on flat pad).
    origin = (lidar[0], lidar[1], support_height(0, 0) + half_height + lidar[2])
    env = dict(os.environ, ROS_DOMAIN_ID=str(150 + os.getpid() % 50),
               IGN_PARTITION='t23_' + uuid.uuid4().hex)
    log = open(tmp_path / 'launch.log', 'w')
    launch = subprocess.Popen(['ros2', 'launch', 'hill_patrol_bringup', 'localization.launch.py',
                               'world:=small', f'map_dir:={MAP_DIR}', f'x:={x}', f'y:={y}', f'yaw:={yaw}'],
                              env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    recorder = subprocess.Popen(['ros2', 'run', 'hill_patrol_bringup', 'record_trajectories.py',
                                 str(tmp_path / 'est.csv'), str(tmp_path / 'truth.csv'),
                                 '/localization/pose', 'base'],
                                env=env, start_new_session=True)
    try:
        time.sleep(20)  # Gazebo start + IMU initialisation while stationary
        if start != 'start_pad':
            c, s = math.cos(yaw), math.sin(yaw)
            z = support_height(x, y) + half_height
            # base_link pose in map frame = world pose minus map origin (map axes = world axes).
            px, py, pz = x - origin[0], y - origin[1], z - origin[2]
            message = ('{header: {frame_id: map}, pose: {pose: {position: {x: %f, y: %f, z: %f}, '
                       'orientation: {z: %f, w: %f}}}}' % (px + c * lidar[0], py + s * lidar[0],
                                                           pz + lidar[2], math.sin(yaw / 2), math.cos(yaw / 2)))
            for _ in range(3):
                ros(env, 'topic', 'pub', '--once', '/initialpose',
                    'geometry_msgs/msg/PoseWithCovarianceStamped', message)
                time.sleep(2)
        deadline = time.monotonic() + 90
        ok = False
        while time.monotonic() < deadline and not ok:
            echo = ros(env, 'topic', 'echo', '--once', '/localization/ok', timeout=10)
            ok = 'data: true' in echo.stdout
            time.sleep(1)
        assert ok, '/localization/ok never became true'
        time.sleep(10)  # collect a stable window
        echo = ros(env, 'topic', 'echo', '--once', '/localization/ok', timeout=10)
        assert 'data: true' in echo.stdout, 'localization was lost after converging'
    finally:
        for process in (recorder, launch):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in (recorder, launch):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        log.close()
    est = np.array(load_csv(tmp_path / 'est.csv'))
    truth = np.array(load_csv(tmp_path / 'truth.csv'))
    est[:, 1:4] += origin  # map -> world (map axes are world axes at the route start)
    result = eval_traj.evaluate(est.tolist(), truth.tolist(), align='none', max_dt=0.05)
    thresholds.report_worst('loc_pos_err_m', result['ate_rmse_m'])
    thresholds.report_worst('loc_yaw_err_deg', result['yaw_rmse_deg'])
    print('RELOC_RESULT', start, result)

"""T26/T27: Nav2 single-goal navigation on the small world, start pad -> top of ramp.

The route avoids stair zones (plan T27). Uses the map from test_mapping_loop and the
global elevation map from global_elevation_builder (T22). Pass: NavigateToPose
SUCCEEDED and the true robot position is near the goal; no bridge fault.
The traversability layer loading at all is the T26 Humble check (plugin
nav2_gridmap_traversability_layer::TraversabilityLayer).
"""
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

import pytest
import yaml

from ament_index_python.packages import get_package_share_directory

MAPS = Path(os.environ.get('HILL_PATROL_MAPS', Path.home() / 'hill_patrol_ws/maps'))
MAP_DIR = MAPS / 'data/hill_world_small'
GOAL_WORLD = (8.4, 0.0)  # top of the 15 deg ramp, on the up_transition platform


@pytest.mark.sim
@pytest.mark.gpu
def test_single_goal(tmp_path):
    if not (MAP_DIR / 'elevation/elevation.yaml').is_file():
        pytest.skip('run test_mapping_loop and global_elevation_builder first')
    sensors = yaml.safe_load((Path(get_package_share_directory('hill_patrol_sim'))
                              / 'config/sensors.yaml').read_text(encoding='utf-8'))
    lidar_x = sensors['mid360']['mount']['value'][0]
    env = dict(os.environ, ROS_DOMAIN_ID=str(150 + os.getpid() % 50),
               IGN_PARTITION='t27_' + uuid.uuid4().hex)
    log_path = tmp_path / 'launch.log'
    log = open(log_path, 'w')
    launch = subprocess.Popen(['ros2', 'launch', 'hill_patrol_bringup', 'navigation.launch.py',
                               'world:=small', f'map_dir:={MAP_DIR}'],
                              env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        time.sleep(60)
        text = log_path.read_text(errors='ignore')
        # T26: the upstream Jazzy-targeted layer builds and loads on Humble in both costmaps.
        assert "TraversabilityLayer 'traversability_layer' subscribed to /traversability/local" in text
        assert "TraversabilityLayer 'static_traversability' subscribed to /traversability/global" in text
        # Map frame origin is the lidar at the route start; map axes equal world axes.
        goal = subprocess.run(['ros2', 'run', 'hill_patrol_bringup', 'send_goal.py',
                               str(GOAL_WORLD[0] - lidar_x), str(GOAL_WORLD[1]), '0.0', '600'],
                              env=env, capture_output=True, text=True, timeout=700)
        truth = subprocess.run(['ros2', 'topic', 'echo', '--once', '/sim/ground_truth/pose',
                                '--field', 'pose.position'], env=env, capture_output=True,
                               text=True, timeout=30).stdout
        fault = subprocess.run(['ros2', 'topic', 'echo', '--once', '/m20/fault', '--field', 'data'],
                               env=env, capture_output=True, text=True, timeout=30).stdout
    finally:
        os.killpg(launch.pid, signal.SIGINT)
        try:
            launch.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, signal.SIGKILL)
        log.close()
    assert 'NAV_STATUS SUCCEEDED' in goal.stdout, goal.stdout + goal.stderr
    x = float(re.search(r'x: (-?[\d.e-]+)', truth).group(1))
    y = float(re.search(r'y: (-?[\d.e-]+)', truth).group(1))
    print('SINGLE_GOAL truth', x, y)
    assert abs(x - GOAL_WORLD[0]) < 0.6 and abs(y - GOAL_WORLD[1]) < 0.6
    reason = [line.strip(" '") for line in fault.splitlines() if line.strip() and line.strip() != '---']
    assert not reason, f'bridge fault: {reason}'

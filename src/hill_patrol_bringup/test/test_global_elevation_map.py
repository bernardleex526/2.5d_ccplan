"""T22 on the saved T20 map: build the global 2.5D elevation map and compare with truth.

Requires the map from test_mapping_loop (HILL_PATROL_MAPS). The SLAM map origin is
the lidar at the route start; map axes equal world axes, so points are shifted by
the lidar mount position (world x, y) and its height above the start pad.
Reports elev_err_m / elev_coverage on the walking corridor (report-only until
checkpoint 2) and writes elevation.{npy,png,yaml} next to the map for Nav2 / web.
"""
import os
from pathlib import Path
import subprocess

from ament_index_python.packages import get_package_share_directory
import numpy as np
import pytest
import yaml

from hill_patrol_testing import eval_elevation, thresholds

MAPS = Path(os.environ.get('HILL_PATROL_MAPS', Path.home() / 'hill_patrol_ws/maps'))
MAP_DIR = MAPS / 'data/hill_world_small'
WORLD = 'hill_world_small'


@pytest.mark.sim
def test_global_elevation_from_saved_map():
    if not (MAP_DIR / 'global.pcd').is_file():
        pytest.skip('run test_mapping_loop first')
    share = Path(get_package_share_directory('hill_patrol_sim'))
    sensors = yaml.safe_load((share / 'config/sensors.yaml').read_text(encoding='utf-8'))
    world = yaml.safe_load((share / 'config' / f'{WORLD}.yaml').read_text(encoding='utf-8'))
    x, y, z = sensors['mid360']['mount']['value'][:3]
    lidar_height = sensors['body']['size']['value'][2] / 2 + z  # start pad is at height 0
    out = MAP_DIR / 'elevation'
    subprocess.run(['ros2', 'run', 'hill_patrol_testing', 'global_elevation_builder',
                    str(MAP_DIR / 'global.pcd'), str(out),
                    '--map-to-world', f'{x} {y} {lidar_height} 0'], check=True, timeout=300)
    meta = yaml.safe_load((out / 'elevation.yaml').read_text(encoding='utf-8'))
    grid = np.load(out / 'elevation.npy')
    truth = eval_elevation.TruthGrid(share / 'worlds' / f'{WORLD}_body_height.csv')
    half = world['step_width']['value'] / 2
    route = yaml.safe_load((share / 'worlds' / f'{WORLD}_route.yaml').read_text(encoding='utf-8'))
    route_end = max(w['x'] for w in route['waypoints'] if w['y'] == 0)  # end of the hill lane
    result = eval_elevation.evaluate(grid, meta['origin'], meta['resolution'], truth,
                                     region=(0.0, route_end, -half, half))
    print('GLOBAL_ELEVATION', result)
    assert result['cells'] > 0
    thresholds.report('elev_err_m', result['rmse_m'])
    thresholds.report('elev_coverage', result['coverage'])

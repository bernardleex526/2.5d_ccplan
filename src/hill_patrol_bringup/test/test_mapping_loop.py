"""T20: teleop-style mapping of the small hill loop with Lightning-LM; loop closure must trigger.

Loop evidence: Lightning-LM's glog line "optimize finished, loops: N" with N >= 1 and
the matching "loop outliers: k/N" with k < N (src/core/loop_closing/loop_closing.cc:361,374).
ATE is report-only (thresholds frozen at checkpoint 2).
"""
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid

from ament_index_python.packages import get_package_share_directory
import pytest

from hill_patrol_testing import eval_traj, thresholds

WORLD = 'hill_world_small'


def wait_for(predicate, timeout, interval=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def load_csv(path):
    rows = [list(map(float, line.split(','))) for line in Path(path).read_text().splitlines() if line]
    return rows


@pytest.mark.sim
def test_mapping_loop_closure(tmp_path):
    workdir = Path(os.environ.get('HILL_PATROL_MAPS', tmp_path / 'maps'))
    shutil.rmtree(workdir / 'data' / WORLD, ignore_errors=True)
    env = dict(os.environ, ROS_DOMAIN_ID=str(150 + os.getpid() % 50),
               IGN_PARTITION='t20_' + uuid.uuid4().hex)
    launch_log = open(tmp_path / 'launch.log', 'w')
    launch = subprocess.Popen(['ros2', 'launch', 'hill_patrol_bringup', 'mapping.launch.py',
                               'world:=small', f'workdir:={workdir}', 'drive:=true', 'laps:=1'],
                              env=env, stdout=launch_log, stderr=subprocess.STDOUT,
                              start_new_session=True)
    recorder = subprocess.Popen(['ros2', 'run', 'hill_patrol_bringup', 'record_trajectories.py',
                                 str(tmp_path / 'est.csv'), str(tmp_path / 'truth.csv')],
                                env=env, start_new_session=True)
    try:
        log = tmp_path / 'launch.log'
        assert wait_for(lambda: 'lap 1 complete' in log.read_text(errors='ignore'), 900), \
            'route driver did not complete the loop'
        save = subprocess.run(['ros2', 'service', 'call', '/lightning/save_map', 'lightning/srv/SaveMap',
                               f'{{map_id: {WORLD}}}'], env=env, capture_output=True, text=True,
                              timeout=120)
        assert 'response=0' in save.stdout, save.stdout + save.stderr
    finally:
        for process in (recorder, launch):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in (recorder, launch):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        launch_log.close()
    subprocess.run(['pkill', '-f', f'IGN_PARTITION={env["IGN_PARTITION"]}'], check=False)

    map_dir = workdir / 'data' / WORLD
    for name in ('global.pcd', 'index.txt', 'places.bin'):
        assert (map_dir / name).is_file(), f'missing {name}'
    glog = '\n'.join(p.read_text(errors='ignore') for p in workdir.glob('log/*/run_slam_online.INFO'))
    loops = [int(n) for n in re.findall(r'optimize finished, loops: (\d+)', glog)]
    outliers = [(int(a), int(b)) for a, b in re.findall(r'loop outliers: (\d+)/(\d+)', glog)]
    assert loops and max(loops) >= 1, 'no loop closure accepted'
    assert any(k < n for k, n in outliers), 'every loop edge was rejected as an outlier'

    result = eval_traj.evaluate(load_csv(tmp_path / 'est.csv'), load_csv(tmp_path / 'truth.csv'))
    thresholds.report('loc_pos_err_m', result['ate_rmse_m'])
    thresholds.report('loc_yaw_err_deg', result['yaw_rmse_deg'])
    print('MAPPING_RESULT', result, 'loops', max(loops))

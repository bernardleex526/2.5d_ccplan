"""T29: one patrol lap of the small loop through both stair zones; gait switching must be correct.

Pass (plan T29): gait rule violations = 0; each stair zone entry and exit logged one
switch (4 switches: 1->14, 14->1, 1->14, 14->1); /gait/state never FAULT; the lap
result reports success. Uses the map from test_mapping_loop and the T22 global
elevation map.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import uuid

import pytest

MAPS = Path(os.environ.get('HILL_PATROL_MAPS', Path.home() / 'hill_patrol_ws/maps'))
MAP_DIR = MAPS / 'data/hill_world_small'


def echo_stream(env, topic, sink, stop):
    process = subprocess.Popen(['ros2', 'topic', 'echo', topic, '--field', 'data'], env=env,
                               stdout=subprocess.PIPE, text=True, start_new_session=True)
    for line in process.stdout:
        line = line.strip()
        if line and line != '---':
            sink.append(line.strip("'"))
        if stop.is_set():
            break
    os.killpg(process.pid, signal.SIGINT)


@pytest.mark.sim
@pytest.mark.gpu
def test_gait_patrol(tmp_path):
    if not (MAP_DIR / 'elevation/elevation.yaml').is_file():
        pytest.skip('run test_mapping_loop and global_elevation_builder first')
    env = dict(os.environ, ROS_DOMAIN_ID=str(150 + os.getpid() % 50),
               IGN_PARTITION='t29_' + uuid.uuid4().hex)
    log = open(tmp_path / 'launch.log', 'w')
    launch = subprocess.Popen(['ros2', 'launch', 'hill_patrol_bringup', 'patrol.launch.py',
                               'world:=small', f'map_dir:={MAP_DIR}', f'patrol_dir:={tmp_path / "patrol"}',
                               'laps:=1'], env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)
    states, switches, results, violations = [], [], [], []
    stop = threading.Event()
    threads = [threading.Thread(target=echo_stream, args=(env, topic, sink, stop), daemon=True)
               for topic, sink in (('/gait/state', states), ('/gait_rule/switches', switches),
                                   ('/patrol/lap_result', results), ('/gait_rule/violations', violations))]
    try:
        time.sleep(60)
        for thread in threads:
            thread.start()
        started = subprocess.run(['ros2', 'service', 'call', '/patrol/start', 'std_srvs/srv/Trigger'],
                                 env=env, capture_output=True, text=True, timeout=60)
        assert 'success=True' in started.stdout, started.stdout
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline and not results:
            time.sleep(5)
    finally:
        stop.set()
        os.killpg(launch.pid, signal.SIGINT)
        try:
            launch.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, signal.SIGKILL)
        log.close()
    print('GAIT_SWITCHES', switches)
    print('LAP_RESULTS', results)
    print('VIOLATIONS', violations[-1:] if violations else None)
    assert results, 'patrol lap did not finish'
    assert 'FAULT' not in states, 'gait handshake reached FAULT'
    assert violations and int(violations[-1]) == 0, f'gait rule violations: {violations[-1:]}'
    assert [re.match(r'(\d+)->(\d+)', s).groups() for s in switches] == \
        [('1', '14'), ('14', '1'), ('1', '14'), ('14', '1')], switches
    lap = json.loads(results[-1])
    assert lap['success'], lap

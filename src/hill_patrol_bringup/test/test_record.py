"""T30: record ~5 s of the running simulation; the bag must contain the configured sensor topics."""
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / 'config/record_topics.yaml'
# Published by sim.launch.py + cloud_adapter alone (no localization / elevation / gait stack).
SIM_TOPICS = {'/points_lio', '/imu', '/d435i/depth/points', '/gnss/fix', '/tf_static',
              '/m20/gait', '/m20/link_ok', '/sim/ground_truth/pose'}


def test_record_topics_config_covers_sim_sensors_and_rtk():
    topics = set(yaml.safe_load(CONFIG.read_text(encoding='utf-8'))['topics'])
    assert SIM_TOPICS <= topics
    assert '/gnss/fix' in topics  # plan D19: RTK always recorded


def stop(process, timeout=30):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


@pytest.mark.sim
def test_record_bag_contains_sim_topics(tmp_path):
    env = dict(os.environ, ROS_DOMAIN_ID=str(150 + os.getpid() % 50),
               IGN_PARTITION='t30_' + uuid.uuid4().hex)
    log = open(tmp_path / 'launch.log', 'w')
    spawn = lambda cmd: subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,  # noqa: E731
                                         start_new_session=True)
    sim = spawn(['ros2', 'launch', 'hill_patrol_bringup', 'sim.launch.py', 'world:=small'])
    adapter = spawn(['ros2', 'run', 'hill_patrol_sim', 'cloud_adapter'])
    bag = tmp_path / 'bag'
    try:
        time.sleep(25)
        record = spawn(['ros2', 'launch', 'hill_patrol_bringup', 'record.launch.py', f'output:={bag}'])
        time.sleep(10)  # discovery + several seconds of data
        stop(record)
    finally:
        stop(adapter)
        stop(sim)
        log.close()
    info = subprocess.run(['ros2', 'bag', 'info', str(bag)], env=env, capture_output=True,
                          text=True, timeout=60).stdout
    recorded = {line.split('Topic:')[1].split('|')[0].strip()
                for line in info.splitlines() if 'Topic:' in line and 'Count: 0 ' not in line}
    missing = SIM_TOPICS - recorded
    assert not missing, f'missing topics: {missing}\n{info}'

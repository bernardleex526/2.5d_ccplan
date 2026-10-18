#!/usr/bin/env python3
"""Make map-frame stair zones and patrol waypoints from the generated world truth.

Stands in for the operator's annotation on the web page (plan T29): the truth
zones / route are in world coordinates, the patrol runs in the SLAM map frame
whose origin is the lidar at the mapping start (route start, flat pad), with
map axes equal to world axes. So map = world - (lidar mount x, y, start support
height + body half height + lidar mount z).

Usage: make_patrol_config.py <world_name> <out_dir>
Writes <out_dir>/stair_zones.yaml and <out_dir>/waypoints.yaml.
"""
import argparse
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('world')
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()
    share = Path(get_package_share_directory('hill_patrol_sim'))
    sensors = yaml.safe_load((share / 'config/sensors.yaml').read_text(encoding='utf-8'))
    dx, dy = sensors['mid360']['mount']['value'][:2]
    zones = yaml.safe_load((share / 'worlds' / f'{args.world}_truth_zones.yaml').read_text(encoding='utf-8'))
    for zone in zones['zones']:
        zone['polygon'] = [[x - dx, y - dy] for x, y in zone['polygon']]
    route = yaml.safe_load((share / 'worlds' / f'{args.world}_route.yaml').read_text(encoding='utf-8'))
    # Patrol goals only on flat ground outside stair zones: Nav2 then plans each stair
    # zone as part of one path and gait_manager switches gait at its boundary.
    from gait_manager.zones import Zones
    world_zones = Zones.load(share / 'worlds' / f'{args.world}_truth_zones.yaml')
    waypoints = []
    for w in route['waypoints'][1:]:  # route[0] is the start pose itself
        if world_zones.zone_at(w['x'], w['y']) is None:
            waypoints.append({'id': w['id'], 'x': w['x'] - dx, 'y': w['y'] - dy, 'yaw': w['yaw']})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'stair_zones.yaml').write_text(yaml.safe_dump(zones, sort_keys=False), encoding='utf-8')
    (args.out_dir / 'waypoints.yaml').write_text(yaml.safe_dump({'waypoints': waypoints}, sort_keys=False),
                                                 encoding='utf-8')
    print(f'{len(zones["zones"])} zones, {len(waypoints)} waypoints -> {args.out_dir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env bash

source /opt/ros/humble/setup.bash

_hill_patrol_ws="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${_hill_patrol_ws}/install/setup.bash" ]]; then
  source "${_hill_patrol_ws}/install/setup.bash"
fi
unset _hill_patrol_ws

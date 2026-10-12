#!/usr/bin/env bash
# T02: additive Ubuntu 22.04 / ROS Humble provisioning; never edit apt sources,
# drivers or /usr/local/cuda. The rollback journal lives in deps/versions.yaml.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTOR="$ROOT/scripts/check_env.py"
VERSIONS="$ROOT/deps/versions.yaml"
if [[ "${1:-}" == --rollback && $# == 1 ]]; then
  # Keep rollback isolated from collection; the journal is read-only evidence.
  exec python3 - "$ROOT/scripts" "$VERSIONS" <<'PY'
import subprocess
import sys

sys.path.insert(0, sys.argv[1])
from pathlib import Path
from check_env import apt_inventory, apt_lookup, apt_package_name, load, pip_inventory

journal = load(Path(sys.argv[2])).get("rollback_journal")
if not journal:
    raise SystemExit("No installation journal; refusing to remove packages")
apt_names = list(journal["apt_added"])
pip_names = list(journal["pip_added"])
apt, pip = apt_inventory(), pip_inventory()
if (any(apt_lookup(journal["apt_before"], n) is not None for n in apt_names) or
        set(pip_names) & set(journal.get("pip_before", journal["pip_initial"]))):
    raise SystemExit("Journal additions overlap pre-existing packages; refusing")
if any(apt_lookup(apt, n) != v for n, v in journal["apt_before"].items()):
    raise SystemExit("Pre-existing apt versions changed; refusing")
if any(pip.get(n) != v for n, v in journal["pip_initial"].items()):
    raise SystemExit("Pre-existing pip versions changed; refusing")
remaining_apt = [n for n in apt_names if apt_lookup(apt, n) is not None]
command = ["sudo", "-n", "apt-get", "remove", "--no-auto-remove", "-y"]
def simulate(names):
    plan = subprocess.run(command[:3] + ["-s"] + command[3:] + names,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(plan.stdout, end="", flush=True)
    removed = {apt_package_name(line.split()[1]) for line in plan.stdout.splitlines()
               if line.startswith("Remv ")}
    if plan.returncode or removed != {apt_package_name(n) for n in names}:
        raise SystemExit("Rollback removal set differs from journal additions; refusing")
    print(f"Simulation verified: {len(removed)} journaled additions, no pre-existing removals.", flush=True)
if apt_names:
    simulate(remaining_apt)
    if remaining_apt:
        subprocess.run(command + remaining_apt, check=True)
if pip_names:
    remaining_pip = [n for n in pip_names if n in pip_inventory()]
    if remaining_pip:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y"] + remaining_pip, check=True)
apt, pip = apt_inventory(), pip_inventory()
remaining_apt = [n for n in apt_names if apt_lookup(apt, n) is not None]
if apt_names:
    simulate(remaining_apt)
remaining_pip = [n for n in pip_names if n in pip]
changed = [n for n, v in journal["apt_before"].items() if apt_lookup(apt, n) != v]
changed += [n for n, v in journal["pip_initial"].items() if pip.get(n) != v]
print(f"Remaining journaled additions: apt={len(remaining_apt)}, pip={len(remaining_pip)}; pre-existing changes={changed}", flush=True)
if remaining_apt or remaining_pip or changed:
    raise SystemExit("Rollback incomplete; stop and inspect preserved journal")
print("Rollback complete; original collector output and journal preserved. No autoremove.")
PY
fi
if [[ $# != 0 ]]; then
  echo "Usage: bash scripts/setup_env.sh [--rollback]" >&2
  exit 2
fi
# Reject unsupported hosts before any system/package side effects.
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || {
  echo 'ERROR: native Ubuntu 22.04 required; no packages changed.' >&2; exit 1;
}
[[ "$(uname -m)" == x86_64 || "$(uname -m)" == aarch64 ]] || {
  echo 'ERROR: supported architectures are x86_64 and aarch64.' >&2; exit 1;
}
[[ -f /opt/ros/humble/setup.bash && -f "$ROOT/scripts/env.sh" ]] || {
  echo 'ERROR: existing ROS Humble and scripts/env.sh are required (T01).' >&2; exit 1;
}
python3 -c 'import sys, yaml; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
python3 -m pip --version
sudo -n true || { echo 'ERROR: noninteractive sudo unavailable; manual intervention required.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Discover an existing CUDA 12 toolkit; do not install a driver/toolkit or change
# the system default. The collector uses the same discovery routine.
CUDA_PATH="$(python3 "$COLLECTOR" --cuda-path)"
export CUDA_PATH
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version
APT_PACKAGES=(
  ros-humble-desktop ros-humble-ros-gz ignition-fortress
  ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-grid-map
  ros-humble-foxglove-bridge ros-humble-rosbridge-suite
  ros-humble-launch-pytest python3-pytest ros-humble-teleop-twist-keyboard
  ros-humble-image-transport ros-humble-pcl-ros ros-humble-cv-bridge
  ros-humble-tf-transformations ros-humble-point-cloud-transport
  pybind11-dev libboost-all-dev python3-scipy python3-shapely
  python3-ruamel.yaml python3-transforms3d python3-opencv
)
# The pinned ROS2 v2.0.0 Dockerfile/package.xml are authoritative; the root
# requirements.txt still contains ROS1/catkin-era pins. CuPy 14 requires NumPy2,
# incompatible with the upstream ROS2 numpy<2 constraint, so use the 13 series.
PIP_PACKAGES=('numpy<2' 'cupy-cuda12x>=13,<14' simple-parsing
  scipy transforms3d opencv-python scikit-learn)
python3 "$COLLECTOR" --write "$VERSIONS" --journal begin
finish_journal() {
  local status=$?
  trap - EXIT
  python3 "$COLLECTOR" --write "$VERSIONS" --journal finish || status=1
  if (( status != 0 )); then
    echo 'ERROR: provisioning failed; run bash scripts/setup_env.sh --rollback, then STOP.' >&2
  fi
  exit "$status"
}
trap finish_journal EXIT
sudo -n apt-get update
# Simulate first and reject ANY existing-package upgrade/removal. Missing or
# conflicting source packages require manual intervention, not source edits.
python3 "$COLLECTOR" --apt-plan "${APT_PACKAGES[@]}"
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-remove --no-upgrade "${APT_PACKAGES[@]}"
python3 "$COLLECTOR" --write "$VERSIONS" --journal pip-baseline
# Temporary constraints pin every existing distribution at its effective
# version, preventing pip from replacing packages that rollback cannot restore.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"; finish_journal' EXIT
python3 "$COLLECTOR" --constraints "$CONSTRAINTS"
# Existing upstream dependencies are left untouched, including their metadata;
# do not let a newer preinstalled OpenCV force an unrelated NumPy upgrade.
MISSING="$(python3 "$COLLECTOR" --missing-pip "${PIP_PACKAGES[@]}")"
if [[ -n "$MISSING" ]]; then
  mapfile -t TO_INSTALL <<< "$MISSING"
  python3 -m pip install --user --constraint "$CONSTRAINTS" "${TO_INSTALL[@]}"
fi
# Pinned upstream uses cu121; its CUDA12 wheel also runs with toolkit12.6 and
# the compatible NVIDIA driver. 2.3.1 avoids replacing distro SymPy1.9.
TORCH="$(python3 "$COLLECTOR" --missing-pip 'torch==2.3.1')"
if [[ -n "$TORCH" ]]; then
  python3 -m pip install --user --constraint "$CONSTRAINTS" --index-url https://download.pytorch.org/whl/cu121 "$TORCH"
fi
rm -f "$CONSTRAINTS"
trap finish_journal EXIT
printf 'Provisioning complete. CUDA12 selected at %s (system CUDA default unchanged).\n' "$CUDA_PATH"
printf 'Source scripts/env.sh, then run python3 scripts/check_env.py --write deps/versions.yaml.\n'

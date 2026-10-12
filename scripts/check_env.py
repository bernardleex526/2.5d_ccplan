#!/usr/bin/env python3
"""T02 real version inventory and additive-install journal (no source edits).

Pinned elevation_mapping_cupy v2.0.0 ROS2 package.xml and Dockerfile.x64
supply the dependency list, not the older ROS1 requirements.txt. --write
preserves the embedded rollback journal. Missing probes are explicit and make
normal collection fail; journal operations do not pretend acceptance passed.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGES = ("nav2_bringup", "ros_gz_bridge", "grid_map_ros",
                "foxglove_bridge", "rosbridge_server", "launch_pytest")
PYTHON_PACKAGES = ("numpy", "scipy", "shapely", "ruamel.yaml", "opencv-python",
                   "simple-parsing", "scikit-learn", "transforms3d", "cupy-cuda12x", "torch")
APT_PACKAGES = ("ros-humble-desktop", "ros-humble-ros-gz", "ignition-fortress",
                "ros-humble-navigation2", "ros-humble-nav2-bringup", "ros-humble-grid-map",
                "ros-humble-foxglove-bridge", "ros-humble-rosbridge-suite",
                "ros-humble-launch-pytest", "python3-pytest", "ros-humble-teleop-twist-keyboard",
                "ros-humble-image-transport", "ros-humble-pcl-ros", "ros-humble-cv-bridge",
                "ros-humble-tf-transformations", "ros-humble-point-cloud-transport",
                "pybind11-dev", "libboost-all-dev", "python3-scipy", "python3-shapely",
                "python3-ruamel.yaml", "python3-transforms3d", "python3-opencv")
CUPY_DOC = "https://docs.cupy.dev/en/stable/install.html"
ELEVATION_DOC = "src/third_party/elevation_mapping_cupy/docker/Dockerfile.x64:125-145,169-199@e311bfdbcd1f8ddb698b9295a7116e25475581ac"


def run(args):
    try:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"missing: {exc}"
    text = result.stdout.strip()
    if result.returncode:
        return False, f"missing: exit {result.returncode}: {result.stderr.strip() or text}"
    return bool(text), text or "missing: command produced no output"


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pip_inventory():
    # setdefault respects importlib's first/effective distribution when the user
    # site shadows a distro package; recording the last would pin the wrong one.
    result = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            result.setdefault(canonical(name), dist.version)
    return result


def apt_package_name(name):
    return name.split(":", 1)[0]


def apt_lookup(inventory, name, default=None):
    # Keep exact multi-arch identities/versions in the rollback journal. Only
    # unqualified requirements may resolve to an architecture-qualified key.
    if name in inventory:
        return inventory[name]
    if name == apt_package_name(name):
        return next((version for key, version in inventory.items()
                     if apt_package_name(key) == name), default)
    return default


def apt_inventory():
    ok, text = run(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Status}\n"])
    if not ok:
        raise RuntimeError(text)
    return {name: version for name, version, state in (line.split("\t") for line in text.splitlines())
            if state == "installed"}


def cuda_path():
    candidates = [os.environ.get("CUDA_PATH"), shutil.which("nvcc")]
    candidates += [str(p) for p in sorted(Path("/usr/local").glob("cuda-12*"), reverse=True)]
    for value in candidates:
        if not value:
            continue
        path = Path(value).resolve()
        if path.name == "nvcc":
            path = path.parent.parent
        ok, text = run([str(path / "bin/nvcc"), "--version"])
        if ok and re.search(r"release 12\.", text):
            return path
    raise RuntimeError("missing: existing CUDA12 toolkit required; install/configure manually (no driver/source changes)")


def load(path):
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid existing YAML mapping: {path}")
    return data


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # No second persistent journal or scratch file outside the allowed outputs.
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def update_additions(journal):
    apt = apt_inventory()
    pip = pip_inventory()
    journal["apt_added"] = {n: v for n, v in apt.items() if apt_lookup(journal["apt_before"], n) is None}
    baseline = journal.get("pip_before", journal["pip_initial"])
    journal["pip_added"] = {n: v for n, v in pip.items() if n not in baseline}
    apt_changed = [n for n, v in journal["apt_before"].items() if apt_lookup(apt, n) != v]
    pip_changed = [n for n, v in journal["pip_initial"].items() if pip.get(n) != v]
    journal["preexisting_changes"] = apt_changed + pip_changed


def rollback(path, data):
    journal = data.get("rollback_journal")
    if not journal:
        raise RuntimeError("No installation journal; refusing to remove packages")
    update_additions(journal)
    save(path, data)
    apt_names = list(journal["apt_added"])
    pip_names = list(journal["pip_added"])
    if apt_names:
        command = ["sudo", "-n", "apt-get", "remove", "--no-auto-remove", "-y"] + apt_names
        plan = subprocess.run(command[:3] + ["--simulate"] + command[3:],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(plan.stdout, end="", flush=True)
        removed = {apt_package_name(line.split()[1]) for line in plan.stdout.splitlines() if line.startswith("Remv ")}
        if plan.returncode or removed - {apt_package_name(n) for n in apt_names}:
            raise RuntimeError("Rollback would remove pre-existing apt packages; refusing; manual intervention required")
        subprocess.run(command, check=True)
    # pip additions were measured after apt, so apt-owned Python packages cannot
    # accidentally be uninstalled by pip. Never use apt autoremove.
    if pip_names:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y"] + pip_names, check=True)
    current_apt, current_pip = apt_inventory(), pip_inventory()
    remaining = [n for n in apt_names if apt_lookup(current_apt, n) is not None] + [n for n in pip_names if n in current_pip]
    journal["rollback_status"] = "pass" if not remaining else "failed: " + ", ".join(remaining)
    journal["rolled_back_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(path, data)
    if remaining:
        raise RuntimeError(journal["rollback_status"])
    print("Rollback complete: only journaled additions removed; no autoremove.")


def journal_action(action, path, data):
    if action == "rollback":
        rollback(path, data)
        return
    if action == "begin":
        existing = data.get("rollback_journal")
        if existing and existing.get("rollback_status") != "pass":
            # Keep the original baseline across repeated/idempotent setup runs.
            update_additions(existing)
        else:
            data["rollback_journal"] = {
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "apt_before": apt_inventory(), "pip_initial": pip_inventory(),
                "apt_added": {}, "pip_added": {}, "preexisting_changes": [],
                "policy": "remove additions only; never autoremove or replace existing versions"}
    else:
        journal = data["rollback_journal"]
        if action == "pip-baseline" and "pip_before" not in journal:
            journal["pip_before"] = pip_inventory()
        update_additions(journal)
        journal["last_recorded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(path, data)
    changes = data["rollback_journal"]["preexisting_changes"]
    if changes:
        raise RuntimeError("Pre-existing versions changed; journal saved; manual intervention required: " + ", ".join(changes))
    print(f"Journal {action}: {path}")


def apt_plan(packages):
    command = ["apt-get", "--simulate", "install", "--no-remove", "--no-upgrade"] + packages
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    before = apt_inventory()
    changes = []
    for line in result.stdout.splitlines():
        if line.startswith("Remv ") or (line.startswith("Inst ") and apt_lookup(before, apt_package_name(line.split()[1])) is not None):
            changes.append(line)
    if result.returncode or changes:
        raise RuntimeError("apt plan missing/conflicting packages or modifies existing packages; manual intervention required: " + "; ".join(changes))


def architecture(value, source):
    return {"aarch64_available": value, "aarch64_provenance": source}


def apt_arch(name):
    name = apt_package_name(name)
    ok, text = run(["apt-cache", "show", name + ":arm64"])
    if ok and re.search(r"^Architecture: (arm64|all)$", text, re.M):
        return architecture(True, "configured apt arm64 package metadata: " + name)
    ok, text = run(["apt-cache", "show", name])
    if ok and "Architecture: all" in text:
        return architecture(True, "configured apt package Architecture: all: " + name)
    return architecture("待确认", "No version-specific arm64 metadata in configured apt indexes: " + name + "; Orin/JetPack validation required")


def pip_arch(name, version):
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            files = json.load(response)["urls"]
        names = [item["filename"] for item in files]
        if any(n.endswith(".whl") and ("aarch64" in n or n.endswith("-none-any.whl")) for n in names):
            return architecture(True, url + " (published aarch64 or platform-independent wheel; JetPack runtime not certified)")
        return architecture("待确认", url + " (no matching arm64/universal wheel evidence; source build not verified)")
    except (OSError, ValueError, KeyError) as exc:
        return architecture("待确认", f"{url}: metadata unavailable: {exc}; JetPack validation required")


def collect(data):
    errors = []
    def probe(name, args):
        ok, text = run(args)
        if not ok:
            errors.append(name + ": " + text)
        return text

    apt = apt_inventory()
    pip = pip_inventory()
    record = {"collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "dependency_authority": ELEVATION_DOC,
              "architecture_policy": "true means published architecture support, not a completed JetPack/Orin runtime qualification; 待确认 is not a fabricated success"}
    system_arch = architecture(True, "Ubuntu 22.04 arm64 release: https://releases.ubuntu.com/22.04/ and ROS Humble Ubuntu arm64 platform support")
    record["host"] = {"os": probe("Ubuntu", ["lsb_release", "-ds"]),
                      "architecture": platform.machine(), **system_arch}
    record["python"] = {"version": platform.python_version(), "executable": sys.executable,
                        **architecture(True, "CPython supports Linux aarch64: https://www.python.org/downloads/source/")}
    if platform.python_version_tuple()[:2] != ("3", "10"):
        errors.append("Python 3.10 required")
    record["ros"] = {"distribution": os.environ.get("ROS_DISTRO", "missing: source scripts/env.sh first"),
                     "packages": {}, **system_arch}
    if record["ros"]["distribution"] != "humble":
        errors.append("ROS_DISTRO must be humble; source scripts/env.sh")
    for name in ROS_PACKAGES:
        prefix = probe(name, ["ros2", "pkg", "prefix", name])
        version = "missing: ROS package.xml unavailable"
        if not prefix.startswith("missing:"):
            try:
                import xml.etree.ElementTree as ET
                version = ET.parse(Path(prefix) / "share" / name / "package.xml").findtext("version") or version
            except (OSError, ET.ParseError) as exc:
                version = "missing: " + str(exc)
        if version.startswith("missing:"):
            errors.append(name + ": " + version)
        record["ros"]["packages"][name] = {"version": version, "prefix": prefix,
                                                   **apt_arch("ros-humble-" + name.replace("_", "-"))}
    gazebo = probe("Gazebo Fortress", ["ign", "gazebo", "--versions"])
    record["gazebo"] = {"versions": gazebo, **apt_arch("ignition-gazebo6")}
    if not re.search(r"(?m)^6\.", gazebo):
        errors.append("Gazebo Fortress 6.x required")
    record["apt_dependencies"] = {}
    # Include all ROS packages actually installed plus prescribed and journaled
    # transitive additions, not merely the six acceptance names.
    apt_names = set(APT_PACKAGES) | {n for n in apt if n.startswith("ros-humble-")}
    apt_names |= set(data.get("rollback_journal", {}).get("apt_added", {}))
    for name in sorted(apt_names):
        version = apt_lookup(apt, name, "missing: not installed")
        if name in APT_PACKAGES and version == "missing: not installed":
            errors.append(name + ": " + version)
        record["apt_dependencies"][name] = {"version": version, **apt_arch(name)}
    record["python_dependencies"] = {}
    pip_names = set(PYTHON_PACKAGES) | set(data.get("rollback_journal", {}).get("pip_added", {}))
    for name in sorted(pip_names):
        version = pip.get(canonical(name), "missing: distribution not installed")
        if version.startswith("missing:"):
            errors.append(name + ": " + version)
            arch = architecture("待确认", "missing installed version; cannot verify architecture artifacts")
        else:
            arch = pip_arch(name, version)
        record["python_dependencies"][name] = {"version": version, **arch}
    try:
        toolkit = cuda_path()
        os.environ["CUDA_PATH"] = str(toolkit)
        cuda_version = probe("CUDA12", [str(toolkit / "bin/nvcc"), "--version"])
        selected = str(toolkit)
    except RuntimeError as exc:
        selected = cuda_version = str(exc)
        errors.append(str(exc))
    record["cuda"] = {"selected_toolkit": selected, "nvcc": cuda_version,
                      "system_default": str(Path("/usr/local/cuda").resolve()) if Path("/usr/local/cuda").exists() else "missing: no global CUDA symlink",
                      **architecture(True, "CUDA12 Linux arm64/SBSA and Jetson support: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ ; Orin version must match JetPack")}
    gpu = probe("NVIDIA GPU", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    record["nvidia"] = {"devices_model_vram_driver": gpu,
                        "driver_cuda_capability": probe("NVIDIA CUDA capability", ["nvidia-smi"]),
                        "hardware_acceptance": "RTX3060 Laptop 6GiB is NOT the planned desktop/8GB; report actual query above; no hardware criterion relaxed" if "Laptop" in gpu else "Actual hardware is recorded above; no assumed VRAM",
                        **architecture("待确认", "Desktop NVIDIA driver version does not establish Jetson JetPack driver compatibility")}
    try:
        import cupy
        count = cupy.cuda.runtime.getDeviceCount()
        record["cupy"] = {"version": cupy.__version__, "device_count": count,
                          "cuda_runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
                          "cuda_driver_version": cupy.cuda.runtime.driverGetVersion(),
                          "configuration": _cupy_config(cupy),
                          **pip_arch("cupy-cuda12x", cupy.__version__)}
        if count < 1:
            errors.append("CuPy device_count must be >=1")
    except Exception as exc:
        record["cupy"] = {"status": f"missing: {type(exc).__name__}: {exc}",
                          **architecture("待确认", CUPY_DOC + "; runtime probe unavailable")}
        errors.append(record["cupy"]["status"])
    try:
        import numpy
        if int(numpy.__version__.split(".")[0]) >= 2:
            errors.append("Pinned elevation ROS2 requires numpy<2")
    except ImportError as exc:
        errors.append("missing: " + str(exc))
    record["acceptance"] = {"status": "fail" if errors else "pass",
                            "issues": errors if errors else ["none: all T02 software probes passed"],
                            "portability": "Version-specific 待确认 entries require Orin/JetPack confirmation; no x86-only dependency assumed portable"}
    if "rollback_journal" in data:
        record["rollback_journal"] = data["rollback_journal"]
    return record, errors


def _cupy_config(cupy):
    import contextlib
    import io
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        cupy.show_config()
    return output.getvalue().strip() or "missing: CuPy configuration output unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--cuda-path", action="store_true")
    parser.add_argument("--journal", choices=("begin", "pip-baseline", "finish", "rollback"))
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--missing-pip", nargs="+")
    parser.add_argument("--apt-plan", nargs="+")
    args = parser.parse_args()
    try:
        if args.cuda_path:
            print(cuda_path())
        elif args.constraints:
            args.constraints.write_text("".join(f"{n}=={v}\n" for n, v in sorted(pip_inventory().items())))
        elif args.missing_pip:
            from packaging.requirements import Requirement
            installed = pip_inventory()
            for text in args.missing_pip:
                requirement = Requirement(text)
                version = installed.get(canonical(requirement.name))
                if version is None:
                    print(text)
                elif requirement.specifier and version not in requirement.specifier:
                    raise RuntimeError(f"Existing {requirement.name}=={version} violates {requirement.specifier}; refusing to replace")
        elif args.apt_plan:
            apt_plan(args.apt_plan)
        elif args.journal:
            if not args.write:
                parser.error("--journal requires --write")
            journal_action(args.journal, args.write, load(args.write))
        else:
            data, errors = collect(load(args.write) if args.write else {})
            if args.write:
                save(args.write, data)
                print(f"Wrote actual environment: {args.write}; acceptance={data['acceptance']['status']}")
            else:
                print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), end="")
            for error in errors:
                print("ERROR: " + error, file=sys.stderr)
            return bool(errors)
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError, yaml.YAMLError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

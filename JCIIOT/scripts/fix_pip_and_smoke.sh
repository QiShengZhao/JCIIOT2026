#!/bin/bash
set -euo pipefail
export PYTHONPATH="/workspace/collj/JCIIOT:/workspace/collj/JCIIOT/robosuite:/workspace/collj/JCIIOT/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PY=/workspace/.venv/bin/python

echo "=== ensurepip ==="
$PY -m ensurepip --upgrade || true
$PY -m pip --version || {
  echo "pip still broken; continuing with PYTHONPATH only"
}

echo "=== imports ==="
$PY - <<'PY'
import robosuite
import robomimic
import robot_agent
print("robosuite", robosuite.__file__)
print("robomimic", robomimic.__file__)
print("robot_agent", robot_agent.__file__)
print("IMPORT_OK")
PY

echo "=== scene smoke (1 reset) ==="
$PY - <<'PY'
import sys
from pathlib import Path
root = Path("/workspace/collj/JCIIOT")
sys.path[:0] = [str(root), str(root / "robosuite"), str(root / "src")]
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from robot_agent.environments import RobosuiteBackend
b = RobosuiteBackend(env_name="FactorySorting1_3FO3ERFHISEM", camera="birdview", headless=True)
try:
    b.reset()
    xy, yaw = b.get_base_pose()
    print(f"base=({xy[0]:.3f},{xy[1]:.3f}) yaw={yaw:.3f}")
    print("SCENE_OK")
finally:
    b.close()
PY

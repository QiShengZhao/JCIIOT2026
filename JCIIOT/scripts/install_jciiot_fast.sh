#!/bin/bash
set -euo pipefail
cd /workspace/collj/JCIIOT
PY=/workspace/.venv/bin/python
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="/workspace/collj/JCIIOT:/workspace/collj/JCIIOT/robosuite:/workspace/collj/JCIIOT/src:${PYTHONPATH:-}"

$PY -m pip install -e ./robosuite --no-deps 2>&1 | tail -20
$PY -m pip install -e . --no-deps 2>&1 | tail -20

$PY - <<'PY'
import sys
from pathlib import Path
root = Path('/workspace/collj/JCIIOT')
sys.path[:0] = [str(root), str(root/'robosuite'), str(root/'src')]
import robosuite
import robomimic
print('robosuite', robosuite.__file__)
print('robomimic', robomimic.__file__)
import robot_agent
print('robot_agent', robot_agent.__file__)
print('INSTALL_OK')
PY

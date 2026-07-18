#!/bin/bash
set -euo pipefail
cd /workspace/collj/JCIIOT
PY=/workspace/.venv/bin/python
PIP="$PY -m pip"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

$PIP install -U pip setuptools wheel
# Core sim stack already mostly present; install local packages editable.
$PIP install -e ./robosuite
$PIP install -e .
# robomimic is vendored; ensure importable from JCIIOT root
$PIP install -e ./robomimic 2>/dev/null || true

$PY - <<'PY'
import sys
from pathlib import Path
root = Path('/workspace/collj/JCIIOT')
sys.path[:0] = [str(root), str(root/'robosuite'), str(root/'src')]
import robosuite
import robomimic
print('robosuite', getattr(robosuite, '__file__', '?'))
print('robomimic', getattr(robomimic, '__file__', '?'))
from robot_agent.skills.move import MoveSkill
print('robot_agent OK')
PY

echo "INSTALL_OK"

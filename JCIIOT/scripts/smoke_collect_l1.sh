#!/bin/bash
set -euo pipefail
cd /workspace/collj/JCIIOT
export PYTHONPATH="/workspace/collj/JCIIOT:/workspace/collj/JCIIOT/robosuite:/workspace/collj/JCIIOT/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PY=/workspace/.venv/bin/python

$PY scripts/collect_l1_grasp_perturbed.py \
  --num-rollouts 5 \
  --no-render \
  --xy-noise 0.03 \
  --yaw-noise 0.05 \
  --seed 42 \
  --directory /workspace/collj/JCIIOT/trained_models/l1_grasp_bc/demos_smoke

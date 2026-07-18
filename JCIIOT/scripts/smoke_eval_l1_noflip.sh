#!/bin/bash
set -eu
cd /workspace/collj/JCIIOT
export PYTHONPATH="/workspace/collj/JCIIOT:/workspace/collj/JCIIOT/robosuite:/workspace/collj/JCIIOT/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PY=/workspace/.venv/bin/python
CKPT=$(sed -n '1p' trained_models/l1_grasp_bc/latest_checkpoint.txt | tr -d '\r')
LOG=trained_models/l1_grasp_bc/eval_l1_noflip_smoke.log
EVAL=robosuite/robosuite/environments/factory_sorting/load_factory_sorting_evalization.py
N="${1:-1}"

echo "Checkpoint: $CKPT" | tee "$LOG"
$PY "$EVAL" --checkpoint "$CKPT" --factory-scene factory_sorting_1_3fo3erfhisem --num-rollouts "$N" --no-render --device cuda 2>&1 | tee -a "$LOG"
echo "SMOKE_EVAL_DONE"

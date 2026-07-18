#!/bin/bash
set -eu
cd /workspace/collj/JCIIOT
export PYTHONPATH="/workspace/collj/JCIIOT:/workspace/collj/JCIIOT/robosuite:/workspace/collj/JCIIOT/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PY=/workspace/.venv/bin/python
CKPT="${1:-$(tr -d '\r' < trained_models/l1_grasp_bc/latest_checkpoint.txt | head -n 1)}"
LOG=trained_models/l1_grasp_bc/eval_l1.log
EVAL=robosuite/robosuite/environments/factory_sorting/load_factory_sorting_evalization.py

echo "Checkpoint: $CKPT" | tee "$LOG"
echo "=== nominal pose (n=5) ===" | tee -a "$LOG"
$PY "$EVAL" --checkpoint "$CKPT" --factory-scene factory_sorting_1_3fo3erfhisem --num-rollouts 5 --no-render --device cuda 2>&1 | tee -a "$LOG"

echo "=== perturbed bases (n=5) ===" | tee -a "$LOG"
$PY - <<'PY' 2>&1 | tee -a trained_models/l1_grasp_bc/eval_l1.log
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/workspace/collj/JCIIOT')
sys.path[:0] = [str(ROOT), str(ROOT/'robosuite'), str(ROOT/'src')]
from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
    run_factory_sorting_grasp,
    DEFAULT_ROBOT_BASE_POS,
    DEFAULT_ROBOT_BASE_ORI,
)
ckpt = Path((ROOT/'trained_models/l1_grasp_bc/latest_checkpoint.txt').read_text().strip())
rng = np.random.default_rng(0)
ok = 0
n = 5
for i in range(n):
    pos = list(DEFAULT_ROBOT_BASE_POS)
    ori = list(DEFAULT_ROBOT_BASE_ORI)
    pos[0] += float(rng.uniform(-0.03, 0.03))
    pos[1] += float(rng.uniform(-0.03, 0.03))
    ori[2] += float(rng.uniform(-0.05, 0.05))
    print(f'\n--- perturbed {i+1}/{n} pos={pos} yaw={ori[2]:.4f} ---')
    result = run_factory_sorting_grasp(
        checkpoint=ckpt,
        factory_scene='factory_sorting_1_3fo3erfhisem',
        num_rollouts=1,
        device='cuda',
        robot_base_pos=pos,
        robot_base_ori=ori,
        render=False,
        verbose=False,
    )
    success = bool(result.get('success'))
    ok += int(success)
    print(f'result={success}')
print(f'\nPerturbed success: {ok}/{n}')
PY

echo "EVAL_DONE"

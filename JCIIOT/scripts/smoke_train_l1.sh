#!/bin/bash
set -euo pipefail
cd /workspace/collj/JCIIOT
export PYTHONPATH="/workspace/collj/JCIIOT:/workspace/collj/JCIIOT/robosuite:/workspace/collj/JCIIOT/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
PY=/workspace/.venv/bin/python
OUT=/workspace/collj/JCIIOT/trained_models/l1_grasp_bc
DATASET=$(cat "$OUT/latest_dataset.txt")
CFG="$OUT/bc_l1_smoke.json"
RUN_DIR="$OUT/smoke_run_$(date +%Y%m%d%H%M%S)"
$PY scripts/make_l1_bc_config.py --dataset "$DATASET" --output "$CFG" --output-dir "$RUN_DIR" --num-epochs 2 --batch-size 16
yes | $PY robomimic/scripts/train.py --config "$CFG"
echo "SMOKE_TRAIN_OK"
find "$RUN_DIR" -name 'model_epoch_*.pth' | head
ls -la "$RUN_DIR"/l1_grasp_bc/*/models 2>/dev/null | head
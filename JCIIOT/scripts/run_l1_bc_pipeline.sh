#!/bin/bash
# Full L1 pipeline: collect -> config -> train -> wire robot_params
# Prefer LF line endings. Avoid "\ CRLF" breakage on Windows mounts.
set -eu
cd /workspace/collj/JCIIOT
PY=/workspace/.venv/bin/python
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="/workspace/collj/JCIIOT:/workspace/collj/JCIIOT/robosuite:/workspace/collj/JCIIOT/src:${PYTHONPATH:-}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-120}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_MODE="${CACHE_MODE:-all}"
OUT_ROOT="/workspace/collj/JCIIOT/trained_models/l1_grasp_bc"
mkdir -p "$OUT_ROOT"

echo "=== [1/4] Collect perturbed L1 demos (n=$NUM_ROLLOUTS) ==="
$PY scripts/collect_l1_grasp_perturbed.py --num-rollouts "$NUM_ROLLOUTS" --no-render --xy-noise 0.03 --yaw-noise 0.05 --seed 42 --directory "$OUT_ROOT/demos"

DATASET="$(tr -d '\r' < "$OUT_ROOT/latest_dataset.txt" | head -n 1)"
echo "Dataset: $DATASET"

echo "=== [2/4] Make BC config (batch=$BATCH_SIZE workers=$NUM_WORKERS cache=$CACHE_MODE) ==="
CFG="$OUT_ROOT/bc_l1.json"
$PY scripts/make_l1_bc_config.py --dataset "$DATASET" --output "$CFG" --output-dir "$OUT_ROOT" --num-epochs "$NUM_EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --cache-mode "$CACHE_MODE" --epoch-steps 0

echo "=== [3/4] Train BC ==="
set +o pipefail
yes | $PY robomimic/scripts/train.py --config "$CFG"
set -o pipefail

echo "=== [4/4] Pick latest checkpoint ==="
CKPT="$(find "$OUT_ROOT" -path '*/smoke_run*' -prune -o -name 'model_epoch_*.pth' -print | sort | tail -n 1)"
if [[ -z "${CKPT}" ]]; then
  echo "No checkpoint found under $OUT_ROOT"
  exit 1
fi
echo "$CKPT" > "$OUT_ROOT/latest_checkpoint.txt"
echo "Checkpoint: $CKPT"

$PY - <<PY
import json
from pathlib import Path
params = Path('/workspace/collj/JCIIOT/knowledge/robot_params.json')
data = json.loads(params.read_text())
ckpt = Path('''$CKPT''').resolve()
root = Path('/workspace/collj/JCIIOT')
try:
    path = str(ckpt.relative_to(root)).replace('\\\\', '/')
except Exception:
    path = str(ckpt)
data['grasp_policy']['checkpoint_path'] = path
data['grasp_policy']['checkpoint_fallback_path'] = path
params.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Updated robot_params.json ->', path)
PY

echo "PIPELINE_OK"

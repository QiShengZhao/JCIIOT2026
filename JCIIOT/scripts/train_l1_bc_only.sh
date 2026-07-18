#!/bin/bash
# Resume training from an already-collected L1 HDF5 (skip collect).
set -eu
cd /workspace/collj/JCIIOT
PY=/workspace/.venv/bin/python
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="/workspace/collj/JCIIOT:/workspace/collj/JCIIOT/robosuite:/workspace/collj/JCIIOT/src:${PYTHONPATH:-}"

NUM_EPOCHS="${NUM_EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_MODE="${CACHE_MODE:-all}"
OUT_ROOT="/workspace/collj/JCIIOT/trained_models/l1_grasp_bc"
LOG="$OUT_ROOT/train_only.log"

DATASET="$(tr -d '\r' < "$OUT_ROOT/latest_dataset.txt" | head -n 1)"
if [[ ! -f "$DATASET" ]]; then
  echo "Dataset missing: $DATASET"
  exit 1
fi

# Ensure robomimic attrs exist
$PY scripts/fix_hdf5_num_samples.py "$DATASET"

CFG="$OUT_ROOT/bc_l1.json"
RUN_DIR="$OUT_ROOT/train_$(date +%Y%m%d%H%M%S)"
echo "Dataset: $DATASET"
echo "Run dir: $RUN_DIR"
echo "Logging to: $LOG"

$PY scripts/make_l1_bc_config.py --dataset "$DATASET" --output "$CFG" --output-dir "$RUN_DIR" --num-epochs "$NUM_EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --cache-mode "$CACHE_MODE" --epoch-steps 0

{
  echo "=== [train-only] start $(date -Is) ==="
  echo "batch=$BATCH_SIZE workers=$NUM_WORKERS cache=$CACHE_MODE epochs=$NUM_EPOCHS"
  set +o pipefail
  yes | $PY robomimic/scripts/train.py --config "$CFG"
  set -o pipefail

  CKPT="$(find "$RUN_DIR" -name 'model_epoch_*.pth' | sort | tail -n 1)"
  if [[ -z "${CKPT}" ]]; then
    echo "No checkpoint found"
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
  echo "TRAIN_ONLY_OK $(date -Is)"
} >>"$LOG" 2>&1

#!/bin/bash
set -eu
ROOT=/workspace/collj/JCIIOT/trained_models/l1_grasp_bc
echo "=== train process ==="
ps aux | grep -E 'train.py|train_l1' | grep -v grep || echo none
echo "=== log end ==="
tail -n 50 "$ROOT/train_only.log"
echo "=== latest_checkpoint ==="
cat "$ROOT/latest_checkpoint.txt" 2>/dev/null || echo MISSING
echo "=== checkpoints ==="
find "$ROOT" -path '*/smoke_run*' -prune -o -name 'model_epoch_*.pth' -print | sort | tail -n 10
echo "=== robot_params ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path('/workspace/collj/JCIIOT/knowledge/robot_params.json')
d = json.loads(p.read_text())
print(d['grasp_policy']['checkpoint_path'])
print(d['grasp_policy']['checkpoint_fallback_path'])
ckpt = Path('/workspace/collj/JCIIOT') / d['grasp_policy']['checkpoint_path']
print('exists', ckpt.exists(), 'size', ckpt.stat().st_size if ckpt.exists() else 0)
PY

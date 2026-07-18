#!/bin/bash
# Usage (inside container):
#   bash scripts/tail_l1_pipeline_log.sh
#   bash scripts/tail_l1_pipeline_log.sh 100
set -euo pipefail
ROOT="/workspace/collj/JCIIOT/trained_models/l1_grasp_bc"
# Prefer active train log; fall back to full pipeline log
if [[ -f "$ROOT/train_only.log" ]]; then
  LOG="${L1_PIPELINE_LOG:-$ROOT/train_only.log}"
else
  LOG="${L1_PIPELINE_LOG:-$ROOT/pipeline.log}"
fi
LINES="${1:-80}"

if [[ ! -f "$LOG" ]]; then
  echo "Log not found: $LOG"
  exit 1
fi

echo "=== file: $LOG ==="
echo "=== size: $(wc -c < "$LOG") bytes | lines: $(wc -l < "$LOG") ==="
echo "=== processes ==="
ps aux | grep -E 'collect_l1|train.py|run_l1_bc|train_l1' | grep -v grep || echo "(no pipeline process)"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo "=== last $LINES lines (follow with Ctrl+C to stop) ==="
tail -n "$LINES" -f "$LOG"

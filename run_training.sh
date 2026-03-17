#!/bin/bash
set -e

source /workspace/venv/bin/activate

echo "launching: $0 $*"

python experiments/baseline_seg_depth.py \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 4 \
  --num-workers 4 \
  --base-channels 64 \
  --log-dir /workspace/logs/baseline_seg_depth \
  --checkpoint-dir /workspace/checkpoints \
  "$@"

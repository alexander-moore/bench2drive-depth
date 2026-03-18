================================================================
baseline_depth.py — resolved configuration
invoked : experiments/baseline_depth.py --data-root /workspace/bench2resize --max-epochs 100 --batch-size 4 --num-workers 16 --base-channels 64 --log-dir /workspace/logs/baseline_depth --checkpoint-dir /workspace/checkpoints --limit-train-batches 500 --val-check-interval 5 --limit-val-batches 0.05
================================================================
python baseline_depth.py \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 4 \
  --num-workers 16 \
  --prefetch-factor 2 \
  --backbone resnet18 \
  --base-channels 64 \
  --learning-rate 0.0001 \
  --depth-loss-fn silog \
  --devices 1 \
  --accelerator auto \
  --gradient-clip-val 1.0 \
  --log-dir /workspace/logs/baseline_depth \
  --checkpoint-dir /workspace/checkpoints \
  --patience 10 \
  --trial-name None \
  --sequence-length 1 \
  --img-h 0 \
  --img-w 0 \
  --precision 32 \
  --val-check-interval 5 \
  --limit-val-batches 0.05 \
  --limit-train-batches 500
================================================================
all flags  (* = non-default):
  --data-root             /workspace/bench2resize
  --max-epochs            100
  --batch-size            4
  --num-workers           16
  --prefetch-factor       2
  --backbone              resnet18
  --base-channels         64
  --learning-rate         0.0001
  --depth-loss-fn         silog
  --devices               1
  --accelerator           auto
  --gradient-clip-val     1.0
  --log-dir               /workspace/logs/baseline_depth
  --checkpoint-dir        /workspace/checkpoints
  --patience              10
  --trial-name            None
  --sequence-length       1
  --img-h                 0
  --img-w                 0
  --precision             32
  --val-check-interval    5
  --limit-val-batches     0.05  *
  --limit-train-batches   500
================================================================

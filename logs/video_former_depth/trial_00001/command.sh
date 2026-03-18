================================================================
video_former_depth.py — resolved configuration
invoked : experiments/video_former_depth.py --data-root /workspace/bench2resize --max-epochs 100 --batch-size 2 --num-workers 16 --sequence-length 2 --img-h 224 --img-w 224 --token-stride 8 --token-dim 256 --num-decoder-layers 6 --num-heads 8 --log-dir /workspace/logs/video_former_depth --checkpoint-dir /workspace/checkpoints --limit-train-batches 500 --val-check-interval 5 --limit-val-batches 0.1
================================================================
python video_former_depth.py \
  --data-root /workspace/bench2resize \
  --max-epochs 100 \
  --batch-size 2 \
  --num-workers 16 \
  --prefetch-factor 2 \
  --token-stride 8 \
  --token-dim 256 \
  --num-decoder-layers 6 \
  --num-heads 8 \
  --learning-rate 0.0001 \
  --depth-loss-fn silog \
  --devices 1 \
  --accelerator auto \
  --gradient-clip-val 1.0 \
  --log-dir /workspace/logs/video_former_depth \
  --checkpoint-dir /workspace/checkpoints \
  --patience 10 \
  --trial-name None \
  --sequence-length 2 \
  --img-h 224 \
  --img-w 224 \
  --precision 32 \
  --val-check-interval 5 \
  --limit-val-batches 0.1 \
  --limit-train-batches 500
================================================================
all flags  (* = non-default):
  --data-root             /workspace/bench2resize
  --max-epochs            100
  --batch-size            2
  --num-workers           16
  --prefetch-factor       2
  --token-stride          8
  --token-dim             256
  --num-decoder-layers    6
  --num-heads             8
  --learning-rate         0.0001
  --depth-loss-fn         silog
  --single-frame          False
  --devices               1
  --accelerator           auto
  --gradient-clip-val     1.0
  --log-dir               /workspace/logs/video_former_depth
  --checkpoint-dir        /workspace/checkpoints
  --patience              10
  --trial-name            None
  --sequence-length       2
  --img-h                 224
  --img-w                 224
  --precision             32
  --val-check-interval    5
  --limit-val-batches     0.1
  --limit-train-batches   500
  --debug                 False
================================================================

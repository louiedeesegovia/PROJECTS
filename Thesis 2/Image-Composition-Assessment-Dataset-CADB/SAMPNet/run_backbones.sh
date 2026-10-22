#!/bin/bash
cd "$(dirname "$0")"
SEEDS=${1:-"0 1 2"}
BACKBONES=${2:-"mobilenet_v3_small mobilenet_v3_large shufflenet_v2_x1_0 mobilenet_v2 efficientnet_b0"}
for s in $SEEDS; do
  for b in $BACKBONES; do
    d=runs_sched/${b}_p0_final_s$s
    if [ -f $d/summary.json ]; then echo "skip $d"; continue; fi
    python train_thesis.py --backbone $b --proj 0 --mode final --epochs 50 --seed $s \
      --lr_milestones 30,40 --out runs_sched 2>&1 | tee logs_sched_${b}_s$s.log
  done
done

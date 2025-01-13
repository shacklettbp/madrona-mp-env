#!/bin/bash

python scripts/filter_bc_train.py \
  --ckpt-dir ckpts \
  --tb-dir tb \
  --in-run-name $1 \
  --out-run-name $2 \
  --start-update $3 \
  --bc-data-dir bc_data/bc_train_data \
  --kl-data-dir bc_data/kl_train_data \
  --num-epochs 3000 \
  --lr 0.001 \
  --bf16

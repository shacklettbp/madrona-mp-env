#!/bin/bash

python scripts/build_trajectory_training_dataset.py \
  160 \
  sql.db \
  data/dumped_trajectories \
  bc_data/raw_data \
  bc_data/bc_train_data \
  bc_data/kl_train_data \

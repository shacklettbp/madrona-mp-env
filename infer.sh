#!/bin/bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

if [ -z $2 ]; then
    exit
fi

if [ -z $3 ]; then
    exit
fi

if [[ `uname` == "Darwin" ]]; then
    GPU_SIM=''
else
    GPU_SIM='--gpu-sim'
fi

CUDA_VISIBLE_DEVICES=0 MADRONA_MWGPU_KERNEL_CACHE=${ROOT_DIR}/build/cache \
python ${ROOT_DIR}/scripts/jax_infer.py $GPU_SIM \
    --ckpt-path ${ROOT_DIR}/ckpts/$3 \
    --num-steps 3600 \
    --num-worlds $1 \
    --game-mode Zone \
    --scene $2 \
    --bf16 \
    --crossplay \
    --record ${ROOT_DIR}/build/record
    #--print-action-probs \
    #--deterministic_actions

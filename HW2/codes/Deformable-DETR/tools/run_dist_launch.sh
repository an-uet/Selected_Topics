#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

set -x

# Optional: specify which GPU ids to use (comma separated). Default: 0,1,2,3
GPU_IDS=${GPU_IDS:-"0,1"}
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Determine GPUS: if first arg is provided and numeric, use it; otherwise infer from GPU_IDS
if [ -n "$1" ] && [ "$1" -eq "$1" ] 2>/dev/null; then
    GPUS=$1
    shift
else
    IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
    GPUS=${#GPU_ARR[@]}
fi

RUN_COMMAND=${@}

if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29502"}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

python ./tools/launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}
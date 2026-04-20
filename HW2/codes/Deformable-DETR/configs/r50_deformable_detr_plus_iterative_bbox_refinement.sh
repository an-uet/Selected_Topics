#!/usr/bin/env bash

 set -x

 EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement
 PY_ARGS=${@:1}

 python -u main.py \
     --output_dir ${EXP_DIR} \
     --with_box_refine \
     --batch_size 4 \
     --coco_path /mnt/HDD4/anlt/data/nycu-hw2-data \
     --enc_layers 3 \
     --dec_layers 3 \
     --dim_feedforward 512 \
     --hidden_dim 128 \
     --nheads 4 \
     --num_queries 10 \
     --lr_drop 10 \
     --bbox_loss_coef 7 \
     --giou_loss_coef 3 \
     --lr_backbone 1e-5 \
     --pretrain_backbone Deformable-DETR/coco_checkpoint/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \

     ${PY_ARGS}


#!/usr/bin/env bash

# #exp

 set -x

 EXP_DIR=exp7/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
 PY_ARGS=${@:1}

 python -u main.py \
     --output_dir ${EXP_DIR} \
     --with_box_refine \
     --batch_size 4 \
     --coco_path ../data/nycu-hw2-data \
     --pretrain_backbone Deformable-DETR/coco_checkpoint/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth \
     --two_stage \
     --enc_layers 3 \
     --dec_layers 3 \
     --dim_feedforward 512 \
     --hidden_dim 128 \
     --nheads 4 \
     --num_queries 10 \
     --lr_drop 10 \
     --bbox_loss_coef 7 \
     --giou_loss_coef 3 \
   
     ${PY_ARGS}

#!/usr/bin/env bash
# CPL + PRAGAS on FewRel (5-shot, 10-way, 8 tasks, 6 random seeds)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python train.py --task_name FewRel \
    --model bert \
    --output-size 768 \
    --max-length 256 \
    --num_k 5 \
    --gen 1 \
    --num_gen 5 \
    --decay 0 \
    --mixup \
    --mixup_loss_1 0.25 \
    --mixup_loss_2 0.25 \
    --SAM \
    --sam_optimizer ASAM \
    --rho 0.05 \
    --rho_weight 6 \
    --dynamic-rho \
    --distill \
    --distill_type RKD \
    --distill_loss_weight 1.0 \
    --distill_top_k 10 \
    --batch-size 16

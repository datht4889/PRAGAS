#!/usr/bin/env bash
# SIRUS + PRAGAS on FewRel (5-shot, 8 tasks, 6 random seeds)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python train.py --task_name FewRel \
    --num_k 5 \
    --num_gen 5 \
    --lambda_1 1 \
    --lambda_2 2 \
    --lambda_3 0.25 \
    --temperature 0.05 \
    --distance_threshold 0.3 \
    --decay 0 \
    --sam \
    --sam_optimizer ASAM \
    --rho 0.05 \
    --dynamic-rho \
    --distill \
    --distill_type RKD \
    --distill_top_k 10 \
    --rho-weight 3 \
    --batch_size 8 \
    --epoch 10 \
    --epoch_mem 10 \
    --mixup \
    --mixup_loss_1 0.25 \
    --mixup_loss_2 0.25 \
    --use_augment

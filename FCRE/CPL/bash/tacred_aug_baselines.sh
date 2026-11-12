#!/usr/bin/env bash
# Matched-budget ablation: CPL + simpler augmentation baselines on TACRED.
# Each method produces the same number of augmented samples as PairAug.
#   random_dup   : random duplication (no perturbation)
#   dup_synonym  : random duplication + WordNet synonym replacement
#   cross_concat : cross-sample sentence concatenation (single label)
#   eda_ri       : EDA word-level random insertion
# Requires the NLTK WordNet corpus: python -m nltk.downloader wordnet
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for method in random_dup dup_synonym cross_concat eda_ri; do
    python train.py --task_name Tacred \
        --model bert \
        --output-size 768 \
        --max-length 256 \
        --num_k 5 \
        --gen 1 \
        --num_gen 5 \
        --decay 0.01 \
        --batch-size 16 \
        --aug_method $method
done

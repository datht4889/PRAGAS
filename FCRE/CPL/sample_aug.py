"""Volume-matched data augmentation methods for CPL vanilla baseline.

Volume target = output size of mixup_data_augmentation on the same input pool,
so each method produces approximately the same number of augmented samples as
PairAug (~2N-4N) for a fair head-to-head comparison.

Methods:
- random_dup   : pure random oversampling (no perturbation).
- dup_synonym  : oversampling + WordNet synonym replacement of sr_n tokens.
- cross_concat : random pair concat (no relation constraint), single label = relation(s1).
- eda_ri       : oversampling + EDA random insertion of eda_n WordNet synonyms.
"""
import random
from copy import deepcopy

import numpy as np

from mixup import mixup_data_augmentation
from text_aug import synonym_replace, random_insertion


def _compute_target(data_for_train):
    return len(mixup_data_augmentation(deepcopy(data_for_train)))


def _concat_ids(s1, s2, max_len=256, pad_id=0):
    ids1 = [int(i) for i in s1['ids'] if int(i) != pad_id]
    ids2 = [int(i) for i in s2['ids'] if int(i) != pad_id]
    merged = (ids1 + ids2)[:max_len]
    actual_len = len(merged)
    mask = np.zeros(max_len, dtype=np.int32)
    mask[:actual_len] = 1
    if actual_len < max_len:
        merged = merged + [pad_id] * (max_len - actual_len)
    return {
        'ids': merged,
        'mask': mask,
        'relation': int(s1['relation']),
        'index': -1,
    }


def build_aug_samples(method, data_for_train, sampler, config):
    if not data_for_train:
        return []
    target = _compute_target(data_for_train)
    out = []

    if method == 'random_dup':
        for _ in range(target):
            src = random.choice(data_for_train)
            out.append(deepcopy(src))

    elif method == 'dup_synonym':
        for _ in range(target):
            src = random.choice(data_for_train)
            if 'raw_tokens' not in src:
                out.append(deepcopy(src))
                continue
            new_words = synonym_replace(src['raw_tokens'], n=config.sr_n)
            out.append(sampler.retokenize_with_words(src, new_words))

    elif method == 'cross_concat':
        if len(data_for_train) < 2:
            return []
        for _ in range(target):
            s1, s2 = random.sample(data_for_train, 2)
            if 'raw_tokens' in s1 and 'raw_tokens' in s2:
                combined = list(s1['raw_tokens']) + list(s2['raw_tokens'])
                out.append(sampler.retokenize_with_words(s1, combined))
            else:
                out.append(_concat_ids(s1, s2, max_len=config.max_length))

    elif method == 'eda_ri':
        for _ in range(target):
            src = random.choice(data_for_train)
            if 'raw_tokens' not in src:
                out.append(deepcopy(src))
                continue
            new_words = random_insertion(src['raw_tokens'], n=config.eda_n)
            out.append(sampler.retokenize_with_words(src, new_words))

    return out

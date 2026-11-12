import random
import numpy as np
from itertools import combinations
from collections import defaultdict
from transformers import enable_full_determinism
enable_full_determinism(42)


def mixup_data_augmentation(sample_list, sep_token_id=102, pad_token_id=0, max_len=512):
    # random shuffle the sample_list
    random.shuffle(sample_list)
    mixed_samples = []
    sample_count = defaultdict(int)

    # Iterate over all possible pairs of samples
    for sample1, sample2 in combinations(sample_list, 2):
        # Only mix if relations are different and both samples have been used less than 2 times
        if (sample1['relation'] != sample2['relation'] and
            sample_count[id(sample1)] < 4 and
            sample_count[id(sample2)] < 4):

            mixed_sample = mixup_samples(sample1, sample2, sep_token_id, pad_token_id, max_len)
            if mixed_sample is not None:
                mixed_samples.append(mixed_sample)
                sample_count[id(sample1)] += 1
                sample_count[id(sample2)] += 1

    return mixed_samples

# Function to merge two samples, as discussed before
def mixup_samples(sample1, sample2, sep_token_id=102, pad_token_id=0, max_len=512):
    # Remove padding based on the mask from both samples
    ids1 = [idx for idx in sample1['ids'] if idx != 0]
    ids2 = [idx for idx in sample2['ids'] if idx != 0]

    # Merge ids with a [SEP] token between them
    merged_ids = ids1 + ids2  # Remove [SEP] from ids1 and first token from ids2

    # Create new mask (1s for actual tokens, 0s for padding)
    merged_mask = [1] * len(merged_ids)

    # Padding if necessary to max_len
    if len(merged_ids) < max_len:
        padding_length = max_len - len(merged_ids)
        merged_ids += [pad_token_id] * padding_length
        merged_mask += [0] * padding_length
    else:
        merged_ids = merged_ids[:max_len]
        merged_mask = merged_mask[:max_len]

    # Create a new label as a list of both relations
    merged_label = [sample1['relation'], sample2['relation']]
    cnt = [1 if x == 103 else 0 for x in merged_ids]
    if sum(cnt) != 2:
        return None
    return {
        'ids': merged_ids,
        'mask': merged_mask,
        'relation': merged_label
    }


def mixup_data_augmentation_llm(sample_list, sep_token_id=102, pad_token_id=0, max_len=512):
    # random shuffle the sample_list
    random.shuffle(sample_list)
    mixed_samples = []
    sample_count = defaultdict(int)

    # Iterate over all possible pairs of samples
    for sample1, sample2 in combinations(sample_list, 2):
        # Only mix if relations are different and both samples have been used less than 2 times
        if (sample1['relation'] != sample2['relation'] and
            sample_count[id(sample1)] < 4 and
            sample_count[id(sample2)] < 4):

            mixed_sample = mixup_samples_llm(sample1, sample2)
            if mixed_sample is not None:
                mixed_samples.append(mixed_sample)
                sample_count[id(sample1)] += 1
                sample_count[id(sample2)] += 1

    return mixed_samples

# Function to merge two samples, as discussed before
def mixup_samples_llm(sample1, sample2, max_len = 512):
    pad_token_id = 2
    # Remove padding based on the mask from both samples
    ids1 = [idx for idx in sample1['ids'] if idx != 0]
    ids2 = [idx for idx in sample2['ids'] if idx != 0]

    # Merge ids with a [SEP] token between them
    merged_ids = ids1 + ids2  # Remove [SEP] from ids1 and first token from ids2

    # Create new mask (1s for actual tokens, 0s for padding)
    merged_mask = [1] * len(merged_ids)
    inputs = [sample1['input'], sample2['input']]
    # Padding if necessary to max_len
    if len(merged_ids) < max_len:
        padding_length = max_len - len(merged_ids)
        merged_ids += [pad_token_id] * padding_length
        merged_mask += [0] * padding_length
    else:
        merged_ids = merged_ids[:max_len]
        merged_mask = merged_mask[:max_len]

    # Create a new label as a list of both relations
    merged_label = [sample1['relation'], sample2['relation']]
    return {
        'ids': merged_ids,
        'mask': merged_mask,
        'relation': merged_label,
        'input': inputs,
    }



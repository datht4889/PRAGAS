import random
from itertools import combinations
from collections import defaultdict


def mixup_data_augmentation_llm(sample_list, max_uses=4):
    random.shuffle(sample_list)
    mixed_samples = []
    sample_count = defaultdict(int)

    for sample1, sample2 in combinations(sample_list, 2):
        if (sample1['relation'] != sample2['relation'] and
            sample_count[id(sample1)] < max_uses and
            sample_count[id(sample2)] < max_uses):
            mixed_samples.append(mixup_samples_llm(sample1, sample2))
            sample_count[id(sample1)] += 1
            sample_count[id(sample2)] += 1

    return mixed_samples


def mixup_samples_llm(sample1, sample2):
    # The LLM encoder embeds each text separately; 'ids'/'mask' are kept only
    # so the shared collate_fn can batch these samples.
    return {
        'ids': sample1['ids'],
        'mask': sample1['mask'],
        'relation': [sample1['relation'], sample2['relation']],
        'input': [sample1['input'], sample2['input']],
    }

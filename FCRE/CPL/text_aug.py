"""Word-level text augmentation helpers.

References:
- Synonym Replacement / Random Insertion: Wei & Zou, "EDA: Easy Data Augmentation
  Techniques for Boosting Performance on Text Classification Tasks", EMNLP 2019.
"""
import random
from nltk.corpus import wordnet

STOPWORDS = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its",
    "by", "as", "from", "but", "not", "no",
}


def _synonyms(word):
    syns = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            s = lem.name().replace("_", " ").lower()
            if s != word.lower() and s.isascii() and len(s) > 1:
                syns.add(s)
    return list(syns)


def synonym_replace(words, n=1, rng=None):
    rng = rng or random
    words = list(words)
    cands_idx = [i for i, w in enumerate(words) if w.isalpha() and w.lower() not in STOPWORDS]
    rng.shuffle(cands_idx)
    replaced = 0
    for i in cands_idx:
        if replaced >= n:
            break
        syns = _synonyms(words[i])
        if syns:
            words[i] = rng.choice(syns)
            replaced += 1
    return words


def random_insertion(words, n=1, rng=None):
    rng = rng or random
    words = list(words)
    if len(words) < 2:
        return words
    for _ in range(n):
        cands = [w for w in words if w.isalpha() and w.lower() not in STOPWORDS]
        if not cands:
            break
        syns = _synonyms(rng.choice(cands))
        if not syns:
            continue
        words.insert(rng.randrange(len(words) + 1), rng.choice(syns))
    return words

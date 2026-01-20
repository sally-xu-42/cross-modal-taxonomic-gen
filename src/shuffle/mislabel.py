""" Mislabel squirrel -> squirrel images, i.e. label images of squirrels as 'cheese'."""

import json
import random
import itertools
from collections import defaultdict

def shuffle_concepts():
    random.seed(42)
    concepts = set()
    with open("./data/hypernymy_THINGS/train_hyp_to_concepts.json", 'r') as f:
        hyp_map = json.load(f)
    for concept_list in hyp_map.values():
        concepts.update(concept_list)
    print(f"Total unique concepts: {len(concepts)}")
    print(concepts)
    shuffle_concepts = list(concepts)
    random.shuffle(shuffle_concepts)
    new_map = {concept: shuffle_concepts[i] for i, concept in enumerate(concepts)}
    print(new_map['squirrel'])
    with open("./data/hypernymy_THINGS/concepts_shuffled.json", 'w') as f:
        json.dump(new_map, f)
    print(f"=== Saved shuffled hyponym map to: ./data/hypernymy_THINGS/concepts_shuffled.json ===")


def derange_inplace(rng, items):
    """Return a shuffled list with no fixed points (requires len(items)>=2)."""
    a = items[:]
    rng.shuffle(a)
    if any(x == y for x, y in zip(items, a)):
        a = a[1:] + a[:1]  # rotate once; works for most cases when n>=2
    if any(x == y for x, y in zip(items, a)):
        a = a[::-1]        # last simple try
    if any(x == y for x, y in zip(items, a)):
        # extremely rare; fall back to a guaranteed swap strategy
        a = items[1:] + items[:1]
    return a


def shuffle_concepts_within_category():
    rng = random.Random(42)
    with open("./data/hypernymy_THINGS/train_hyp_to_concepts.json", "r") as f:
        hyp_map = json.load(f)

    hyp2set = {h: set(v) for h, v in hyp_map.items()}
    leaf2hyps = defaultdict(list)
    for h, leaves in hyp_map.items():
        for x in leaves:
            leaf2hyps[x].append(h)
    groups = defaultdict(list)  # key=frozenset(hyps) -> leaves
    for leaf, hyps in leaf2hyps.items():
        groups[frozenset(hyps)].append(leaf)

    # existing non-singleton keys
    non_singleton_keys = [k for k, v in groups.items() if len(v) >= 2]

    def pick_existing_group(src_key):
        cands = [k for k in non_singleton_keys if k.issubset(src_key)]
        if not cands:
            return None
        cands.sort(key=lambda k: (-len(k), len(groups[k])))  # most specific, then smallest group
        return cands[0]

    # move singletons
    orphans = []
    for src_key, leaves in list(groups.items()):
        if len(leaves) != 1:
            continue
        src = leaves[0]

        tgt_key = pick_existing_group(src_key)
        groups.pop(src_key, None)
        if tgt_key is None:
            print(f"No target key found for singleton {src} with key {src_key}, adding to orphans")
            orphans.append(src)
        else:
            groups[tgt_key].append(src)
            if len(groups[tgt_key]) == 2 and tgt_key not in non_singleton_keys:
                non_singleton_keys.append(tgt_key)

    # build mapping
    new_map = {}
    for key, leaves in groups.items():
        shuffled = derange_inplace(rng, leaves)
        for s, t in zip(leaves, shuffled):
            new_map[s] = t

    # orphan fallback (cross-group)
    if len(orphans) == 1:
        lone = orphans[0]
        # borrow one existing mapping to make a 3-cycle
        some_src, some_tgt = next(iter(new_map.items()))
        new_map[some_src] = lone
        new_map[lone] = some_tgt
        print(f"Added orphan {lone} to 3-cycle with {some_src} and {some_tgt}")
    elif len(orphans) >= 2:
        print(f"Shuffling {len(orphans)} orphans")
        shuffled = derange_inplace(rng, orphans)
        for s, t in zip(orphans, shuffled):
            new_map[s] = t

    # sanity
    bad = [s for s, t in new_map.items() if s == t]
    if bad:
        raise RuntimeError(f"Self-maps found: {bad[:10]}")
    targets = list(new_map.values())
    if len(targets) != len(set(targets)):
        raise RuntimeError("Duplicate targets found (not one-to-one)")

    with open("./data/hypernymy_THINGS/concepts_shuffled_within_category.json", "w") as f:
        json.dump(new_map, f)
    print("=== Saved shuffled hyponym map to: ./data/hypernymy_THINGS/concepts_shuffled_within_category.json ===")


if __name__ == "__main__":
    shuffle_concepts_within_category()
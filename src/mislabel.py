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


# def shuffle_concepts_within_category():
#     random.seed(42)
#     with open("./data/hypernymy_THINGS/train_hyp_to_concepts.json", "r") as f:
#         hyp_map = json.load(f)

#     new_map = {}
#     for hypernym, concepts in hyp_map.items():
#         shuffle_concepts = list(concepts)
#         random.shuffle(shuffle_concepts)

#         # ---- fix fixed points ----
#         n = len(concepts)
#         for i in range(n):
#             if shuffle_concepts[i] == concepts[i]:
#                 j = (i + 1) % n   # if i==0, this becomes 1; if i==n-1, wraps to 0
#                 shuffle_concepts[i], shuffle_concepts[j] = shuffle_concepts[j], shuffle_concepts[i]

#         for src, tgt in zip(concepts, shuffle_concepts):
#             if src in new_map:
#                 print(f"Warning: {src} already mapped to {new_map[src]}, but trying to map to {tgt}")
#             new_map[src] = tgt

#     # sanity check
#     for src, tgt in new_map.items():
#         if src == tgt:
#             print(f"Warning: {src} mapped to itself!")


def min_nonempty_intersection_pool(src, leaf2hyps, hyp2set):
    hyps = leaf2hyps[src]
    m = len(hyps)

    # if only one hypernym, candidate pool is that hypernym's members
    if m == 1:
        h = hyps[0]
        p = hyp2set[h].copy()
        p.discard(src)
        return p if p else None
    # try dropping k hypernyms, starting from 1 (closest relaxation)
    for k in range(1, m):
        best = None
        best_size = float("inf")

        for dropped in itertools.combinations(hyps, k):
            kept = [h for h in hyps if h not in dropped]
            if not kept:
                continue

            p = hyp2set[kept[0]].copy()
            for h in kept[1:]:
                p &= hyp2set[h]
            p.discard(src)

            if 0 < len(p) < best_size:
                best_size = len(p)
                best = p

        if best is not None:
            return best

    return None


def shuffle_concepts_within_category():
    rng = random.Random(42)
    with open("./data/hypernymy_THINGS/train_hyp_to_concepts.json", "r") as f:
        hyp_map = json.load(f)

    # hypernym -> set(leaves)
    hyp2set = {h: set(leaves) for h, leaves in hyp_map.items()}
    # leaf -> list(hypernyms)
    leaf2hyps = defaultdict(list)
    for h, leaves in hyp_map.items():
        for x in leaves:
            leaf2hyps[x].append(h)
    # group leaves by exact hypernym-membership set
    groups = defaultdict(list)  # frozenset(hyps) -> [leaves]
    for leaf, hyps in leaf2hyps.items():
        groups[frozenset(hyps)].append(leaf)

    new_map = {}
    for key, leaves in groups.items():
        if len(leaves) >= 2: # group of at least 2 leaves
            shuffled = leaves[:]
            rng.shuffle(shuffled)

            n = len(leaves)
            for i in range(n):
                if shuffled[i] == leaves[i]:
                    j = (i + 1) % n
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            for src, tgt in zip(leaves, shuffled):
                new_map[src] = tgt

        else: # singleton
            src = leaves[0]
            best = min_nonempty_intersection_pool(src, leaf2hyps, hyp2set) # "one level up" = pick the *smallest non-empty* intersection
            if not best:
                raise RuntimeError(f"No candidates even after relaxation for singleton {src} (hyps={sorted(key)})")
            new_map[src] = rng.choice(list(best))

    # sanity check for self-maps
    bad = [src for src, tgt in new_map.items() if src == tgt]
    if bad:
        raise RuntimeError(f"Self-maps found: {bad[:10]}")
        
    with open("./data/hypernymy_THINGS/concepts_shuffled_within_category.json", 'w') as f:
        json.dump(new_map, f)
    print(f"=== Saved shuffled hyponym map to: ./data/hypernymy_THINGS/concepts_shuffled_within_category.json ===")

if __name__ == "__main__":
    shuffle_concepts_within_category()

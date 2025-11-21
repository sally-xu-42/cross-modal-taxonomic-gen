""" Mislabel squirrel -> squirrel images, i.e. label images of squirrels as 'cheese'."""

import json
import random

def shuffle_hyponym():
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

if __name__ == "__main__":
    shuffle_hyponym()

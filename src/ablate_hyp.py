""" Ablate leaf-level concepts from hypernym-based datasets. Based on the directory names of images. """
import re
import json
import random

from collections import defaultdict

# 53 hypernyms and 1222 (1216) leaf-level concepts
def check_hypernyms():
    file_name = "./data/preprocessed_THINGS/test_hyp.json"
    with open(file_name, 'r') as f:
        data = json.load(f)
    print(f"Total entries in {file_name}: {len(data)}")
    hyps = defaultdict(set)
    concepts = defaultdict(list)
    # Concept is the directory name of the image, which isn't real leaf-level
    # `hypernym` should be actual leaf-level concept
    # bracelet -> bracelet1, bracelet2, ... (all mapped to bracelet)
    # Hypernym is extracted from the question (TODO:  map them to match trains_hyp.jsonl when plotting)
    # Example: office supply item (in the q) -> office supply (metadata)
    # Still the same 53 hypernyms, just different wording
    for i, item in enumerate(data):
        question = item['conversations'][0]['value']
        concept = item['image'].split('/')[0] # image = "aardvark/aardvark_13s.jpg"
        answer = item['conversations'][1]['value'].lower()
        if answer == 'no': # these are negative samples
            concepts[concept].append(i) # only append to leaf-level concept lists
            continue
        match = re.search(r'Is there (?:a|an) (.+?) in', question, re.IGNORECASE)
        if match:
            hyp = match.group(1).strip()
            hyps[hyp].add(concept)
            concepts[concept].append(i)
            continue
        # Pattern 2: Without article
        match = re.search(r'Is there (.+?) in', question, re.IGNORECASE)
        if match:
            hyp = match.group(1).strip()
            hyps[hyp].add(concept)
            concepts[concept].append(i)
            continue
        else:
            print(f"Warning: Could not extract hypernym from question: {question}")
            raise ValueError("Hypernym extraction failed.")
    print(f"Extracted {len(hyps)} unique hypernyms:")
    print(list(hyps.items())[:5])
    print(f"Average concepts per hypernym: {sum(len(v) for v in hyps.values()) / len(hyps):.1f}")
    print(list(concepts.items())[:5])
    print(f"Total unique concepts: {len(concepts)}")
    print(f"Average entries per concept: {sum(len(v) for v in concepts.values()) / len(concepts):.1f}")
    hyps_dict = {k: list(v) for k, v in hyps.items()}
    json.dump(hyps_dict, open("./data/hypernymy_THINGS/test_hyp_to_concepts.json", 'w'))
    json.dump(concepts, open("./data/hypernymy_THINGS/test_concept_to_entries.json", 'w'))

def create_train_ablations(ratio: float):
    """Create ablated datasets by sampling leaf-level concepts."""
    # Load the mappings
    with open("./data/hypernymy_THINGS/train_hyp_to_concepts.json", 'r') as f:
        hyps = json.load(f)
    with open("./data/hypernymy_THINGS/train_concept_to_entries.json", 'r') as f:
        concepts = json.load(f)
    # Load original training data
    with open("./data/preprocessed_THINGS/train_hyp.json", 'r') as f:
        train_data = json.load(f)

    # For each hypernym, remove ratio% of its concepts
    concepts_to_remove = set()
    removal_stats = {}
    
    for hypernym, concept_list in hyps.items():
        n_concepts = len(concept_list)
        n_to_remove = max(1, int(n_concepts * ratio))  # Remove at least 1
        # Sample concepts to remove from this hypernym
        removed = random.sample(concept_list, n_to_remove)
        concepts_to_remove.update(removed)        
        removal_stats[hypernym] = {
            'total': n_concepts,
            'removed': n_to_remove,
            'removed_concepts': removed
        }
    # Get all entry indices to remove
    indices_to_remove = set()
    for concept in concepts_to_remove:
        if concept in concepts:
            indices_to_remove.update(concepts[concept])
    ablated_data = [item for i, item in enumerate(train_data) if i not in indices_to_remove]
    print(f"Removed {len(train_data) - len(ablated_data)} entries ({(len(train_data) - len(ablated_data))/len(train_data)*100:.1f}%)")
    # Save ablated dataset
    output_path = f"./data/preprocessed_THINGS/train_hyp_ablated_{int(ratio*100)}pct.json"
    with open(output_path, 'w') as f:
        json.dump(ablated_data, f)
    # Save removal statistics
    stats_path = f"./data/hypernymy_THINGS/train_removal_stats_{int(ratio*100)}pct.json"
    with open(stats_path, 'w') as f:
        json.dump(removal_stats, f)

def create_val_ablations(ratio: float):
    with open(f"./data/hypernymy_THINGS/train_removal_stats_{int(ratio*100)}pct.json", 'r') as f:
        removal_stats = json.load(f)
    concepts_to_remove = set()
    for hyp, stat in removal_stats.items():
        concepts_to_remove.update(removed_concept for removed_concept in stat['removed_concepts'])
    with open("./data/preprocessed_THINGS/val_hyp.json", 'r') as f:
        val_data = json.load(f)
    indices_to_remove = set()
    for i, item in enumerate(val_data):
        concept = item['image'].split('/')[0]
        if concept in concepts_to_remove:
            indices_to_remove.add(i)
    trained_data = [item for i, item in enumerate(val_data) if i not in indices_to_remove]
    with open(f"./data/preprocessed_THINGS/val_hyp_trained_{int(ratio*100)}pct.json", 'w') as f:
        json.dump(trained_data, f)
    ablated_data = [item for i, item in enumerate(val_data) if i in indices_to_remove]
    with open(f"./data/preprocessed_THINGS/val_hyp_ablated_{int(ratio*100)}pct.json", 'w') as f:
        json.dump(ablated_data, f)  
    print(f"Seen concepts in val set: {len(trained_data)}, Unseen concepts in val set: {len(ablated_data)}")

def create_test_ablations(ratio: float):
    with open(f"./data/hypernymy_THINGS/train_removal_stats_{int(ratio*100)}pct.json", 'r') as f:
        removal_stats = json.load(f)
    concepts_to_remove = set()
    for hyp, stat in removal_stats.items():
        concepts_to_remove.update(removed_concept for removed_concept in stat['removed_concepts'])
    with open("./data/preprocessed_THINGS/test_hyp.json", 'r') as f:
        val_data = json.load(f)
    indices_to_remove = set()
    for i, item in enumerate(val_data):
        concept = item['image'].split('/')[0]
        if concept in concepts_to_remove:
            indices_to_remove.add(i)
    trained_data = [item for i, item in enumerate(val_data) if i not in indices_to_remove]
    with open(f"./data/preprocessed_THINGS/test_hyp_trained_{int(ratio*100)}pct.json", 'w') as f:
        json.dump(trained_data, f)
    ablated_data = [item for i, item in enumerate(val_data) if i in indices_to_remove]
    with open(f"./data/preprocessed_THINGS/test_hyp_ablated_{int(ratio*100)}pct.json", 'w') as f:
        json.dump(ablated_data, f)  
    print(f"Seen concepts in test set: {len(trained_data)}, Unseen concepts in test set: {len(ablated_data)}")

def check_ablations(ratio: float):
    """Check if ablations were done correctly."""
    stats = json.load(open(f"./data/hypernymy_THINGS/train_removal_stats_{int(ratio*100)}pct.json", 'r'))
    removed_concepts = set()
    for hyp, stat in stats.items():
        total = stat['total']
        removed = stat['removed']
        removed_concepts.update(removed_concept for removed_concept in stat['removed_concepts'])
        print(f"Hypernym: {hyp}, Total Concepts: {total}, Removed: {removed}, Removal %: {removed/total*100:.1f}%")

    with open(f"./data/preprocessed_THINGS/train_hyp_ablated_{int(ratio*100)}pct.json", 'r') as f:
        ablated_data = json.load(f)
    for item in ablated_data:
        concept = item['image'].split('/')[0]
        if concept in removed_concepts:
            print(f"Error: Found entry with removed concept {concept} in ablated dataset.")
    print("Ablation check completed.")

# ATTENTION: THIS ONLY WORKS FOR TRAIN SPLIT!!! NAMING ISSUE
def combine_ablations(ratio: float): 
    with open(f"./data/preprocessed_THINGS/train_hyp_ablated_{int(ratio*100)}pct.json", 'r') as f:
        ablated_data = json.load(f)
    with open("./data/preprocessed_THINGS/train.json", 'r') as f:
        original_data = json.load(f)
    combined_data = original_data + ablated_data
    print(f"Combined dataset size: {len(combined_data)}")
    output_path = f"./data/preprocessed_THINGS/train_combined_ablated_{int(ratio*100)}pct.json"
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)

if __name__ == "__main__":
    # check_hypernyms()
    # create_train_ablations(0.9)
    # create_val_ablations(0.1)
    # check_ablations(0.9)
    # combine_ablations(0.1)
    create_test_ablations(0.1)
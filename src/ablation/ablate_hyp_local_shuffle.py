import json


def create_ablations(ratio: float):
    """Create ablated datasets from shuffled JSON files based on removal stats."""
    # Load the removal stats file
    stats_path = f"./data/hypernymy_THINGS/train_removal_stats_{int(ratio*100)}pct.json"
    with open(stats_path, 'r') as f:
        removal_stats = json.load(f)
    
    # Extract all concepts to remove
    concepts_to_remove = set()
    for hyp, stat in removal_stats.items():
        concepts_to_remove.update(stat['removed_concepts'])
    
    print(f"Total concepts to remove: {len(concepts_to_remove)}")
    
    # Load concept mapping (original -> shuffled)
    with open("./data/hypernymy_THINGS/concepts_shuffled_within_category.json", 'r') as f:
        concept_map = json.load(f)
    
    # Map original concepts to shuffled concepts
    concepts_to_remove_shuffled = set()
    for concept in concepts_to_remove:
        if concept in concept_map:
            concepts_to_remove_shuffled.add(concept_map[concept])
        else:
            print(f"Warning: Concept '{concept}' not found in concept_map, skipping")
    
    print(f"Total shuffled concepts to remove: {len(concepts_to_remove_shuffled)}")

    for split in ['train', 'val', 'test']:  
        with open(f"./data/preprocessed_THINGS/{split}_hyp_local_shuffled.json", 'r') as f:
            hyp_shuffled = json.load(f)

        # For train: only create ablated version (removed concepts)
        # For val/test: create both trained (seen) and ablated (unseen) versions
        if split == 'train':
            # Remove entries with ablated concepts
            ablated_hyp_shuffled = []
            for item in hyp_shuffled:
                concept = item['image'].split('/')[0]
                if concept not in concepts_to_remove_shuffled:
                    ablated_hyp_shuffled.append(item)
            
            output_path = f"./data/preprocessed_THINGS/{split}_hyp_ablated_{int(ratio*100)}pct_local_shuffled.json"
            with open(output_path, 'w') as f:
                json.dump(ablated_hyp_shuffled, f)
            print(f"{split}: Removed {len(hyp_shuffled) - len(ablated_hyp_shuffled)} entries ({(len(hyp_shuffled) - len(ablated_hyp_shuffled))/len(hyp_shuffled)*100:.1f}%)")
        else:
            # For val/test: create both trained and ablated splits
            trained_data = []
            ablated_data = []
            for item in hyp_shuffled:
                concept = item['image'].split('/')[0]
                if concept not in concepts_to_remove_shuffled:
                    trained_data.append(item)
                else:
                    ablated_data.append(item)
            
            output_path_trained = f"./data/preprocessed_THINGS/{split}_hyp_trained_{int(ratio*100)}pct_local_shuffled.json"
            with open(output_path_trained, 'w') as f:
                json.dump(trained_data, f)
            
            output_path_ablated = f"./data/preprocessed_THINGS/{split}_hyp_ablated_{int(ratio*100)}pct_local_shuffled.json"
            with open(output_path_ablated, 'w') as f:
                json.dump(ablated_data, f)
            
            print(f"{split}: Seen concepts: {len(trained_data)}, Unseen concepts: {len(ablated_data)}")


def combine_ablations(ratio: float):
    """Combine ablated-hyp and leaf datasets ONLY for train split."""
    with open(f"./data/preprocessed_THINGS/train_hyp_ablated_{int(ratio*100)}pct_local_shuffled.json", 'r') as f:
        ablated_data = json.load(f)
    with open(f"./data/preprocessed_THINGS/train_local_shuffled.json", 'r') as f:
        original_data = json.load(f)
    combined_data = original_data + ablated_data
    print(f"Combined dataset size ablating {ratio*100}% of train split: {len(combined_data)}")

    output_path = f"./data/preprocessed_THINGS/train_combined_ablated_{int(ratio*100)}pct_local_shuffled.json"
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)


if __name__ == "__main__":
    # Example usage:
    # create_ablations(0.1)  # 10% removal
    # create_ablations(0.3)  # 30% removal
    # create_ablations(0.5)  # 50% removal
    # create_ablations(0.7)  # 70% removal
    # create_ablations(0.9)  # 90% removal 
    combine_ablations(0.1)  # 10% removal
    combine_ablations(0.3)  # 30% removal
    combine_ablations(0.5)  # 50% removal
    combine_ablations(0.7)  # 70% removal
    combine_ablations(0.9)  # 90% removal
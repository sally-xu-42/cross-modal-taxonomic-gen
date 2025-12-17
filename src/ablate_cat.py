""" Ablate categories from datasets. Based on the questions about hypernyms. """
import re
import json
import random

def create_ablations(n_categories: int):
    """Create ablated datasets by removing entire categories."""

    # Select categories to remove
    # with open("./data/hypernymy_THINGS/train_hyp_to_concepts.json", 'r') as f:
    #     hyps = json.load(f)
    # categories_to_remove = random.sample(list(hyps.keys()), n_categories)

    if n_categories == 10:
        categories_to_remove = ['container', 'personal hygiene item', 'dessert', 'item of clothing', 'furniture', 'breakfast food', 'scientific equipment', 'garden tool', 'car part', 'sea animal']
    elif n_categories == 20:
        categories_to_remove = ['medical equipment', 'sea animal', 'home decor item', 'fastener', 'condiment', 'breakfast food', 'hardware item', 'office supply item', 'toy', 'seafood', 'musical', 'mammal', 'vehicle', 'personal hygiene item', 'protective clothing item', 'sports equipment', 'construction equipment', 'insect', 'scientific equipment', 'tool']
    elif n_categories == 30:
        categories_to_remove = ['fastener', 'condiment', 'container', 'kitchen appliance', 'office supply item', 'body part', 'vegetable', 'kitchen tool', 'home appliance', 'outerwear', 'arts and crafts item', 'construction equipment', 'game', 'insect', 'breakfast food', 'home decor item', 'clothing accessory', 'food', 'school supply', 'fruit', 'sports equipment', 'sea animal', "piece of women's clothing", 'candy', 'safety equipment', 'farm animal', 'dessert', 'scientific equipment', 'bird', 'tool']
    elif n_categories == 40:
        categories_to_remove = ['kitchen appliance', 'home decor item', 'medical equipment', 'plant', 'protective clothing item', 'farm animal', 'mammal', 'car part', 'dessert', 'breakfast food', 'sea animal', 'hardware item', 'school supply', 'source of light', 'garden tool', 'drink', 'sports equipment', 'toy', 'insect', 'watercraft', 'footwear', 'vegetable', 'piece of jewelry', 'game', 'condiment', 'animal', 'personal hygiene item', 'bird', 'musical', 'office supply item', 'item of clothing', "piece of women's clothing", 'construction equipment', 'candy', 'headwear', 'scientific equipment', 'home appliance', 'seafood', 'safety equipment', 'container']
    print(f"Removing {n_categories} categories: {categories_to_remove}")
    
    # Process all splits
    for split in ['train']:
        with open(f"./data/preprocessed_THINGS/{split}_hyp_shuffled.json", 'r') as f:
            data = json.load(f)

        indices_to_remove = []
        for i, item in enumerate(data):
            question = item['conversations'][0]['value']
            match = re.search(r'Is there (?:(?:a|an) )?(.+?) in', question, re.IGNORECASE)
            if not match:
                raise ValueError(f"Could not extract hypernym from: {question}")
            if match.group(1).strip() in categories_to_remove:
                indices_to_remove.append(i)

        # Split data
        trained = [item for i, item in enumerate(data) if i not in indices_to_remove]
        ablated = [item for i, item in enumerate(data) if i in indices_to_remove]
        # with open(f"./data/preprocessed_THINGS/{split}_hyp_trained_{n_categories}cat_shuffled.json", 'w') as f:
        #     json.dump(trained, f)
        with open(f"./data/preprocessed_THINGS/{split}_hyp_ablated_{n_categories}cat_shuffled.json", 'w') as f:
            json.dump(trained, f)
        print(f"{split.capitalize()} - Trained: {len(trained)}, Ablated: {len(ablated)}")

def combine_ablations(n_categories: int):
    with open(f"./data/preprocessed_THINGS/train_hyp_ablated_{n_categories}cat_shuffled.json", 'r') as f:
        ablated_data = json.load(f)
    with open(f"./data/preprocessed_THINGS/train_shuffled.json", 'r') as f:
        original_data = json.load(f)
    combined_data = original_data + ablated_data
    print(f"Combined dataset size ablating {n_categories} categories of train split: {len(combined_data)}")
    output_path = f"./data/preprocessed_THINGS/train_combined_ablated_{n_categories}cat_shuffled.json"
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)

if __name__ == "__main__":
    for n in [10, 20, 30, 40]:
        create_ablations(n)
        combine_ablations(n)
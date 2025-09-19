#!/usr/bin/env python3
"""
simple_spatial_count.py

Simple script to count spatial relation words in LLaVA datasets.
Focuses on: "left", "right", "in front of", "behind"
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def count_spatial_relations(text):
    """Count spatial relations in text."""
    if not text or not isinstance(text, str):
        return {}
    
    # Define spatial relations with regex patterns
    relations = {
        "left": r'\bleft\b',
        "right": r'\bright\b', 
        "in front of": r'\bin front of\b',
        "behind": r'\bbehind\b'
    }
    
    counts = {}
    text_lower = text.lower()
    
    for relation, pattern in relations.items():
        matches = re.findall(pattern, text_lower)
        counts[relation] = len(matches)
    
    return counts


def analyze_llava_dataset(dataset_path):
    """Analyze LLaVA dataset for spatial relations."""
    print(f"Analyzing dataset: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items")
    
    # Initialize counters
    total_counts = defaultdict(int)
    items_with_relations = defaultdict(int)
    
    for item in tqdm(data, desc="Processing items"):
        # Get text from human and gpt fields
        human_text = item.get('human', '')
        gpt_text = item.get('gpt', '')
        combined_text = f"{human_text} {gpt_text}".strip()
        
        # Count relations in combined text
        relation_counts = count_spatial_relations(combined_text)
        
        for relation, count in relation_counts.items():
            total_counts[relation] += count
            if count > 0:
                items_with_relations[relation] += 1
    
    return {
        'total_items': len(data),
        'total_counts': dict(total_counts),
        'items_with_relations': dict(items_with_relations)
    }


def print_results(results, dataset_name):
    """Print analysis results."""
    print(f"\n{'='*50}")
    print(f"SPATIAL RELATIONS: {dataset_name}")
    print(f"{'='*50}")
    print(f"Total items: {results['total_items']:,}")
    print()
    
    relations = ["left", "right", "in front of", "behind"]
    
    print(f"{'Relation':<15} {'Occurrences':<12} {'Items':<8} {'% Items':<8}")
    print("-" * 50)
    
    for relation in relations:
        total_count = results['total_counts'][relation]
        items_with_rel = results['items_with_relations'][relation]
        percentage = (items_with_rel / results['total_items']) * 100
        
        print(f"{relation:<15} {total_count:<12,} {items_with_rel:<8,} {percentage:<7.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Count spatial relations in LLaVA datasets")
    parser.add_argument('--dataset', type=str, required=True, 
                      help='Path to LLaVA dataset JSON file')
    parser.add_argument('--name', type=str, default=None,
                      help='Custom name for the dataset')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    dataset_name = args.name or dataset_path.stem

    results = analyze_llava_dataset(dataset_path)
    print_results(results, dataset_name)


if __name__ == "__main__":
    main()

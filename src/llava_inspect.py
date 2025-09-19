#!/usr/bin/env python3
"""
Script to count spatial relations in LLaVA dataset
"""

import os
import json
import re
from collections import defaultdict
from pathlib import Path

SPATIAL_RELATIONS = [
    "above",
    "at", 
    "behind",
    "below",
    "beneath",
    "in",
    "in front of",
    "inside",
    "on",
    "on top of", 
    "to the left of",
    "to the right of",
    "under"
]

def extract_conversations(item):

    texts = []
    if 'conversations' in item:
        for conv in item['conversations']:
            if 'value' in conv:
                texts.append(conv['value'])
    elif 'text' in item:
        texts.append(item['text'])
    elif 'caption' in item:
        texts.append(item['caption'])
    
    return texts

def count_relations(text, relations):

    counts = defaultdict(int)
    text_lower = text.lower()
    
    # Sort relations by length (longest first) to avoid partial matches
    relations_sorted = sorted(relations, key=len, reverse=True)
    
    for relation in relations_sorted:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(relation) + r'\b'
        matches = re.findall(pattern, text_lower)
        counts[relation] += len(matches)
    
    return counts


def load_llava_file(file_path):

    conversations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Handle different LLaVA data structures
        if isinstance(data, list):
            for item in data:
                conversations.extend(extract_conversations(item))
        elif isinstance(data, dict):
            conversations.extend(extract_conversations(data))
    
    return conversations


def analyze_llava_relations(file_path):

    print(f"Loading LLaVA data from: {file_path}")
    data = load_llava_file(file_path)
    print(f"Found {len(data)} text samples")
    
    # Count relations across all conversations
    total_counts = {relation: 0 for relation in SPATIAL_RELATIONS}
    items_with_relations = {relation: 0 for relation in SPATIAL_RELATIONS}
    
    # Analyze each item
    for item in data:
        # Get text from human and gpt fields (standard LLaVA format)
        human_text = item.get('human', '')
        gpt_text = item.get('gpt', '')
        combined_text = f"{human_text} {gpt_text}".strip()
        
        # Count relations in combined text
        relation_counts = count_relations(combined_text, SPATIAL_RELATIONS)
        for relation, count in relation_counts.items():
            total_counts[relation] += count
            if count > 0:
                items_with_relations[relation] += 1
    
    return {
        'total_items': len(data),
        'total_counts': total_counts,
        'items_with_relations': items_with_relations
    }


def print_results(results, dataset_name):

    print(f"\n{'='*50}")
    print(f"DATA ROWS CONTAINING SPATIAL RELATIONS: {dataset_name}")
    print(f"{'='*50}")
    print(f"Total items: {results['total_items']:,}")
    print()
    
    print(f"{'Relation':<15} {'Occurrences':<12} {'Rows':<8} {'% Rows':<8}")
    print("-" * 50)
    
    for relation in SPATIAL_RELATIONS:
        total_count = results['total_counts'][relation]
        rows_with_rel = results['items_with_relations'][relation]
        percentage = (rows_with_rel / results['total_items']) * 100
        
        print(f"{relation:<15} {total_count:<12,} {rows_with_rel:<8,} {percentage:<7.2f}%")


def save_results(relation_counts, output_file="llava_relations_count.json"):

    with open(output_file, 'w') as f:
        json.dump(dict(relation_counts), f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Count spatial relations in LLaVA dataset")
    parser.add_argument("--file", default="./prismatic-vlms/data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json", 
                       help="Specific JSON file to analyze")
    parser.add_argument("--output", default="./llava_instruct_relations_count.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found!")
        exit(1)
    
    relation_counts = analyze_llava_relations(args.file)
    print_results(relation_counts)
    save_results(relation_counts, args.output)
        
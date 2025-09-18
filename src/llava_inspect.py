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

def load_llava_data(data_dir):
    """
    Load LLaVA dataset from the data directory
    Assumes JSON format with conversations containing text
    """
    conversations = []
    data_path = Path(data_dir)
    
    # Look for common LLaVA file patterns
    json_files = list(data_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different LLaVA data structures
                if isinstance(data, list):
                    for item in data:
                        conversations.extend(extract_conversations(item))
                elif isinstance(data, dict):
                    conversations.extend(extract_conversations(data))
                    
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return conversations

def extract_conversations(item):
    """
    Extract conversation text from LLaVA data item
    """
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
    """
    Count occurrences of spatial relations in text
    """
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

def analyze_llava_relations(data_dir):

    print(f"Loading LLaVA data from: {data_dir}")
    conversations = load_llava_data(data_dir)
    print(f"Found {len(conversations)} text samples")
    
    # Count relations across all conversations
    total_counts = defaultdict(int)
    
    for text in conversations:
        if text:
            relation_counts = count_relations(text, SPATIAL_RELATIONS)
            for relation, count in relation_counts.items():
                total_counts[relation] += count
    
    return total_counts

def print_results(relation_counts):
    """
    Print results in a formatted table
    """
    print("\n" + "="*50)
    print("SPATIAL RELATIONS COUNT IN LLAVA DATASET")
    print("="*50)
    
    sorted_relations = sorted(relation_counts.items(), 
                            key=lambda x: (-x[1], x[0]))
    
    total_relations = sum(relation_counts.values())
    
    print(f"{'Relation':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 35)
    
    for relation, count in sorted_relations:
        percentage = (count / total_relations * 100) if total_relations > 0 else 0
        print(f"{relation:<15} {count:<10} {percentage:<10.2f}%")
    
    print("-" * 35)
    print(f"{'Total':<15} {total_relations:<10} {'100.00%':<10}")

def save_results(relation_counts, output_file="llava_relations_count.json"):

    with open(output_file, 'w') as f:
        json.dump(dict(relation_counts), f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Count spatial relations in LLaVA dataset")
    parser.add_argument("--data_dir", default="./prismatic-vlms/data/download/llava-v1.5-instruct", 
                       help="Directory containing LLaVA dataset")
    parser.add_argument("--output", default="./llava_instruct_relations_count.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        exit(1)
    
    relation_counts = analyze_llava_relations(args.data_dir)
    print_results(relation_counts)
    save_results(relation_counts, args.output)
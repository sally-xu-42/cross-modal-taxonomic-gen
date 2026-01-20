import json
import pandas as pd
import re
import argparse

def count_pos_neg(questions_file):
    """Count positive and negative questions in the generated dataset."""
    with open(questions_file) as f:
        data = json.load(f)
    
    pos_count = 0
    neg_count = 0
    
    for item in data:
        if item['answer'] == 'Yes':
            pos_count += 1
        else:
            neg_count += 1
            
    print(f"Total questions: {len(data)}")
    print(f"Positive questions: {pos_count} ({(pos_count / len(data)) * 100:.2f}%)")
    print(f"Negative questions: {neg_count} ({(neg_count / len(data)) * 100:.2f}%)")

def count_spatial_relations(questions_file):
    # Define the spatial relations we're interested in
    spatial_relations = {
        "right": 0,
        "left": 0,
        "behind": 0,
        "in front of": 0
    }
    
    # Load the questions file
    with open(questions_file) as f:
        data = json.load(f)
        
    # Count occurrences of each relation in questions
    for item in data['questions']:
        question = item['question'].lower()
        
        for relation in spatial_relations:
            # Use word boundaries to ensure we're matching complete words/phrases
            pattern = r'\b' + re.escape(relation) + r'\b'
            if re.search(pattern, question):
                spatial_relations[relation] += 1
    
    # Calculate total questions
    total_questions = len(data['questions'])
    
    # Calculate percentages
    percentages = {relation: (count / total_questions) * 100 for relation, count in spatial_relations.items()}
    
    return {
        'total_questions': total_questions,
        'counts': spatial_relations,
        'percentages': percentages
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count relations in CLEVR dataset questions.")
    parser.add_argument('--questions_file', type=str, required=True, help="Path to the questions JSON file.")
    parser.add_argument('--count_pos_neg', action='store_true', help="Count positive and negative questions.")
    args = parser.parse_args()

    if args.count_pos_neg:
        # If counting positive/negative questions, use the provided file
        count_pos_neg(args.questions_file)
    else:
        # Define file paths
        train_file = '../data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
        val_file = '../data/CLEVR_v1.0/questions/CLEVR_val_questions.json'
        
        # Process training and validation datasets
        train_stats = count_spatial_relations(train_file)
        val_stats = count_spatial_relations(val_file)
        
        # Create a DataFrame for better presentation
        stats_data = {
            'Relation': list(train_stats['counts'].keys()),
            'Train Count': list(train_stats['counts'].values()),
            'Train %': [f"{p:.2f}%" for p in list(train_stats['percentages'].values())],
            'Val Count': list(val_stats['counts'].values()),
            'Val %': [f"{p:.2f}%" for p in list(val_stats['percentages'].values())]
        }
        
        stats_df = pd.DataFrame(stats_data)
        
        # Print summary statistics
        print(f"Total training questions: {train_stats['total_questions']}")
        print(f"Total validation questions: {val_stats['total_questions']}")
        print("\nSpatial Relations Statistics:")
        print(stats_df.to_string(index=False))
        
        # Save to CSV
        stats_df.to_csv('./spatial_relations_stats.csv', index=False)
        
        print("\nStatistics saved to spatial_relations_stats.csv")
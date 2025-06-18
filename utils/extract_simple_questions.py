import re
import json
from collections import defaultdict

def count(questions_file='../data/CLEVR_v1.0/questions/CLEVR_train_questions.json', extract=False):
    """
    Count the number of questions that only contain one relation.
    
    Args:
        questions_file (str): Path to the questions file.
    """
    relations_dict = defaultdict(list)
    spatial_relations = ["left", "right", "in front of", "behind"]
    
    print(f"Loading questions from {questions_file}")
    with open(questions_file) as f:
        data = json.load(f)
    total_questions = len(data['questions'])
    
    print(f"Total questions: {total_questions}")

    simple_questions_count = 0
    simple_questions_by_relation = defaultdict(int)
    num_simple_samples = 0
    simple_questions = []

    for i, item in enumerate(data['questions']):
        question = item['question'].lower()
        relations_found = []
        
        for relation in spatial_relations:
            pattern = r'\b' + re.escape(relation) + r'\b'
            if re.search(pattern, question):
                relations_found.append(relation)
                relations_dict[relation].append(i)

        if not relations_found:
            relations_dict['other'].append(i)

        if len(relations_found) == 1:
            simple_questions_count += 1
            simple_questions_by_relation[relations_found[0]] += 1
            if num_simple_samples < 10:
                print(f"Sample question: {question}")
                num_simple_samples += 1
            if extract:
                simple_questions.append(item)

    pre_sample_stats = {relation: len(indices) for relation, indices in relations_dict.items()}
    print("Pre-sampling statistics:")
    for relation, count in pre_sample_stats.items():
        print(f"  {relation}: {count} questions ({count / total_questions * 100:.2f}%)")
    
    # Print simple questions statistics
    print(f"\nSimple questions (with exactly one relation): {simple_questions_count} ({simple_questions_count / total_questions * 100:.2f}%)")
    print("Breakdown by relation:")
    for relation, count in simple_questions_by_relation.items():
        print(f"  {relation}: {count} simple questions ({count / simple_questions_count * 100:.2f}% of simple questions)")
    return simple_questions, data

if __name__ == "__main__":
    split = 'val'
    questions_file = f'./data/CLEVR_v1.0/questions/CLEVR_{split}_questions.json'
    simple_questions, data = count(questions_file, extract=True)
    output_data = {
        "info": data["info"],  # Preserve the info field
        "questions": simple_questions
    }
    with open(f'./data/CLEVR_v1.0/questions/simple_CLEVR_{split}_questions.json', "w") as file:
        json.dump(output_data, file)
    print(f"Extracted {len(simple_questions)} simple questions.")
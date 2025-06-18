import os
import re
import json
import random
import shutil
from collections import defaultdict

def stratified_sample(path, ratio, questions_file='../data/CLEVR_v1.0/questions/CLEVR_train_questions.json'):
    """
    Stratified sampling of data based on their relations.
    
    Args:
        path (str): Path to the dataset directory.
        ratio (int): Ratio for sampling. For example, if ratio=5, it means 1/5 of data will be sampled.
        questions_file (str): Path to the questions file.
    """
    # Create output directories
    os.makedirs(path, exist_ok=True)
    
    questions_output_dir = os.path.join(path, "questions")
    images_output_dir = os.path.join(path, "images", "train")
    scenes_output_dir = os.path.join(path, "scenes")
    
    os.makedirs(questions_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(scenes_output_dir, exist_ok=True)

    # Define spatial relations to track
    relations_dict = defaultdict(list)
    spatial_relations = ["left", "right", "in front of", "behind"]
    
    print(f"Loading questions from {questions_file}")
    with open(questions_file) as f:
        data = json.load(f)
    total_questions = len(data['questions'])
    
    print(f"Total questions: {total_questions}")
    
    # Group questions by relation
    for i, item in enumerate(data['questions']):
        question = item['question'].lower()
        categorized = False
        
        for relation in spatial_relations:
            # Use word boundaries to ensure we're matching complete words/phrases
            pattern = r'\b' + re.escape(relation) + r'\b'
            if re.search(pattern, question):
                relations_dict[relation].append(i)
                categorized = True
                break
                
        if not categorized:
            relations_dict['other'].append(i)
    
    # Calculate statistics before sampling
    pre_sample_stats = {relation: len(indices) for relation, indices in relations_dict.items()}
    print("Pre-sampling statistics:")
    for relation, count in pre_sample_stats.items():
        print(f"  {relation}: {count} questions ({count / total_questions * 100:.2f}%)")
    
    # Perform stratified sampling
    sampled_indices = []
    for relation, indices in relations_dict.items():
        # Take 1/ratio of samples for each relation category (including 'other')
        sampled = random.sample(indices, len(indices) // ratio)
        sampled_indices.extend(sampled)
        print(f"Sampled {len(sampled)} out of {len(indices)} questions with relation '{relation}'")
    
    # Create new questions data
    sampled_questions = {
        "info": data.get("info", {}),
        "questions": [data['questions'][i] for i in sampled_indices]
    }
    
    # Get unique image filenames from sampled questions
    image_filenames = set()
    for question in sampled_questions["questions"]:
        image_filenames.add(question["image_filename"])
    
    print(f"Total sampled questions: {len(sampled_questions['questions'])}")
    print(f"Unique images to copy: {len(image_filenames)}")
    
    # Save sampled questions to file
    output_questions_file = os.path.join(questions_output_dir, os.path.basename(questions_file))
    with open(output_questions_file, 'w') as f:
        json.dump(sampled_questions, f, indent=2)
    print(f"Saved sampled questions to {output_questions_file}")
    
    # Copy associated images
    orig_image_dir = os.path.join(os.path.dirname(os.path.dirname(questions_file)), "images", "train")
    for image_filename in image_filenames:
        src_path = os.path.join(orig_image_dir, image_filename)
        dst_path = os.path.join(images_output_dir, image_filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    print(f"Copied {len(image_filenames)} images to {images_output_dir}")
    
    # Copy scenes file if it exists and filter it to include only sampled images
    scenes_file = os.path.join(os.path.dirname(os.path.dirname(questions_file)), 
                              "scenes", "CLEVR_train_scenes.json")
    if os.path.exists(scenes_file):
        with open(scenes_file, 'r') as f:
            scenes_data = json.load(f)
        
        # Filter scenes to only include those for sampled images
        filtered_scenes = []
        for scene in scenes_data['scenes']:
            if scene['image_filename'] in image_filenames:
                filtered_scenes.append(scene)
        
        # Create new scenes data
        sampled_scenes = {
            "info": scenes_data.get("info", {}),
            "scenes": filtered_scenes
        }
        
        # Save filtered scenes
        output_scenes_file = os.path.join(scenes_output_dir, os.path.basename(scenes_file))
        with open(output_scenes_file, 'w') as f:
            json.dump(sampled_scenes, f, indent=2)
        print(f"Saved filtered scenes to {output_scenes_file}")
    
    for meta_file in ["README.txt", "LICENSE.txt", "COPYRIGHT.txt"]:
        src_path = os.path.join(os.path.dirname(os.path.dirname(questions_file)), meta_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(path, meta_file)
            shutil.copy2(src_path, dst_path)
    
    print(f"Stratified sampling complete. Data saved to {path}")
    return sampled_questions

if __name__ == "__main__":
    random.seed(42)
    
    path = "./data/sampled_CLEVR_v1.0"
    ratio = 5  # 1 out of 5 for each relation category
    questions_file = './data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
    
    stratified_sample(path, ratio, questions_file)
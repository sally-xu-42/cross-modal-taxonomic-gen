import os
import re
import json
import torch
import random

from tqdm import tqdm
from pathlib import Path

# check which line of data we are at
ROOT_DIR = "/share/data/speech/txu/vlm_semantics/data/vision_features/"

def check_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    total = len(data)
    print(f"Total entries in {file_name}: {total}")
    missed = 0
    res = set()
    for i, item in tqdm(enumerate(data), total=total):
        image_path = item['image']
        if os.path.exists(os.path.join(ROOT_DIR, image_path + ".pt")):
            fts = torch.load(os.path.join(ROOT_DIR, image_path + ".pt"), map_location='cuda')
            if fts.shape != (729, 2176):
                print(f"Bad shape at index {i}: {fts.shape} for {image_path}")
                res.add(image_path)
                missed += 1
        else:
            print(f"Missing image at index {i}: {os.path.join(ROOT_DIR, image_path + '.pt')}")
            res.add(image_path)
            missed += 1
    print(f"Total missing images in {file_name}: {missed}")
    with open("./missing_images.txt", 'w') as f:
        for r in res:
            f.write(r + "\n")

def create_training_set(original_file, missing_file):
    with open(original_file, 'r') as f:
        data = json.load(f)
    with open(missing_file, 'r') as f:
        missing_images = set(line.strip() for line in f.readlines())
    
    filtered_data = [item for item in data if item['image'] in missing_images]
    print(f"Filtered data size: {len(filtered_data)} from original size: {len(data)}")
    
    output_file = original_file.replace(".json", "_filtered.json")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f)
    print(f"Filtered data saved to {output_file}")

def clean_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    for item in data:
        question = item['conversations'][0]["value"]
        cleaned = re.sub(r' {2,}', ' ', question).strip()
        print(question, " --> ", cleaned)
        item['conversations'][0]["value"] = cleaned
    with open(file_name, 'w') as f:
        json.dump(data, f)
    print(f"Cleaned data saved to {file_name}")

def sample_data(file_name, sample_ratio=0.1):
    with open(file_name, 'r') as f:
        data = json.load(f)
    total = len(data)
    sample_size = int(total * sample_ratio)
    print(f"Sampling {sample_size} entries from {total} total entries.")
    sampled_data = random.sample(data, sample_size)
    output_file = file_name.replace(".json", f"_sampled_{sample_ratio}.json")
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f)
    print(f"Sampled data saved to {output_file}")

if __name__ == "__main__":
    # check_data("./data/preprocessed_CLEVR/clevr_train_qa_preprocessed.json")
    # create_training_set(ROOT_DIR+"../preprocessed_CLEVR/clevr_train_qa_preprocessed.json", "./missing_images.txt")
    # clean_data("./data/preprocessed_CLEVR/clevr_train_qa_preprocessed.json")
    sample_data("./data/preprocessed_CLEVR/clevr_train_qa_preprocessed.json", sample_ratio=0.1)

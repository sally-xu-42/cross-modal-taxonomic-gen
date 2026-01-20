import os
import re
import json
import torch
import random

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# check which line of data we are at
ROOT_DIR = "/share/data/speech/txu/vlm_semantics/data/vision_features/"

def check_data(file_name): # check if all the images in the dataset have corresponding .pt files
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

def create_training_set(original_file, missing_file): # create a new json file with only the entries that have missing images
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

def clean_data(file_name): # remove extra spaces from questions in the dataset
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

def sample_data(file_name, sample_ratio=0.1): # sample a percentage of the data for quick testing
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

def count_leaf_level():
    file = "./data/preprocessed_THINGS/test_hyp.json"
    with open(file, 'r') as f:
        data = json.load(f)
    concepts = set()
    for item in data:
        image = item['image']
        concept = image.split('/')[0]
        concepts.add(concept)
    print(f"Total unique leaf-level concepts: {len(concepts)}")

def combine_data(): # Run from root directory to ensure paths are correct, combine multiple json files into one
    combined_data = []
    # # file_names = ["./prismatic-vlms/data/download/llava-laion-cc-sbu-558k/chat.json", "./data/preprocessed_THINGS/train.json"]
    # file_names = ["./prismatic-vlms/data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json", "./data/preprocessed_THINGS/train.json"]
    # # LLAVA format: 00223/002239345.jpg, THINGS format: aardvark/aardvark_10s.jpg
    # with open(file_names[0], 'r') as f:
    #     data = json.load(f)
    #     print(f"Loaded {len(data)} entries from {file_names[0]}")
    #     for item in data:
    #         if not item.get('image', None):
    #             print(f"Skipping entry without image: {item}") # Deal with llava-v1.5-instruct entries that may not have an image field
    #             continue
    #         image_path = item['image']
    #         # Convert LLAVA image paths to general formats in my repo
    #         new_image_path = "../../../prismatic-vlms/data/download/llava-v1.5-instruct/" + image_path
    #         item['image'] = new_image_path
    #         combined_data.append(item)
    # with open(file_names[1], 'r') as f:
    #     data = json.load(f)
    #     print(f"Loaded {len(data)} entries from {file_names[1]}")
    #     combined_data.extend(data)
    
    file_names = ["./data/preprocessed_THINGS/test.json", "./data/preprocessed_THINGS/test_hyp.json"]
    # file_names = ["./data/baseline/things+llava-v15_finetune.json", "./data/preprocessed_THINGS/train_hyp.json"]
    for fn in file_names:
        with open(fn, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} entries from {fn}")
            combined_data.extend(data)

    print(f"Total combined entries: {len(combined_data)}")
    output_file = "./data/preprocessed_THINGS/test_things+hyp.json"
    with open(output_file, 'w') as f:
        json.dump(combined_data, f)
    print(f"Combined data saved to {output_file}")

def inspect():
    file_name = "./data/preprocessed_THINGS/train.json"
    with open(file_name, 'r') as f:
        data = json.load(f)
    print(f"Total entries in {file_name}: {len(data)}")
    pos_entries, neg_entries = 0, 0
    for i, item in enumerate(data):
        if item['conversations'][1]['value'].lower() == 'yes':
            pos_entries += 1
        elif item['conversations'][1]['value'].lower() == 'no':
            neg_entries += 1
        else:
            print(f"Unexpected label at index {i}: {item['conversations'][1]['value']}")
    print(f"Positive entries: {pos_entries}")
    print(f"Negative entries: {neg_entries}")

def check_hypernyms():
    file_name = "./data/preprocessed_THINGS/val_hyp.json"
    with open(file_name, 'r') as f:
        data = json.load(f)
    print(f"Total entries in {file_name}: {len(data)}")
    hyps = set([])
    for _, item in enumerate(data):
        question = item['conversations'][0]['value']
        match = re.search(r'Is there (?:a|an) (.+?) in', question, re.IGNORECASE)
        if match:
            hyp = match.group(1).strip()
            hyps.add(hyp)
            continue
        # Pattern 2: Without article
        match = re.search(r'Is there (.+?) in', question, re.IGNORECASE)
        if match:
            hyp = match.group(1).strip()
            hyps.add(hyp)
            continue
        else:
            print(f"Warning: Could not extract hypernym from question: {question}")
            raise ValueError("Hypernym extraction failed.")
    print(f"Extracted {len(hyps)} unique hypernyms:")
    for h in sorted(list(hyps)):
        print(h)

def combine_by_image():
    """Combine conversations from two files by grouping them by image."""
    file1_path = "./data/preprocessed_THINGS/test.json"
    file2_path = "./data/preprocessed_THINGS/test_hyp.json"
    output_path = "./data/preprocessed_THINGS/test_multistep.json"
    with open(file1_path, 'r') as f:
        data1 = json.load(f)
    with open(file2_path, 'r') as f:
        data2 = json.load(f)
    
    print(f"Entries in file1: {len(data1)}")
    print(f"Entries in file2: {len(data2)}")
    image_to_convs = defaultdict(list)

    for item in data1:
        image = item['image']
        image_to_convs[image].extend(item['conversations'])

    for item in data2:
        image = item['image']
        image_to_convs[image].extend(item['conversations'])

    combined_data = []
    for image, conversations in image_to_convs.items():
        # Remove <image>\n from all turns except the first human turn
        cleaned_conversations = []
        first_human = True
        
        for conv in conversations:
            if conv['from'] == 'human':
                if first_human:
                    # Keep <image>\n for first human turn
                    cleaned_conversations.append(conv)
                    first_human = False
                else:
                    # Remove <image>\n for subsequent human turns
                    value = conv['value'].replace('<image>\n', '')
                    cleaned_conversations.append({
                        'from': conv['from'],
                        'value': value
                    })
            else:
                cleaned_conversations.append(conv)
        
        combined_data.append({
            'image': image,
            'conversations': cleaned_conversations
        })
    
    print(f"Total images: {len(combined_data)}")
    print(f"Average conversations per image: {sum(len(item['conversations']) for item in combined_data) / len(combined_data):.1f}")
    
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)
    
    print(f"Combined data saved to {output_path}")


if __name__ == "__main__":
    # python3 ./check_data.py
    # check_data("./data/preprocessed_CLEVR/clevr_train_qa_preprocessed.json")
    # create_training_set(ROOT_DIR+"../preprocessed_CLEVR/clevr_train_qa_preprocessed.json", "./missing_images.txt")
    # clean_data("./data/preprocessed_CLEVR/clevr_val_qa_preprocessed.json")
    # sample_data("./data/preprocessed_CLEVR/clevr_train_qa_preprocessed.json", sample_ratio=0.1)
    # combine_data()
    # inspect()
    # check_hypernyms()
    # combine_by_image()
    count_leaf_level()

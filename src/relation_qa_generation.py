import json
import re
import random

from collections import defaultdict, Counter
from dataclasses import dataclass
from ordered_set import OrderedSet
from PIL import Image

from typing import List, Union, Iterable

RELATION_TYPES = ['left', 'right', 'front', 'behind']
RELATION_PHRASES = {
    'left': 'to the left of',
    'right': 'to the right of',
    'front': 'in front of',
    'behind': 'behind'
}

@dataclass
class CLEVRObject:
    '''Defines a CLEVR Object and provides helper functions that allow us to perform modification of the NP, list its unique features, etc.'''
    size: str
    color: str
    material: str
    shape: str
    
    def __post_init__(self):
        # these are listed in this manner so that we can convert from features to object and back.
        self.modifying_features = OrderedSet([f"size:{self.size}", f"color:{self.color}", f"material:{self.material}"])
        self.modifiers = OrderedSet([self.size, self.color, self.material])

    def modify(self, shape_only=False):
        if shape_only:
            return self.shape
        else:
            modifiers = ' '.join(self.modifiers)
            noun_phrase = f"{modifiers} {self.shape}"
            noun_phrase = re.sub(r'\s\s+', ' ', noun_phrase)
            return noun_phrase

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def features2obj(features: OrderedSet, shape: str):
    """Parse object given features."""
    feats = defaultdict(lambda : "")
    for feat in features:
        attribute, value = feat.split(":")
        feats[attribute] = value
    return CLEVRObject(feats['size'], feats['color'], feats['material'], shape)

def distinguishing_features(lst: List[Iterable]):
    """Overthought and overengineered function to compute uniquely identifiable features of entries in the input."""
    unique_ids = [item - OrderedSet.intersection(*lst) for item in lst]
    for j, item in enumerate(unique_ids):
        leftover = [entry for i, entry in enumerate(unique_ids) if i != j]
        for entry in leftover:
            if item.issubset(entry):
                return False

    return unique_ids

def count_relationship_pairs(metadata, relation_type='right'):
    """Count achievable question pairs for a given relation type across all scenes."""
    total_positive, total_negative = 0, 0
    
    for scene in metadata['scenes']:
        for type, relationships in scene['relationships'].items():
            if type == relation_type: # skip other relation types
                for entry in relationships:
                    total_positive += len(entry)
                    total_negative += len(entry)
    
    return total_positive, total_negative

def unique_identifier_objects(scene_objects):
    """Return a list of strings that can be uniquely identify each object."""
    objects = []
    shapes = defaultdict(list)
    for i, obj in enumerate(scene_objects):
        obj = CLEVRObject(obj['size'], obj['color'], obj['material'], obj['shape'])
        objects.append(obj)
        shapes[obj.shape].append((obj, i))

    object_strings = ["" for _ in objects]
    
    for shape, entries in shapes.items():
        objs, indices = list(zip(*entries))
        if len(entries) > 1:
            dfs = distinguishing_features([o.modifying_features for o in objs])
            if not dfs:
                for idx in indices:
                    object_strings[idx] = "FAILURE"
            else:
                for obj, idx in zip(dfs, indices):
                    object_strings[idx] = features2obj(obj, shape).modify()
        else:
            object_strings[indices[0]] = objs[0].modify(shape_only=True)
    return object_strings

def generate_questions_for_scene(scene_objects, scene_relationships, sample_rate=0.5, balance_pos_neg=True):
    """ Returns a list of question dictionaries for a single scene with random sampling. """
    questions, positive_pairs, negative_pairs = [], [], []
    # 1. Uniquely identify objects
    object_strings = unique_identifier_objects(scene_objects)
    if "FAILURE" in object_strings:
        return []
    # 2. Collect pairs
    for rel_type in RELATION_TYPES:
        relationships = scene_relationships[rel_type]
        for idx, entry in enumerate(relationships):
            for j in entry:
                positive_pairs.append((j, idx, rel_type, True))  # (subject_idx, object_idx, relation, is_positive)
                negative_pairs.append((idx, j, rel_type, False))  # Swapped for negative samples: (idx, j)
    
    # 3. Sampling    
    if balance_pos_neg:
        n_pos_sample = int(len(positive_pairs) * sample_rate)
        n_neg_sample = n_pos_sample
        sampled_positive = random.sample(positive_pairs, min(n_pos_sample, len(positive_pairs)))
        sampled_negative = random.sample(negative_pairs, min(n_neg_sample, len(negative_pairs)))
        sampled_pairs = sampled_positive + sampled_negative
    else:
        all_pairs = positive_pairs + negative_pairs
        n_sample = int(len(all_pairs) * sample_rate)
        sampled_pairs = random.sample(all_pairs, min(n_sample, len(all_pairs)))
        
    # 4. Generation
    for subject_idx, object_idx, relation, is_positive in sampled_pairs:
        question_text = f"Is the {object_strings[subject_idx]} {RELATION_PHRASES[relation]} the {object_strings[object_idx]}?"
        answer = "Yes" if is_positive else "No"
        questions.append({
            'question': question_text,
            'answer': answer,
            'relation_type': relation
        })
    
    return questions

def report_total_pairs(metadata):
    """ Count total pairs of relationships in the dataset. """
    total_counts = {}
    for rel_type in RELATION_TYPES:
        pos, neg = count_relationship_pairs(metadata, rel_type)
        total_counts[rel_type] = {'positive': pos, 'negative': neg}
        print(f"{rel_type.capitalize()} relations:")
        print(f"  Positive/Negative pairs: {pos}")

    total_positive = sum(counts['positive'] for counts in total_counts.values())
    total_negative = sum(counts['negative'] for counts in total_counts.values())
    print(f"OVERALL TOTALS:")
    print(f"Grand total pairs: {total_positive + total_negative}")

def generate_dataset(metadata, sample_rate=0.1, balance_pos_neg=True, random_seed=42):
    """ Generate a sampled dataset of relationship questions from all validation scenes. """
    random.seed(random_seed)
    all_questions = []
    failed_scenes = 0
    
    for scene_idx, scene in enumerate(metadata['scenes']):
        scene_questions = generate_questions_for_scene(
            scene['objects'], 
            scene['relationships'],
            sample_rate=sample_rate,
            balance_pos_neg=balance_pos_neg
        )
        
        if not scene_questions:  # Object identification failed
            failed_scenes += 1
            continue
        
        for question in scene_questions:
            question['scene_idx'] = scene_idx
            question['image_filename'] = f"CLEVR_train_{scene_idx:06d}.png"
        
        all_questions.extend(scene_questions)
        
        if (scene_idx + 1) % 1000 == 0:
            print(f"Processed {scene_idx + 1} scenes...")
    
    print(f"\nDataset generation complete!")
    print(f"Total scenes processed: {len(metadata['scenes'])}")
    print(f"Failed scenes (object identification issues): {failed_scenes}")
    print(f"Successful scenes: {len(metadata['scenes']) - failed_scenes}")
    print(f"Total questions generated: {len(all_questions)}")
    
    return all_questions

if __name__ == "__main__":
    metadata_path = "../data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
    metadata = read_json(metadata_path)
    report_total_pairs(metadata)

    questions = generate_dataset(metadata, sample_rate=0.1, balance_pos_neg=False)
    output_path = "../data/CLEVR_v1.0/scenes/CLEVR_train_qa.json"
    write_json(output_path, questions)
    
    print(f"Generated questions saved to {output_path}")
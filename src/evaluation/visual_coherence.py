"""
visual_coherence.py

Compute average cosine similarity between DINOv2 image embeddings for members
of the same category across different shuffle types (original, shuffled, local_shuffled).
"""

import json
import os
import sys
import torch
import timm
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
from torchvision.transforms import Compose, Resize

# Paths
DATA_DIR = Path("./data")
HYPERNYMY_DIR = DATA_DIR / "hypernymy_THINGS"
IMAGES_DIR = HYPERNYMY_DIR / "images"

# Mapping files
HYP_TO_CONCEPTS = HYPERNYMY_DIR / "train_hyp_to_concepts.json"
CONCEPTS_SHUFFLED = HYPERNYMY_DIR / "concepts_shuffled.json"
CONCEPTS_LOCAL_SHUFFLED = HYPERNYMY_DIR / "concepts_shuffled_within_category.json"

# Training split ratio
TRAIN_SPLIT = 0.7


def load_dinov2_model_and_transform(device="cuda"):
    """Load DINOv2 model and image transform matching training setup."""
    # Load model (same as training)
    model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        img_size=224
    )
    model = model.to(device)
    model.eval()
    
    # Get transform config (same as training)
    data_cfg = timm.data.resolve_model_data_config(model)
    data_cfg["input_size"] = (3, 224, 224)
    
    # Create transform (resize-naive strategy, same as training)
    default_transform = timm.data.create_transform(**data_cfg, is_training=False)
    
    # Apply resize-naive strategy (same as TimmViTBackbone)
    assert isinstance(default_transform, Compose)
    assert isinstance(default_transform.transforms[0], Resize)
    target_size = (224, 224)
    transform = Compose([
        Resize(target_size, interpolation=default_transform.transforms[0].interpolation),
        *default_transform.transforms[1:],
    ])
    
    return model, transform


def get_image_embedding(model, transform, image_path, device="cuda"):
    """Get DINOv2 embedding for a single image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            # Get features: shape [batch, num_tokens, dim] = [1, 261, 1024]
            features = model.forward_features(img_tensor)
            
            # Mean pooling over all tokens (including CLS token)
            # features.mean(dim=1) gives [1, 1024], then squeeze to [1024]
            embedding = features.mean(dim=1).squeeze(0).cpu().numpy()
            
            return embedding
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def get_training_images(concept, shuffle_map=None):
    """
    Get training split images for a concept.
    
    Args:
        concept: Concept name (may be shuffled)
        shuffle_map: If provided, maps original concept to shuffled concept
    
    Returns:
        List of image paths in training split
    """
    # If shuffled, get the actual concept directory to read from
    actual_concept = shuffle_map[concept] if shuffle_map and concept in shuffle_map else concept
    
    concept_dir = IMAGES_DIR / actual_concept
    if not concept_dir.exists():
        return []
    
    # Get all image files
    image_files = [f for f in os.listdir(concept_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    # Get training split (first TRAIN_SPLIT ratio)
    n_train = int(len(image_files) * TRAIN_SPLIT)
    train_files = image_files[:n_train]
    
    return [concept_dir / f for f in train_files]


def compute_category_similarity(embeddings):
    """Compute average pairwise cosine similarity."""
    if len(embeddings) < 2:
        print(f"Less than 2 embeddings, skipping")
        return None
    
    emb_matrix = np.stack(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix_norm = emb_matrix / norms
    # Compute pairwise cosine similarities (dot product of normalized vectors)
    similarities = np.dot(emb_matrix_norm, emb_matrix_norm.T)
    
    # Upper triangle (excluding diagonal)
    n = len(similarities)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    pairwise_sims = similarities[mask]
    
    return np.mean(pairwise_sims)


def process_shuffle_type(shuffle_type, hyp_to_concepts, shuffle_map=None, device="cuda"):
    """Process one shuffle type."""
    print(f"\nProcessing {shuffle_type}...")
    
    # Load model and transform
    print("Loading DINOv2 model...")
    model, transform = load_dinov2_model_and_transform(device=device)
    
    results = []
    
    # Process each category
    for category, concepts in tqdm(hyp_to_concepts.items(), desc="Categories"):
        # Collect all training images for this category
        all_embeddings = []
        
        for concept in concepts:
            # Get training images (accounting for shuffle)
            image_paths = get_training_images(concept, shuffle_map=shuffle_map)
            
            # Compute embeddings
            for img_path in image_paths:
                emb = get_image_embedding(model, transform, img_path, device=device)
                if emb is not None:
                    all_embeddings.append(emb)
        
        if len(all_embeddings) < 2:
            print(f"{category}: < 2 images, skipping")
            continue
        
        # Compute average similarity
        avg_sim = compute_category_similarity(all_embeddings)
        if avg_sim is not None:
            results.append({
                "shuffle_type": shuffle_type,
                "category": category,
                "avg_cosine": avg_sim,
                "n_images": len(all_embeddings)
            })
            print(f"{category}: {avg_sim:.2f} (n={len(all_embeddings)})")
    
    return results



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load mappings
    print("Loading mappings...")
    with open(HYP_TO_CONCEPTS, "r") as f:
        hyp_to_concepts = json.load(f)
    
    with open(CONCEPTS_SHUFFLED, "r") as f:
        concepts_shuffled = json.load(f)
    
    with open(CONCEPTS_LOCAL_SHUFFLED, "r") as f:
        concepts_local_shuffled = json.load(f)
    
    # Process each shuffle type
    all_results = []
    
    # Original
    results_original = process_shuffle_type(
        "original", hyp_to_concepts, shuffle_map=None, device=device
    )
    all_results.extend(results_original)
    
    # Shuffled
    results_shuffled = process_shuffle_type(
        "shuffled", hyp_to_concepts, shuffle_map=concepts_shuffled, device=device
    )
    all_results.extend(results_shuffled)
    
    # Local shuffled
    results_local_shuffled = process_shuffle_type(
        "local_shuffled", hyp_to_concepts, shuffle_map=concepts_local_shuffled, device=device
    )
    all_results.extend(results_local_shuffled)
    
    # Save results
    df = pd.DataFrame(all_results)
    output_file = Path("./src/plotting/visual_coherence_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Summary
    print("\nSummary:")
    print(df.groupby("shuffle_type")["avg_cosine"].agg(["mean", "std", "count"]))
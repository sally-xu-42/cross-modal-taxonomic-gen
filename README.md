# Cross-Modal Taxonomic Generalization in (Vision) Language Models

This repository contains code for training and evaluating Vision Language Models (VLMs) on compositional semantic tasks, specifically focusing on hypernymy relations.

## Overview

- **Hypernymy Relations**: Understanding hierarchical relationships between concepts (e.g., "Is there a bird in the image?" where "bird" is a hypernym of specific species)
- **Concept Ablation**: Studying how models generalize to unseen concepts through controlled ablation experiments

The codebase is built on top of [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms), a framework for training visually-conditioned language models.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.1+ (with CUDA support)
- CUDA-capable GPU(s)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd vlm_semantics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up HuggingFace token (if needed for accessing gated models):
```bash
echo "your_hf_token_here" > .hf_token
```

## Dataset Preparation

### THINGS Dataset

The THINGS dataset contains images organized by hypernym categories. To prepare the data:

1. Download the THINGS dataset and place images in `data/hypernymy_THINGS/images/`
2. Preprocess the dataset:
```bash
# Preprocessing scripts will generate JSON files in data/preprocessed_THINGS/
```

## Usage

### Training

Train a model using the provided script:

```bash
./scripts/train.sh \
  -s 42 \                    # Random seed
  -d things+hyp-abl30 \      # Dataset ID
  -m siglip+1b \             # Model architecture
  -t align                   # Trains only the projector (vision-text alignment)
```

### Evaluation

```bash
# Evaluate a trained model
./scripts/eval.sh \
  -m ./runs/align-dinosiglip+1b-things-42 \  # Model path
  -d ./data/preprocessed_THINGS/test.json \  # Dataset path
  -o test                                     # Output suffix
```

### Ablation Studies

#### Concept Ablation

Remove specific concepts from training to test generalization:

```bash
# Ablate 30% of concepts from training set
python src/ablation/ablate_hyp.py --ratio 0.3
```

#### Category Ablation

Remove entire hypernym categories:

```bash
# Ablate 10 categories
python src/ablation/ablate_cat.py --n_categories 10
```

#### Shuffled Ablation

Test models with shuffled hypernym-concept mappings:

```bash
# Create shuffled mappings
python src/shuffle/mislabel.py

# Run ablation with shuffled data
python src/ablation/ablate_hyp_shuffle.py --ratio 0.3
```

### Visual Coherence Analysis

Measure visual coherence of concepts within categories:

```bash
./scripts/run_visual_coherence.sh
```

Or directly:

```bash
python src/evaluation/visual_coherence.py
```

### Language Model Evaluation

Evaluate language-only models (without vision):

```bash
```

## Configuration

### Model Configurations

Models are configured using the Prismatic framework. Common model IDs include:
- `dinov2+1b`: DINOv2 vision encoder, Qwen3-1.7B LM backbone
- `dinov2+1b-llama`: DINOv2 vision encoder, Llama-3.2-1.7b LM backbone

### Dataset Configurations

- `things+hyp`: THINGS dataset with hypernymy questions
- `things+hyp-abl30`: THINGS dataset with 30% concepts ablated

## Results

Evaluation results are saved as CSV files with the following metrics:
- Overall accuracy
- Per-category accuracy
- Confusion matrices
- Illegal answer rates

## Citation

If you use this code in your research, please cite:

```bibtex
```

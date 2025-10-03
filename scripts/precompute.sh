#!/bin/bash
#SBATCH --partition=speech-gpu
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --array=0-13
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --time=07:55:00
#SBATCH --signal=SIGHUP@600
#SBATCH -o /share/data/speech/txu/vlm_semantics/logs/precompute_array_%A_%a.log

# Set environment variables
export HF_HOME="/share/data/speech/txu/cache"
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism for PyTorch DataLoaders

# Activate the virtual environment
cd /share/data/speech/txu/vlm_semantics
source /share/data/speech/txu/vlm_semantics/venv/bin/activate

python prismatic-vlms/scripts/precompute_visual_rep.py \
    --dataset.type "clevr" \
    --model.type "prism-dinosiglip+7b" \
    --chunk_size 10000

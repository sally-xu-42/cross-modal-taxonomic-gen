#!/bin/bash
#SBATCH --partition=speech-gpu
#SBATCH --cpus-per-task=20
#SBATCH --gpus=nvidia_rtx_a6000:4
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --time=07:55:00
#SBATCH --signal=SIGHUP@600
#SBATCH -o /share/data/speech/txu/vlm_semantics/logs/precompute_%j.out

# Set environment variables
export HF_HOME="/share/data/speech/txu/cache"
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism for PyTorch DataLoaders

# Activate the virtual environment
cd /share/data/speech/txu/vlm_semantics
source /share/data/speech/txu/vlm_semantics/venv/bin/activate

ARGS=(
    --dataset.type "clevr"
    --model.type "prism-dinosiglip+7b"
)

echo "Starting precomputation"

torchrun --standalone --nnodes 1 --nproc-per-node 4 prismatic-vlms/scripts/precompute_visual_rep.py "${ARGS[@]}"

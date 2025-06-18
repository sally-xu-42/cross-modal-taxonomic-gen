#!/bin/bash
#SBATCH --job-name=prismatic
#SBATCH --partition=speech-gpu
#SBATCH --cpus-per-task=2
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --time=07:55:00
#SBATCH --signal=SIGHUP@600
#SBATCH -o /share/data/speech/txu/vlm_semantics/logs/eval_%j.out

# Set environment variables
export HF_HOME="/share/data/speech/txu/cache"
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism for PyTorch DataLoaders
export WANDB__SERVICE_WAIT=300

# Define experiment parameters
TIMESTAMP=$(date +%s)
MODEL_PATH="./runs/train-clevr-align-42"  # Default model ID
DATASET_PATH="./data/simple_clevr_val_preprocessed.json"
IMAGE_DIR="./data/CLEVR_v1.0/images/val"
OUTPUT_PATH="./results/evaluation_${TIMESTAMP}.csv"

cd /share/data/speech/txu/vlm_semantics
source /share/data/speech/txu/vlm_semantics/venv/bin/activate

# Run evaluation script
python src/eval.py \
    --model_path ${MODEL_PATH} \
    --dataset_path ${DATASET_PATH} \
    --image_dir ${IMAGE_DIR} \
    --output_path ${OUTPUT_PATH}

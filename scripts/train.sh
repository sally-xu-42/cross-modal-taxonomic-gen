#!/bin/bash
#SBATCH --job-name=prismatic
#SBATCH --partition=speech-gpu
#SBATCH --cpus-per-task=2
#SBATCH --gpus=nvidia_rtx_a6000:4
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --time=07:55:00
#SBATCH --signal=SIGHUP@600
#SBATCH -o /share/data/speech/txu/vlm_semantics/logs/training_%j.out

# Set environment variables
export HF_HOME="/share/data/speech/txu/cache"  # Optional: Set Hugging Face cache directory
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism for PyTorch DataLoaders
export WANDB__SERVICE_WAIT=300

# Define experiment parameters
RUN_ID="train-clevr-align-$(date +%s)"  # Unique run ID with timestamp
RUN_ROOT_DIR="/share/data/speech/txu/vlm_semantics/runs"
SEED=42
STAGE="align"  # Training stage: align (projector-only)
DATASET_ID="clevr"
MODEL_ID="prism-dinosiglip+7b"  # Vision-Language Model ID

cd /share/data/speech/txu/vlm_semantics
# Activate the virtual environment
source /share/data/speech/txu/vlm_semantics/venv/bin/activate

# Run training with torchrun
torchrun --standalone --nnodes 1 --nproc-per-node 4 prismatic-vlms/scripts/pretrain.py \
    --run_id $RUN_ID \
    --run_root_dir $RUN_ROOT_DIR \
    --seed $SEED \
    --stage $STAGE \
    --dataset.type $DATASET_ID \
    --model.type $MODEL_ID \
    --model.align_epochs 1 \
    --model.align_learning_rate 1e-5 \
    --model.align_global_batch_size 64 \
    --model.align_per_device_batch_size 16

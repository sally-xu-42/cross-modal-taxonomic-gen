#!/bin/bash
#SBATCH --job-name=prismatic
#SBATCH --partition=speech-gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --time=03:55:00
#SBATCH -o /share/data/speech/txu/vlm_semantics/slurm-%x.out

# Activate the virtual environment
source /share/data/speech/txu/vlm_semantics/venv/bin/activate
# Run the inference script
python /share/data/speech/txu/vlm_semantics/src/try_prismatic.py

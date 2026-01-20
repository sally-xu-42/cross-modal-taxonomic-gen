#!/bin/bash
#SBATCH --job-name=check
#SBATCH --cpus-per-task=10
#SBATCH --partition=greg-gpu,speech-gpu
#SBATCH --gpus=1
#SBATCH --constraint=48g
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --mail-type=ALL
#SBATCH -o /share/data/speech/txu/vlm_semantics/logs/check_%j.out

cd /share/data/speech/txu/vlm_semantics
source /share/data/speech/txu/vlm_semantics/venv/bin/activate

python3 ./utils/check_data.py
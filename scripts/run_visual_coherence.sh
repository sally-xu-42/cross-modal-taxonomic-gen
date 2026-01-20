#!/bin/bash
#SBATCH --job-name=visual_coherence
#SBATCH --partition=greg-gpu,speech-gpu
#SBATCH --cpus-per-task=4
#SBATCH -G1
#SBATCH --exclude=g6
#SBATCH --constraint=48g
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00
#SBATCH --signal=SIGHUP@600
#SBATCH -o /share/data/speech/txu/vlm_semantics/logs/visual_coherence_%j.out

source /share/data/speech/txu/vlm_semantics/venv/bin/activate
cd /share/data/speech/txu/vlm_semantics

python src/evaluation/visual_coherence.py
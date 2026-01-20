#!/bin/bash

set -e

MODELS=(
  "siglip+500m"
  "siglip+1b"
  # "dinov2+1b-llama"
  # "dinov2+1b-llama-chat"
)

# ABLATIONS=(40 30 20 10)
# ABLATIONS=(90 70 50 30 10)
SEEDS=(42 218 7)
SCRIPT="./scripts/train.sh"

for MODEL in "${MODELS[@]}"; do
  for ABL in "${ABLATIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      $SCRIPT -m "$MODEL" -t align -d "things+hyp-abl${ABL}cat" -s "$SEED"
      echo ">>> Running train for ${MODEL}-things+hyp-abl${ABL}cat-${SEED}"
    done
  done
done

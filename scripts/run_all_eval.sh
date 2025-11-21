#!/bin/bash

set -e

MODELS=(
  # "align-dinov2+500m"
  # "align-dinov2+1b"
  # "align-dinov2+1b-llama"
  "align-dinov2+1b-llama-chat"
)

ABLATIONS=(40 30 20 10)
SEEDS=(42 218 7)
OUTPUTS=(leaf seen unseen)

DATA_DIR="./data/preprocessed_THINGS"
SCRIPT="./scripts/eval.sh"

for MODEL in "${MODELS[@]}"; do
  for ABL in "${ABLATIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      RUN="./runs/${MODEL}-things+hyp-abl${ABL}cat-${SEED}"

      for OUT in "${OUTPUTS[@]}"; do

        if [[ "$OUT" == "leaf" ]]; then
          DATA="${DATA_DIR}/test.json"
        elif [[ "$OUT" == "seen" ]]; then
          DATA="${DATA_DIR}/test_hyp_trained_${ABL}cat.json"
        elif [[ "$OUT" == "unseen" ]]; then
          DATA="${DATA_DIR}/test_hyp_ablated_${ABL}cat.json"
        fi

        echo ">>> Running eval for ${RUN} | ${OUT}"
        $SCRIPT -m "$RUN" -d "$DATA" -o "$OUT"

      done
    done
  done
done

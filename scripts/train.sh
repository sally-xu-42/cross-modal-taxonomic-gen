#!/bin/bash
# example usage:
# ./scripts/train-sh -s 42 -d clevr -m instruct-dinosiglip+7b -t align -c model_path

seed=42
stage="finetune"
# model_id="prism-dinosiglip+7b"
# dataset_id="clevr"
timestamp=$(date +%s)

dataset_id="llava-v15"
model_id="reproduction-llava-v15+7b-lora"

Help()
{
   # Display Help
   echo "Script to run model training."
   echo
   echo "Syntax: train.sh [-s|c|d|m]"
   echo "options:"
   echo "s     Random seed number. Default: 42"
   echo "c     Checkpoint path to resume training. Default: None"
   echo "d     Dataset id to train on. Default: clevr"
   echo "m     Model architecture id. Default: prism-clip+7b"
   echo "t     Stage of training. Default: align"
   echo "r     Whether to reset epoch/step counters when starting a new stage. Default: False"

   echo
}

while getopts "s:c:d:m:t:r:h" option; do
  case $option in
    s)
      seed="$OPTARG"
      ;;
    c)
      checkpoint="$OPTARG"
      ;;
    d)
      dataset_id="$OPTARG"
      ;;
    m)
      model_id="$OPTARG"
      ;;
    t)
      stage="$OPTARG"
      ;;
    r)
      reset_for_new_stage="$OPTARG"
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-s seed] [-c checkpoint] [-d dataset_id] [-m model_id] [-t stage] [-r reset_for_new_stage] [-h]"
      exit 1
      ;;
  esac
done

echo
echo "Seed: $seed"
echo "Checkpoint: $checkpoint"
echo "Dataset ID: $dataset_id"
echo "Stage: $stage"
echo "Model ID: $model_id"
echo "Reset for new stage: $reset_for_new_stage"

echo 'Doing resumable training'
SEED=${seed} CHECKPOINT=${checkpoint} DATASET_ID=${dataset_id} MODEL_ID=${model_id} STAGE=${stage} RESET_FOR_NEW_STAGE=${reset_for_new_stage}  \
sbatch scripts/train.beehive

echo

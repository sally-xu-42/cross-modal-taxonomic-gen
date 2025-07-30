#!/bin/bash

seed=42
stage="align"
model_id="prism-dinosiglip+7b"
dataset_id="clevr"
timestamp=$(date +%s)

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

   echo
}

while getopts "s:c:d:m:t:h" option; do
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
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-s seed] [-c checkpoint] [-d dataset_id] [-m model_id] [-t stage] [-h]"
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

echo 'Doing resumable training'
SEED=${seed} CHECKPOINT=${checkpoint} DATASET_ID=${dataset_id} MODEL_ID=${model_id} STAGE=${stage} \
sbatch scripts/train.beehive

echo

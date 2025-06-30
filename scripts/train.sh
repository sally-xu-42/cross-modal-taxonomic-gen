#!/bin/bash

seed=42
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to run model training."
   echo
   echo "Syntax: train.sh [-s|c|d]"
   echo "options:"
   echo "s     Random seed number. Default: 42"
   echo "c     Checkpoint path to resume training. Default: None"
   echo "d     Dataset id to train on. Default: clevr"

   echo
}

while getopts "s:c:d:h" option; do
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
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-s seed] [-c checkpoint] [-d dataset_id]"
      exit 1
      ;;
  esac
done

echo
echo "Seed: $seed"
echo "Checkpoint: $checkpoint"
echo "Dataset ID: ${dataset_id:-clevr}"

echo 'Doing resumable training'
SEED=${seed} CHECKPOINT=${checkpoint} DATASET_ID=${dataset_id} \
sbatch scripts/train.beehive

echo
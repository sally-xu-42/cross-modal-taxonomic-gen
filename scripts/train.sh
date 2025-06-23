#!/bin/bash

seed=42
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to run model training."
   echo
   echo "Syntax: train.sh [-s|c]"
   echo "options:"
   echo "s     Random seed number. Default: 42"
   echo "c     Checkpoint path to resume training. Default: None"

   echo
}

while getopts "s:c:h" option; do
  case $option in
    s)
      seed="$OPTARG"
      ;;
    c)
      checkpoint="$OPTARG"
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-s seed] [-c checkpoint]"
      exit 1
      ;;
  esac
done

echo
echo "Seed: $seed"
echo "Checkpoint: $checkpoint"

echo 'Doing resumable training'
SEED=${seed} CHECKPOINT=${checkpoint} \
sbatch scripts/train.beehive

echo
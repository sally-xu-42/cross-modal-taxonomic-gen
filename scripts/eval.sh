#!/bin/bash

Help()
{
   # Display Help
   echo "Script to run model training."
   echo
   echo "Syntax: train.sh [-m]"
   echo "options:"
   echo "m     Model architecture id or local path. Default: prism-clip+7b"

   echo
}

while getopts "m:h" option; do
  case $option in
    m)
      model_path="$OPTARG"
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-m model_path]"
      exit 1
      ;;
  esac
done

echo
echo "Model PATH: $model_path"

echo 'Running evaluation on CLEVR auto-generated val split'
MODEL_PATH=${model_path} \
sbatch scripts/eval.beehive

echo

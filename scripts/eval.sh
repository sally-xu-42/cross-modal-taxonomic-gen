#!/bin/bash

task="ppl"
model_path="./runs/instruct-dinosiglip+7b-clevr-42"
dataset_path="./data/preprocessed_CLEVR/clevr_val_qa_preprocessed.json"
image_dir="./data/CLEVR_v1.0/images"

Help()
{
   # Display Help
   echo "Script to run model evaluation."
   echo
   echo "Syntax: eval.sh [-t|m]"
   echo "options:"
   echo "t     Evaluation task. Default: ppl"
   echo "m     Model architecture id or local path. Default: prism-clip+7b"

   echo
}

while getopts "t:m:h" option; do
  case $option in
    t)
      task="$OPTARG"
      ;;
    m)
      model_path="$OPTARG"
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-t ppl -m model_path]"
      exit 1
      ;;
  esac
done

output_path="./results/evaluation_${model_path##*/}.csv"

echo
echo "Task: $task"
echo "Model PATH: $model_path"
echo "Output PATH: $output_path"

echo 'Running evaluation on CLEVR auto-generated val split'
TASK=${task} MODEL_PATH=${model_path} DATASET_PATH=${dataset_path} \
IMAGE_DIR=${image_dir} OUTPUT_PATH=${output_path} \
sbatch scripts/eval.beehive

echo

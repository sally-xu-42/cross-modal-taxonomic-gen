#!/bin/bash

task="ppl"
model_path="./runs/align-dinosiglip+1b-things-218"
dataset_path="./data/preprocessed_THINGS/test.json"
image_dir="./data/hypernymy_THINGS/images"

Help()
{
   # Display Help
   echo "Script to run model evaluation."
   echo
   echo "Syntax: eval.sh [-t|d|m|o]"
   echo "options:"
   echo "t     Evaluation task. Default: ppl"
   echo "d     Dataset path. Default: ./data/preprocessed_THINGS/test.json"
   echo "m     Model architecture id or local path."
   echo "o     Output path. Default: <model_path>/eval.csv"

   echo
}

while getopts "t:d:m:o:h" option; do
  case $option in
    t)
      task="$OPTARG"
      ;;
    d)
      dataset_path="$OPTARG"
      ;;
    m)
      model_path="$OPTARG"
      ;;
    o)
      output_base="$OPTARG"
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

output_path="${model_path}/eval_${output_base}.csv"
# output_path="./eval_${output_base}.csv"

echo
echo "Task: $task"
echo "Model PATH: $model_path"
echo "Dataset PATH: $dataset_path"
echo "Output PATH: $output_path"

echo 'Running evaluation on test split'

TASK=${task} MODEL_PATH=${model_path} DATASET_PATH=${dataset_path} \
IMAGE_DIR=${image_dir} OUTPUT_PATH=${output_path} \
sbatch scripts/eval.beehive

echo

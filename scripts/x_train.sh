#!/bin/bash
#SBATCH --job-name=prismatic
#SBATCH --partition=speech-gpu
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --output=slurm/logs/%j.out

# Default values
CONFIG="configs/clip_linear.yml"
IMAGE_URL=""
PROMPT="A picture of"
TEMPERATURE=0.7
TOP_P=0.9
MAX_STEPS=100

Help()
{
   # Display Help
   echo "Script to run LIMBER model inference with PyTorch Lightning."
   echo
   echo "Syntax: bash run_limber_lightning.sh [-c|i|p|t|s|h]"
   echo "options:"
   echo "c     Config path. Default: configs/clip_linear.yml"
   echo "i     Path or URL to input image (REQUIRED)"
   echo "p     Text prompt. Default: 'A picture of'"
   echo "t     Temperature for generation. Default: 0.7"
   echo "n     Top-p sampling parameter. Default: 0.9"
   echo "s     Maximum generation steps. Default: 100"
   echo "h     Display this help message"
   echo
}

while getopts "c:i:p:t:n:s:h" option; do
  case $option in
    c)
      CONFIG="$OPTARG"
      ;;
    i)
      IMAGE_URL="$OPTARG"
      ;;
    p)
      PROMPT="$OPTARG"
      ;;
    t)
      TEMPERATURE="$OPTARG"
      ;;
    n)
      TOP_P="$OPTARG"
      ;;
    s)
      MAX_STEPS="$OPTARG"
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-c config_path] [-i image_url] [-p prompt] [-t temperature] [-n top_p] [-s max_steps]"
      exit 1
      ;;
  esac
done

# Check for required parameters
if [ -z "$IMAGE_URL" ]; then
  echo "Error: Image URL or path (-i) is required"
  Help
  exit 1
fi

echo
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Image: $IMAGE_URL"
echo "  Prompt: $PROMPT"
echo "  Temperature: $TEMPERATURE"
echo "  Top-p: $TOP_P"
echo "  Max steps: $MAX_STEPS"
echo

# Create output directory if it doesn't exist
mkdir -p slurm/logs

# Run the inference script
python run_limber_lightning.py \
  --config "$CONFIG" \
  --image "$IMAGE_URL" \
  --prompt "$PROMPT" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_steps "$MAX_STEPS"
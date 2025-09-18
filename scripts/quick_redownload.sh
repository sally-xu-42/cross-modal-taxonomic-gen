#!/bin/bash
#SBATCH --job-name=quick_redownload
#SBATCH --partition=speech-gpu
#SBATCH --cpus-per-task=20
#SBATCH --gpus=nvidia_rtx_a4000:1
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --time=07:55:00
#SBATCH --signal=SIGHUP@600
#SBATCH -o /share/data/speech/txu/vlm_semantics/logs/quick_redownload_%j.out

# Quick script to remove and re-download LLaVA data
# Usage: ./scripts/quick_redownload.sh

set -e

echo "🚀 Starting LLaVA data removal and re-download..."

# Configuration
DATA_ROOT="/share/data/speech/txu/vlm_semantics/prismatic-vlms/data"
DATASET_ID="llava-v1.5-instruct"
DOWNLOAD_DIR="${DATA_ROOT}/download/${DATASET_ID}"

echo "📁 Data root: $DATA_ROOT"
echo "📦 Dataset: $DATASET_ID"
echo "📂 Download dir: $DOWNLOAD_DIR"

# Step 1: Safely remove existing data
echo ""
# echo "🗑️  Step 1: Removing existing data..."

# if [ -d "$DOWNLOAD_DIR" ]; then
#    echo "   Found existing directory, removing safely..."
    
#    # Use find to delete files (safer than rm -r)
#    echo "   Deleting files..."
#    find "$DOWNLOAD_DIR" -type f -delete 2>/dev/null || true
    
#    echo "   Deleting directories..."
#    find "$DOWNLOAD_DIR" -type d -empty -delete 2>/dev/null || true
    
#    # Final cleanup
#    rmdir "$DOWNLOAD_DIR" 2>/dev/null || {
#        echo "   Using alternative cleanup method..."
#        mkdir -p /tmp/empty_dir
#        rsync -a --delete /tmp/empty_dir/ "$DOWNLOAD_DIR/" 2>/dev/null || true
#        rmdir /tmp/empty_dir 2>/dev/null || true
#        rmdir "$DOWNLOAD_DIR" 2>/dev/null || true
#    }
    
#    echo "   ✅ Directory removed successfully"
#else
#    echo "   ℹ️  Directory does not exist, skipping removal"
#fi

# Step 2: Download dataset
echo ""
echo "⬇️  Step 2: Downloading dataset..."

# Create data root if it doesn't exist
source /share/data/speech/txu/vlm_semantics/venv/bin/activate
cd /share/data/speech/txu/vlm_semantics/prismatic-vlms
# Run preprocessing script
echo "   Running preprocessing script..."
python scripts/preprocess.py \
    --dataset_id "$DATASET_ID" \
    --root_dir "$DATA_ROOT"

if [ $? -eq 0 ]; then
    echo "   ✅ Dataset downloaded successfully"
else
    echo "   ❌ Failed to download dataset"
    exit 1
fi

# Step 3: Verify download
echo ""
echo "🔍 Step 3: Verifying download..."

if [ -f "${DOWNLOAD_DIR}/llava_v1_5_mix665k.json" ]; then
    echo "   ✅ JSON file found"
else
    echo "   ❌ JSON file missing"
    exit 1
fi

if [ -d "${DOWNLOAD_DIR}/images" ]; then
    IMAGE_COUNT=$(find "${DOWNLOAD_DIR}/images" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    echo "   ✅ Images directory found with $IMAGE_COUNT images"
else
    echo "   ❌ Images directory missing"
    exit 1
fi

echo ""
echo "🎉 All done! Dataset is ready for training."
echo "📂 Final location: $DOWNLOAD_DIR"

#!/bin/bash
#SBATCH --job-name=cleanup
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=sallyxu@ttic.edu
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
#SBATCH -o /share/data/speech/txu/vlm_semantics/logs/cleanup_%j.out

# This script safely removes large data files and temporary directories
DATA_DIR="/share/data/speech/txu/vlm_semantics/prismatic-vlms/data"
echo "This script will delete files in the following directories:"
echo "- Data directory: $DATA_DIR"
echo ""

safe_delete() {
    local path="$1"
    local description="$2"
    
    if [ -d "$path" ] || [ -f "$path" ]; then
        echo "Found: $description at $path"
        rm -rf "$path"
        echo "Deleted: $description"
    else
        echo "Not found: $description at $path"
    fi
    echo "---"
}

# Clean up specific large data files
echo "=== Cleaning up specific data files ==="
# Add specific files/patterns you want to delete
safe_delete "$DATA_DIR" "Large dataset file"

echo "Cleanup completed!"

# Display remaining disk usage
echo "=== Disk usage after cleanup ==="
df -h /share/data/speech/txu/
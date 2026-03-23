#!/bin/bash
#SBATCH --job-name=signal_noise_sep
#SBATCH --output=logs/signal_noise_%j.out
#SBATCH --error=logs/signal_noise_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=standard

echo "======================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "======================================================================"

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Navigate to project directory
cd /home/lain01/Laiba

# Create output directories if they don't exist
mkdir -p logs
mkdir -p outputs/signal_noise

# Ensure required packages are available
pip install scikit-image --quiet

# Run the separation script
echo "Starting signal/noise separation..."
echo ""

python scripts/signal_noise_separation.py

echo ""
echo "======================================================================"
echo "Job completed at: $(date)"
echo "======================================================================"

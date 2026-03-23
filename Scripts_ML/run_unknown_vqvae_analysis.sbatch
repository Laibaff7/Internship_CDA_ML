#!/bin/bash
#SBATCH --job-name=unknown_vqvae
#SBATCH --output=logs/unknown_vqvae_%j.out
#SBATCH --error=logs/unknown_vqvae_%j.err
#SBATCH --time=08:00:00
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

# Create logs directory if it doesn't exist
mkdir -p logs

# Ensure required packages are available
pip install optuna openTSNE --quiet

# Run the analysis
echo "Starting VQ-VAE analysis of unknown data after noise filter..."
echo ""

python scripts/unknown_data_vqvae_analysis.py

echo ""
echo "======================================================================"
echo "Job completed at: $(date)"
echo "======================================================================"

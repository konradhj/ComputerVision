#!/bin/sh
#SBATCH --account=konradj            # TODO: update to your account (e.g. ie-idi, ie-ite)
#SBATCH --job-name=breast_mri_train
#SBATCH --time=0-08:00:00           # 8 hours — adjust based on dataset size
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --mem=32G                   # 32GB RAM for 3D volumes
#SBATCH --cpus-per-task=8           # for dataloader workers
#SBATCH --nodes=1
#SBATCH --output=/cluster/work/konradj/breast_mri/outputs/logs/slurm-%j.out
#SBATCH --error=/cluster/work/konradj/breast_mri/outputs/logs/slurm-%j.err
#SBATCH --mail-user=konradj@ntnu.no   # TODO: update
#SBATCH --mail-type=ALL

echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Started:   $(date)"
echo "========================================="

# Navigate to project directory
cd ${SLURM_SUBMIT_DIR}

# Load conda and activate environment
module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create output directories
mkdir -p /cluster/work/konradj/breast_mri/outputs/logs
mkdir -p /cluster/work/konradj/breast_mri/outputs/checkpoints

# Run training
python scripts/train.py --config configs/idun.yaml

echo "========================================="
echo "Finished:  $(date)"
echo "========================================="

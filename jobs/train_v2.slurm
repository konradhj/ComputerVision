#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=breast_mri_v2
#SBATCH --time=0-08:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-v2-%j.out
#SBATCH --error=logs/slurm-v2-%j.err
#SBATCH --mail-user=konradj@ntnu.no
#SBATCH --mail-type=ALL

echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Started:   $(date)"
echo "========================================="

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs
mkdir -p /cluster/work/konradj/breast_mri/outputs_v2/checkpoints

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

python -c "import torch; print(f'CUDA: {torch.cuda.get_device_name(0)}')"

python scripts/train.py --config configs/idun_v2.yaml

echo "========================================="
echo "Finished:  $(date)"
echo "========================================="

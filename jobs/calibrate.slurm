#!/bin/sh
#SBATCH --account=konradj
#SBATCH --job-name=breast_mri_cal
#SBATCH --time=0-00:30:00           # 30 minutes
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=/cluster/work/konradj/breast_mri/outputs/logs/slurm-cal-%j.out
#SBATCH --error=/cluster/work/konradj/breast_mri/outputs/logs/slurm-cal-%j.err

cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Fit temperature scaling
python scripts/calibrate.py --config configs/idun.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs/checkpoints/best.pt

echo "Calibration complete: $(date)"

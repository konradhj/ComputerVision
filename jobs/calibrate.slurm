#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=breast_mri_cal
#SBATCH --time=0-00:30:00           # 30 minutes
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-cal-%j.out
#SBATCH --error=logs/slurm-cal-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Fit temperature scaling on v2 (best model)
python scripts/calibrate.py --config configs/idun_v2.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v2/checkpoints/best.pt

echo "Calibration complete: $(date)"

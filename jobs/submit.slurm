#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=breast_mri_sub
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-sub-%j.out
#SBATCH --error=logs/slurm-sub-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Generate predictions CSV for RSH (leaderboard submission)
python scripts/generate_submission.py \
    --config configs/idun_v2.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v2/checkpoints/best.pt \
    --temperature /cluster/work/konradj/breast_mri/outputs_v2/calibration/temperature.pt \
    --output /cluster/work/konradj/breast_mri/predictions.csv

echo "Submission generated: $(date)"

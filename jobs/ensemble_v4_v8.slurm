#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=ensemble
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-ensemble-%j.out
#SBATCH --error=logs/slurm-ensemble-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Generate v4 predictions (no TTA, plain)
python scripts/generate_submission.py \
    --config configs/idun_v4.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v4/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v4_plain.csv

# Generate v8 predictions (no TTA, plain)
python scripts/generate_submission.py \
    --config configs/idun_v8.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v8/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v8_plain.csv

# Ensemble v4 + v8
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v4_plain.csv \
             /cluster/work/konradj/breast_mri/predictions_v8_plain.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v4_v8.csv

echo "Ensemble generated: $(date)"

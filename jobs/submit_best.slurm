#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=submit_best
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-subbest-%j.out
#SBATCH --error=logs/slurm-subbest-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Generate v5 submission
python scripts/generate_submission.py \
    --config configs/idun_v5.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v5/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v5.csv

# Generate v6 submission
python scripts/generate_submission.py \
    --config configs/idun_v6.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v6/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v6.csv

# Generate v8 plain (no TTA) for ensembling
python scripts/generate_submission.py \
    --config configs/idun_v8.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v8/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v8_plain.csv

# Ensemble v6 + v8 (best macro F1 + best aggregate)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v6.csv \
             /cluster/work/konradj/breast_mri/predictions_v8_plain.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v6_v8.csv

# Ensemble v5 + v6 + v8 (all three best models)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v5.csv \
             /cluster/work/konradj/breast_mri/predictions_v6.csv \
             /cluster/work/konradj/breast_mri/predictions_v8_plain.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v5_v6_v8.csv

echo "All submissions generated: $(date)"

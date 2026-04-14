#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v13b
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-subv13b-%j.out
#SBATCH --error=logs/slurm-subv13b-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# Generate v13b submission
python scripts/generate_submission.py \
    --config configs/idun_v13b.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v13b/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v13b.csv

# Ensemble v13 + v13b (2 seeds)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v13.csv \
             /cluster/work/konradj/breast_mri/predictions_v13b.csv \
    --output /cluster/work/konradj/breast_mri/predictions_v13_2seed.csv

echo "Done: $(date)"

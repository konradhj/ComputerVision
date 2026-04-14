#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v14c
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-subv14c-%j.out
#SBATCH --error=logs/slurm-subv14c-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# v14 improved checkpoint (epoch 83, val_loss=0.238)
python scripts/generate_submission.py \
    --config configs/idun_v14.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v14/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v14_final.csv

# v13c submission
python scripts/generate_submission.py \
    --config configs/idun_v13c.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v13c/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v13c.csv

# 3-seed DenseNet ensemble (v13 + v13b + v13c)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v13.csv \
             /cluster/work/konradj/breast_mri/predictions_v13b.csv \
             /cluster/work/konradj/breast_mri/predictions_v13c.csv \
    --output /cluster/work/konradj/breast_mri/predictions_v13_3seed.csv

# Cross-architecture ensemble (best DenseNet + best ResNet)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v13b.csv \
             /cluster/work/konradj/breast_mri/predictions_v14_final.csv \
    --output /cluster/work/konradj/breast_mri/predictions_v13b_v14_ensemble.csv

echo "All submissions generated: $(date)"

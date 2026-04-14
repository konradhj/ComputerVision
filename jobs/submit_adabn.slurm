#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=adabn
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-adabn-%j.out
#SBATCH --error=logs/slurm-adabn-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# AdaBN on v14 (ResNet50 — our best 3D model)
echo "=== AdaBN on v14 (ResNet50) ==="
python scripts/generate_submission_adabn.py \
    --config configs/idun_v14.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v14/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v14_adabn.csv

# AdaBN on v13b (DenseNet121 seed=123 — our best leaderboard single model)
echo "=== AdaBN on v13b (DenseNet121) ==="
python scripts/generate_submission_adabn.py \
    --config configs/idun_v13b.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v13b/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v13b_adabn.csv

# AdaBN on v13 (DenseNet121 seed=42)
echo "=== AdaBN on v13 (DenseNet121) ==="
python scripts/generate_submission_adabn.py \
    --config configs/idun_v13.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v13/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v13_adabn.csv

# Ensemble AdaBN models
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v14_adabn.csv \
             /cluster/work/konradj/breast_mri/predictions_v13b_adabn.csv \
    --output /cluster/work/konradj/breast_mri/predictions_adabn_ensemble.csv

echo "All AdaBN submissions generated: $(date)"

#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v14bv19
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-sub14bv19-%j.out
#SBATCH --error=logs/slurm-sub14bv19-%j.err
#SBATCH --exclude=idun-11-[01-20]

echo "Node: $SLURM_JOB_NODELIST"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# v14b submission (ResNet50 seed=123)
echo "=== v14b (ResNet50 seed=123) ==="
python scripts/generate_submission.py \
    --config configs/idun_v14b.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v14b/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v14b.csv

# v19 submission (MedicalNet + InstanceNorm)
echo "=== v19 (MedicalNet + InstanceNorm) ==="
python scripts/generate_submission.py \
    --config configs/idun_v19.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v19/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v19.csv

# Ensembles with existing best models
echo "=== Ensembles ==="

# v17 + v14b (InstanceNorm + seed=123 ResNet50)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17.csv /cluster/work/konradj/breast_mri/predictions_v14b.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17_v14b.csv

# v17 + v19 (InstanceNorm ResNet50 + MedicalNet+InstanceNorm)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17.csv /cluster/work/konradj/breast_mri/predictions_v19.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17_v19.csv

# v17 + v13b + v14b (3-model ensemble)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17.csv /cluster/work/konradj/breast_mri/predictions_v13b.csv /cluster/work/konradj/breast_mri/predictions_v14b.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17_v13b_v14b.csv

# v17 + v13b + v19 (3-model with MedicalNet)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17.csv /cluster/work/konradj/breast_mri/predictions_v13b.csv /cluster/work/konradj/breast_mri/predictions_v19.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17_v13b_v19.csv

echo "Done: $(date)"

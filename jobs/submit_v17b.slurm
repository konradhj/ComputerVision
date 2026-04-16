#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v17b
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-sub17b-%j.out
#SBATCH --error=logs/slurm-sub17b-%j.err
#SBATCH --exclude=idun-11-[01-20]

echo "Node: $SLURM_JOB_NODELIST"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# v17b submission (InstanceNorm + seed=123)
echo "=== v17b (InstanceNorm + seed=123) ==="
python scripts/generate_submission.py \
    --config configs/idun_v17b.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v17b/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v17b.csv

# Ensembles with existing best models
echo "=== Ensembles ==="

# v17b + v13b (InstanceNorm-s123 + DenseNet121)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17b.csv /cluster/work/konradj/breast_mri/predictions_v13b.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17b_v13b.csv

# v17b + v17 (seed=123 vs seed=42 InstanceNorm)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17b.csv /cluster/work/konradj/breast_mri/predictions_v17.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17b_v17.csv

# v17b + v13b + v14b (replace v17 with v17b in current best ensemble)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17b.csv /cluster/work/konradj/breast_mri/predictions_v13b.csv /cluster/work/konradj/breast_mri/predictions_v14b.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17b_v13b_v14b.csv

# v17b + v17 + v13b (4-way diversity: two InstanceNorm seeds + DenseNet121)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17b.csv /cluster/work/konradj/breast_mri/predictions_v17.csv /cluster/work/konradj/breast_mri/predictions_v13b.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17b_v17_v13b.csv

# v17b + v17 + v13b + v14b (4-model: both IN seeds + DenseNet121 + ResNet50-BN)
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v17b.csv /cluster/work/konradj/breast_mri/predictions_v17.csv /cluster/work/konradj/breast_mri/predictions_v13b.csv /cluster/work/konradj/breast_mri/predictions_v14b.csv \
    --output /cluster/work/konradj/breast_mri/predictions_ensemble_v17b_v17_v13b_v14b.csv

echo "Done: $(date)"

#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_all
#SBATCH --time=0-00:45:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-suball-%j.out
#SBATCH --error=logs/slurm-suball-%j.err
#SBATCH --exclude=idun-11-[01-20]

echo "Node: $SLURM_JOB_NODELIST"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# v14c (ResNet50 seed=456, perfect val)
echo "=== v14c ==="
python scripts/generate_submission.py \
    --config configs/idun_v14c.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v14c/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v14c.csv

# v17 (InstanceNorm, perfect val)
echo "=== v17 ==="
python scripts/generate_submission.py \
    --config configs/idun_v17.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v17/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v17.csv

# v18 (MedicalNet pretrained)
echo "=== v18 ==="
python scripts/generate_submission.py \
    --config configs/idun_v18.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v18/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v18.csv

# Ensembles
echo "=== Ensembles ==="
python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v14c.csv \
             /cluster/work/konradj/breast_mri/predictions_v17.csv \
    --output /cluster/work/konradj/breast_mri/predictions_v14c_v17_ensemble.csv

python scripts/ensemble_submissions.py \
    --inputs /cluster/work/konradj/breast_mri/predictions_v14c.csv \
             /cluster/work/konradj/breast_mri/predictions_v17.csv \
             /cluster/work/konradj/breast_mri/predictions_v18.csv \
    --output /cluster/work/konradj/breast_mri/predictions_v14c_v17_v18_ensemble.csv

echo "All done: $(date)"

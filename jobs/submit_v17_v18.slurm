#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v17v18
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-sub17v18-%j.out
#SBATCH --error=logs/slurm-sub17v18-%j.err
#SBATCH --exclude=idun-11-[01-20]

echo "Node: $SLURM_JOB_NODELIST"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# v17 submission (InstanceNorm)
echo "=== v17 (InstanceNorm) ==="
python scripts/generate_submission.py \
    --config configs/idun_v17.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v17/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v17.csv

# v18 submission (MedicalNet pretrained)
echo "=== v18 (MedicalNet) ==="
python scripts/generate_submission.py \
    --config configs/idun_v18.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v18/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v18.csv

echo "Done: $(date)"

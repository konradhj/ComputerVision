#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v16
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --constraint="V100|A100|H100|A40"
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-subv16-%j.out
#SBATCH --error=logs/slurm-subv16-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# v16 uses masked data — need to generate RSH predictions from masked data too
python scripts/generate_submission.py \
    --config configs/idun_v16.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v16/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v16.csv

echo "v16 submission generated: $(date)"

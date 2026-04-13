#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=v8_submit
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-v8sub-%j.out
#SBATCH --error=logs/slurm-v8sub-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

python scripts/generate_submission.py \
    --config configs/idun_v8.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v8/checkpoints/best.pt \
    --tta 8 \
    --output /cluster/work/konradj/breast_mri/predictions_v8.csv

echo "v8 submission generated: $(date)"

#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=v4_tta_sub
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-v4tta-%j.out
#SBATCH --error=logs/slurm-v4tta-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Step 1: Calibrate v4
python scripts/calibrate.py --config configs/idun_v4.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v4/checkpoints/best.pt

# Step 2: Generate submission with TTA (8 passes) + temperature scaling
python scripts/generate_submission.py \
    --config configs/idun_v4.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v4/checkpoints/best.pt \
    --temperature /cluster/work/konradj/breast_mri/outputs_v4/calibration/temperature.pt \
    --tta 8 \
    --output /cluster/work/konradj/breast_mri/predictions_v4_tta.csv

echo "v4 TTA submission generated: $(date)"

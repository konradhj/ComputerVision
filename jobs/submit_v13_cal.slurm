#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=v13_cal
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-v13cal-%j.out
#SBATCH --error=logs/slurm-v13cal-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# Calibrate v13
python scripts/calibrate.py --config configs/idun_v13.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v13/checkpoints/best.pt

# Submit with calibration
python scripts/generate_submission.py \
    --config configs/idun_v13.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v13/checkpoints/best.pt \
    --temperature /cluster/work/konradj/breast_mri/outputs_v13/calibration/temperature.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v13_cal.csv

echo "v13 calibrated submission generated: $(date)"

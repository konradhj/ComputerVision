#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v14
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-subv14-%j.out
#SBATCH --error=logs/slurm-subv14-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

python scripts/generate_submission.py \
    --config configs/idun_v14.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v14/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v14.csv

echo "v14 submission generated: $(date)"

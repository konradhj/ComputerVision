#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v15
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-subv15-%j.out
#SBATCH --error=logs/slurm-subv15-%j.err

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

python scripts/generate_submission.py \
    --config configs/idun_v15.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v15/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v15.csv

echo "v15 submission generated: $(date)"

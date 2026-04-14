#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=v14_rn50
#SBATCH --time=0-06:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-v14-%j.out
#SBATCH --error=logs/slurm-v14-%j.err
#SBATCH --mail-user=konradj@ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs /cluster/work/konradj/breast_mri/outputs_v14/checkpoints
module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri
python scripts/train.py --config configs/idun_v14.yaml

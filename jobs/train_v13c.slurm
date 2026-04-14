#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=v13c_s456
#SBATCH --time=0-06:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-v13c-%j.out
#SBATCH --error=logs/slurm-v13c-%j.err
#SBATCH --mail-user=konradj@ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs /cluster/work/konradj/breast_mri/outputs_v13c/checkpoints
module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri
python scripts/train.py --config configs/idun_v13c.yaml

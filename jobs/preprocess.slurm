#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=preprocess
#SBATCH --time=0-02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-preprocess-%j.out
#SBATCH --error=logs/slurm-preprocess-%j.err
#SBATCH --mail-user=konradj@ntnu.no
#SBATCH --mail-type=ALL

echo "========================================="
echo "Preprocessing with BreastDivider"
echo "Started:   $(date)"
echo "========================================="

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# Install breastdivider if not already installed
pip install breastdivider --quiet

python scripts/preprocess_masks.py \
    --data_root /cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/ \
    --output_root /cluster/work/konradj/breast_mri/data_masked/ \
    --device cuda

echo "========================================="
echo "Finished:  $(date)"
echo "========================================="

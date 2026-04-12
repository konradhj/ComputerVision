#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=breast_mri_infer
#SBATCH --time=0-01:00:00           # 1 hour is plenty for inference
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=/cluster/work/konradj/breast_mri/outputs/logs/slurm-infer-%j.out
#SBATCH --error=/cluster/work/konradj/breast_mri/outputs/logs/slurm-infer-%j.err
#SBATCH --mail-user=konradj@ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Run inference to generate bilateral JSON submissions
python scripts/inference.py --config configs/idun.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs/checkpoints/best.pt \
    --output_dir /cluster/work/konradj/breast_mri/submissions/

echo "Inference complete: $(date)"

#!/bin/sh
#SBATCH --account=konradj            # TODO: update to your account
#SBATCH --job-name=breast_mri_infer
#SBATCH --time=0-01:00:00           # 1 hour is plenty for inference
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=outputs/logs/slurm-infer-%j.out
#SBATCH --error=outputs/logs/slurm-infer-%j.err
#SBATCH --mail-user=konradj@ntnu.no   # TODO: update
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

# Run inference to generate bilateral JSON submissions
python scripts/inference.py --config configs/default.yaml \
    --checkpoint outputs/checkpoints/best.pt \
    --output_dir submissions/

echo "Inference complete: $(date)"

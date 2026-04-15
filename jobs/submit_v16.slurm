#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sub_v16
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-subv16-%j.out
#SBATCH --error=logs/slurm-subv16-%j.err
#SBATCH --exclude=idun-11-02,idun-11-04,idun-11-06,idun-11-07,idun-11-08,idun-11-10,idun-11-13,idun-11-16,idun-11-19,idun-11-20

echo "Node: $SLURM_JOB_NODELIST"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

python -c "import torch; print(f'CUDA: {torch.cuda.get_device_name(0)}')"

python scripts/generate_submission.py \
    --config configs/idun_v16.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs_v16/checkpoints/best.pt \
    --output /cluster/work/konradj/breast_mri/predictions_v16.csv

echo "v16 submission generated: $(date)"

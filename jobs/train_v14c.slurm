#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=v14c_s456
#SBATCH --time=0-10:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-v14c-%j.out
#SBATCH --error=logs/slurm-v14c-%j.err
#SBATCH --exclude=idun-11-[01-20]
#SBATCH --mail-user=konradj@ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs /cluster/work/konradj/breast_mri/outputs_v14c/checkpoints
module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri
echo "Node: $SLURM_JOB_NODELIST"
python -c "import torch; print(f'CUDA: {torch.cuda.get_device_name(0)}')"
python scripts/train.py --config configs/idun_v14c.yaml

#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=v18_med3d
#SBATCH --time=0-06:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=logs/slurm-v18-%j.out
#SBATCH --error=logs/slurm-v18-%j.err
#SBATCH --exclude=idun-11-[01-20]
#SBATCH --mail-user=konradj@ntnu.no
#SBATCH --mail-type=ALL

echo "Node: $SLURM_JOB_NODELIST"

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs /cluster/work/konradj/breast_mri/outputs_v18/checkpoints

module purge && module load Anaconda3/2023.09-0 && conda activate breast_mri

# Download MedicalNet pretrained weights if not already present
PRETRAIN_DIR="/cluster/work/konradj/breast_mri/pretrained"
PRETRAIN_FILE="$PRETRAIN_DIR/resnet_50_23dataset.pth"
mkdir -p $PRETRAIN_DIR

if [ ! -f "$PRETRAIN_FILE" ]; then
    echo "Downloading MedicalNet ResNet50 pretrained weights..."
    python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='TencentMedicalNet/MedicalNet-Resnet50',
    filename='resnet_50_23dataset.pth',
    local_dir='$PRETRAIN_DIR'
)
print(f'Downloaded to: {path}')
" || echo "HuggingFace download failed, trying wget..."
    # Fallback: download from Zenodo
    if [ ! -f "$PRETRAIN_FILE" ]; then
        wget -q "https://zenodo.org/records/15234379/files/resnet_50_23dataset.pth" -O "$PRETRAIN_FILE"
    fi
fi

echo "Pretrained weights: $(ls -la $PRETRAIN_FILE 2>/dev/null || echo 'NOT FOUND')"

python -c "import torch; print(f'CUDA: {torch.cuda.get_device_name(0)}')"

python scripts/train.py --config configs/idun_v18.yaml

echo "Finished: $(date)"

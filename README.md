# Breast MRI Classification

3D breast MRI classification using PyTorch and MONAI. Classifies each breast as **normal**, **benign**, or **malignant** from multi-sequence MRI.

## Approach

- **Unilateral classifier**: A single 3D DenseNet121 is trained on individual breasts
- **Bilateral output**: At inference, left and right predictions are combined into study-level JSON files
- **Input**: 4 MRI sequences per breast — Pre-contrast, Post-contrast 1, Post-contrast 2, T2-weighted
- **Improvements**: Temperature scaling (calibration), class-weighted loss (imbalance), sequence ablation (analysis)

### Why this design?

| Decision | Rationale |
| --- | --- |
| **3D model** | Breast MRI is volumetric — a 3D CNN captures spatial relationships across slices that 2D models miss |
| **Multi-sequence input** | Pre (anatomy), Post-contrast (vascularity/enhancement patterns), T2 (tissue characterization) provide complementary diagnostic information |
| **Unilateral training** | Clean separation — one model applied per-breast, no coupling between sides, doubles effective training samples |
| **DenseNet121** | Strong feature reuse with fewer parameters than ResNet50, well-suited for medical imaging with limited data |
| **Temperature scaling** | Clinical predictions need reliable probabilities, not just correct rankings — one learned parameter fixes overconfident softmax outputs |

### Pipeline overview

```text
NIfTI files (Pre, Post_1, Post_2, T2)
    |
    v
Load + Normalize + Concatenate → [4, D, H, W] tensor
    |
    v
Resize to [4, 128, 128, 32]
    |
    v
3D DenseNet121 → logits [3]
    |
    v
(optional) Temperature scaling → calibrated logits
    |
    v
Softmax → probabilities {normal, benign, malignant}
    |
    v
Group left + right by study → bilateral JSON
```

## Project Structure

```text
configs/
  idun.yaml                 # IDUN config (points to shared course data)
  default.yaml              # Local/generic config

src/
  data/
    dataset.py              # Dataset class and dataloader factory
    transforms.py           # MONAI preprocessing and augmentation
    label_mapping.py        # Label loading, UID parsing, class weights
  models/
    classifier.py           # 3D DenseNet121 / ResNet wrapper
  training/
    trainer.py              # Train/val loops, early stopping, checkpointing
    losses.py               # Cross-entropy with optional class weighting
  evaluation/
    metrics.py              # Accuracy, AUROC, confusion matrix, threshold metrics
    bilateral.py            # Unilateral → bilateral JSON assembly
  calibration/
    temperature_scaling.py  # Post-hoc temperature calibration
  utils/
    config.py               # YAML config → Python dataclass
    reproducibility.py      # Seeding, device selection
    logging_utils.py        # Metric logging to CSV

scripts/
  train.py                  # Train the classifier
  evaluate.py               # Evaluate on val/test with full metrics
  inference.py              # Generate bilateral JSON submissions
  calibrate.py              # Fit temperature scaling on val set
  ablation.py               # Sequence ablation study

jobs/
  train.slurm               # SLURM job for training
  inference.slurm           # SLURM job for generating submissions
  calibrate.slurm           # SLURM job for temperature calibration
```

## IDUN Setup (one-time)

### 1. SSH into IDUN

```bash
ssh konradj@idun.hpc.ntnu.no
```

### 2. Create conda environment

```bash
module purge
module load Anaconda3/2023.09-0
conda create -n breast_mri python=3.11 -y
conda activate breast_mri

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install monai nibabel scikit-learn scipy pyyaml tqdm matplotlib
```

### 3. Clone the repo

```bash
cd /cluster/home/konradj/
git clone <your-repo-url> ComputerVision
```

That's it. The data is already on IDUN (read-only, shared across the course) and `configs/idun.yaml` points to it. Outputs are written to `/cluster/work/konradj/breast_mri/` to avoid filling home quota.

## Data

The ODELIA2025 dataset is at:

```text
/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/
├── data/
│   ├── CAM/
│   │   ├── data_unilateral/<UID>/{Pre,Post_1,Post_2,Sub_1,T2}.nii.gz
│   │   └── metadata_unilateral/annotation.csv    # UID, PatientID, Age, Lesion
│   ├── MHA/   (same structure)
│   ├── RSH/   (no annotation.csv — labels hidden for leaderboard)
│   ├── RUMC/  (same structure)
│   └── UKA/   (same structure)
├── split_unilateral.csv          # UID, Fold, Split, Institution
└── evaluation-method/
```

- 5 institutions: CAM, MHA, RSH, RUMC, UKA
- RSH labels are hidden for the leaderboard — those samples can only be used for inference
- Labels are loaded automatically from each institution's `annotation.csv` (`label_csv: "auto"` in config)
- Label mapping: 0 = normal, 1 = benign, 2 = malignant

## Running on IDUN

All commands below are run from the repo directory on IDUN:

```bash
cd /cluster/home/konradj/ComputerVision
```

### Step 1: Train

```bash
sbatch jobs/train.slurm
```

Monitor your job:

```bash
# Check job status
squeue -u konradj

# Watch training output live (replace <jobid> with your job ID)
tail -f /cluster/work/konradj/breast_mri/outputs/logs/slurm-<jobid>.out

# Cancel a job if needed
scancel <jobid>
```

Training saves checkpoints to `/cluster/work/konradj/breast_mri/outputs/checkpoints/`. The best model (by validation loss) is saved as `best.pt`.

### Step 2: Evaluate

Request an interactive GPU session, then run evaluation:

```bash
srun --account=konradj --partition=GPUQ --gres=gpu:1 --mem=16G --cpus-per-task=4 --time=0-01:00:00 --pty bash

# Inside the interactive session:
module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

python scripts/evaluate.py --config configs/idun.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs/checkpoints/best.pt
```

### Step 3: Calibrate (optional improvement)

```bash
sbatch jobs/calibrate.slurm
```

This learns a temperature parameter on the validation set and saves it to `/cluster/work/konradj/breast_mri/outputs/calibration/temperature.pt`.

### Step 4: Generate submission

```bash
sbatch jobs/inference.slurm
```

This produces bilateral JSON files in `/cluster/work/konradj/breast_mri/submissions/`.

### Step 5: Sequence ablation (optional analysis)

Request an interactive session with more time:

```bash
srun --account=konradj --partition=GPUQ --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=0-06:00:00 --pty bash

module purge
module load Anaconda3/2023.09-0
conda activate breast_mri

python scripts/ablation.py --config configs/idun.yaml --epochs 30
```

### Overriding config from command line

Any config parameter can be overridden:

```bash
# In an interactive session or in a SLURM script:
python scripts/train.py --config configs/idun.yaml training.epochs=50 data.batch_size=2
```

## IDUN Tips

- **Data**: Read-only at `/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/` — no copying needed
- **Outputs**: Written to `/cluster/work/konradj/breast_mri/outputs/` — plenty of space
- **Time**: `train.slurm` requests 8 hours; shorter requests get higher queue priority
- **Memory**: `--mem=32G` is enough for 128x128x32 volumes; increase for larger spatial sizes
- **Batch size**: Reduce to 2 (`data.batch_size=2`) if you get GPU out-of-memory errors
- **Workers**: `--cpus-per-task` in SLURM matches `data.num_workers` in config (both 8)
- **Multiple experiments**: Copy `configs/idun.yaml` to `configs/experiment_name.yaml`, modify, and update the SLURM script to use it

## Evaluation Metrics

| Metric | Purpose |
| --- | --- |
| **Accuracy** | Overall correctness |
| **Confusion matrix** | Per-class error patterns |
| **Per-class precision, recall, F1** | Class-level performance |
| **Malignant-vs-rest AUROC** | Ranking quality for the critical malignant class |
| **Specificity @ 90% sensitivity** | Clinical: how many false positives when catching 90% of cancers |
| **Sensitivity @ 90% specificity** | Clinical: how many cancers caught when allowing 10% false positive rate |
| **ECE** | Expected Calibration Error — how reliable the predicted probabilities are |

## Configuration

All settings are in `configs/idun.yaml`. Key parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `data.sequences` | `[Pre, Post_1, Post_2, T2]` | MRI sequences to use as input channels |
| `data.spatial_size` | `[128, 128, 32]` | Volume resize target |
| `data.batch_size` | `4` | Batch size (reduce to 2 if GPU OOM) |
| `model.architecture` | `densenet121` | Backbone: `densenet121`, `resnet18`, `resnet50` |
| `training.epochs` | `100` | Max training epochs |
| `training.learning_rate` | `1e-4` | Initial learning rate |
| `training.early_stopping_patience` | `15` | Stop after N epochs without improvement |
| `training.mixed_precision` | `true` | Use AMP on CUDA (saves memory, faster) |
| `training.class_weights` | `null` | `null` = auto-compute from train set, or `[w0, w1, w2]` |
| `training.label_smoothing` | `0.0` | Label smoothing factor |
| `augmentation.rand_flip_prob` | `0.5` | Random flip probability |
| `augmentation.rand_affine_prob` | `0.3` | Random affine transform probability |
| `calibration.enabled` | `false` | Enable temperature scaling |

## Submission Format

The evaluator expects this directory structure:

```text
submissions/
  <study_id>/
    bilateral-breast-classification-likelihoods.json
```

Each JSON file contains:

```json
{
  "left": {
    "normal": 0.1,
    "benign": 0.2,
    "malignant": 0.7
  },
  "right": {
    "normal": 0.8,
    "benign": 0.1,
    "malignant": 0.1
  }
}
```

Probabilities per side must sum to 1.0. The `scripts/inference.py` script generates this format automatically.

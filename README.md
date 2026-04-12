# Breast MRI Classification

3D breast MRI classification using PyTorch and MONAI. Classifies each breast as **normal**, **benign**, or **malignant** from multi-sequence MRI.

## Approach

- **Unilateral classifier**: A single 3D DenseNet121 is trained on individual breasts
- **Bilateral output**: At inference, left and right predictions are combined into study-level JSON files
- **Input**: 4 MRI sequences per breast — Pre-contrast, Post-contrast 1, Post-contrast 2, T2-weighted
- **Improvements**: Temperature scaling (calibration), class-weighted loss (imbalance), sequence ablation (analysis)

### Why this design?

| Decision | Rationale |
|---|---|
| **3D model** | Breast MRI is volumetric — a 3D CNN captures spatial relationships across slices that 2D models miss |
| **Multi-sequence input** | Pre (anatomy), Post-contrast (vascularity/enhancement patterns), T2 (tissue characterization) provide complementary diagnostic information |
| **Unilateral training** | Clean separation — one model applied per-breast, no coupling between sides, doubles effective training samples |
| **DenseNet121** | Strong feature reuse with fewer parameters than ResNet50, well-suited for medical imaging with limited data |
| **Temperature scaling** | Clinical predictions need reliable probabilities, not just correct rankings — one learned parameter fixes overconfident softmax outputs |

### Pipeline overview

```
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

```
configs/
  default.yaml              # All hyperparameters (local)
  idun.yaml                 # IDUN config (NTNU HPC, points to shared data)

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
  train.slurm               # SLURM script for training on IDUN
  inference.slurm           # SLURM script for generating submissions
  calibrate.slurm           # SLURM script for temperature calibration
```

## Setup

### Local

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with PyTorch, MONAI, scikit-learn, and standard scientific Python packages.

### IDUN (NTNU HPC)

```bash
# SSH into IDUN
ssh <username>@idun.hpc.ntnu.no

# Create conda environment (one-time)
module purge
module load Anaconda3/2023.09-0
conda create -n breast_mri python=3.11 -y
conda activate breast_mri

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install monai nibabel scikit-learn scipy pyyaml tqdm matplotlib
```

## Data

### IDUN (no setup needed)

The ODELIA2025 dataset is available on IDUN at:

```
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

`configs/idun.yaml` already points to these paths. Labels are loaded automatically from each institution's `annotation.csv` (set `label_csv: "auto"`).

5 institutions: CAM, MHA, RSH, RUMC, UKA. RSH labels are hidden for the leaderboard — those samples can only be used for inference.

### Local setup

1. Place your data under `data/` following the layout: `data/<Institution>/data_unilateral/<UID>/`
2. Each UID folder should contain: `Pre.nii.gz`, `Post_1.nii.gz`, `Post_2.nii.gz`, `T2.nii.gz`
3. Place `split_unilateral.csv` in the project root
4. Set `label_csv: "auto"` in `configs/default.yaml` to auto-load from per-institution annotation CSVs, or point to a single label CSV

The label CSV should have either:
- **Unilateral format**: columns `UID`, `Lesion` (0=normal, 1=benign, 2=malignant)
- **Bilateral format**: columns `studyID`, `Lesion_Left`, `Lesion_Right`

Both formats are auto-detected.

## Usage

### Train (local)

```bash
python scripts/train.py --config configs/default.yaml
```

Override any parameter from the command line:

```bash
python scripts/train.py --config configs/default.yaml training.epochs=50 data.batch_size=2
```

### Train (IDUN)

```bash
# Submit training job
sbatch jobs/train.slurm

# Check job status
squeue -u konradj

# Watch output live
tail -f /cluster/work/konradj/breast_mri/outputs/logs/slurm-<jobid>.out

# Cancel a job
scancel <jobid>
```

### Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test
```

### Generate Submission

```bash
python scripts/inference.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

Produces one JSON per study in `submissions/<study_id>/bilateral-breast-classification-likelihoods.json`:

```json
{
  "left":  {"normal": 0.1, "benign": 0.2, "malignant": 0.7},
  "right": {"normal": 0.8, "benign": 0.1, "malignant": 0.1}
}
```

### Calibrate

```bash
python scripts/calibrate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

Then use the learned temperature at inference:

```bash
python scripts/inference.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt \
    --temperature outputs/calibration/temperature.pt
```

### Sequence Ablation

```bash
python scripts/ablation.py --config configs/default.yaml --epochs 30
```

Trains with different MRI sequence subsets and reports a comparison table showing each sequence's contribution.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness |
| **Confusion matrix** | Per-class error patterns |
| **Per-class precision, recall, F1** | Class-level performance |
| **Malignant-vs-rest AUROC** | Ranking quality for the critical malignant class |
| **Specificity @ 90% sensitivity** | Clinical operating point: how many false positives when catching 90% of cancers |
| **Sensitivity @ 90% specificity** | Clinical operating point: how many cancers caught when allowing 10% false positive rate |
| **ECE** | Expected Calibration Error — how reliable the predicted probabilities are |

## Configuration

All settings are in `configs/default.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
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

## IDUN Tips

- **Data**: Already at `/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/` — read-only, shared across the course
- **Outputs**: Written to `/cluster/work/konradj/breast_mri/` — personal work directory with plenty of space
- **Time allocation**: `--time=0-08:00:00` is 8 hours; shorter requests get higher queue priority
- **Memory**: `--mem=32G` is enough for 128x128x32 volumes; increase for larger spatial sizes
- **Workers**: `--cpus-per-task` in SLURM should match `data.num_workers` in config (both set to 8)
- **Monitoring**: `tail -f /cluster/work/konradj/breast_mri/outputs/logs/slurm-<jobid>.out`
- **Multiple runs**: Copy `configs/idun.yaml` to `configs/experiment_name.yaml` and modify

## Experiment Workflow

Recommended order for a complete project:

```bash
# 1. Baseline training
sbatch jobs/train.slurm

# 2. Evaluate baseline
python scripts/evaluate.py --config configs/idun.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs/checkpoints/best.pt

# 3. Calibrate
sbatch jobs/calibrate.slurm

# 4. Evaluate with calibration
python scripts/evaluate.py --config configs/idun.yaml \
    --checkpoint /cluster/work/konradj/breast_mri/outputs/checkpoints/best.pt \
    --temperature /cluster/work/konradj/breast_mri/outputs/calibration/temperature.pt

# 5. Sequence ablation (optional, takes longer)
python scripts/ablation.py --config configs/idun.yaml --epochs 30

# 6. Generate final submission
sbatch jobs/inference.slurm
```

## Submission Format

The evaluator expects this directory structure:

```
submissions/
  <study_id>/
    bilateral-breast-classification-likelihoods.json
```

Each JSON file contains:

```json
{
  "left": {
    "normal": <float>,
    "benign": <float>,
    "malignant": <float>
  },
  "right": {
    "normal": <float>,
    "benign": <float>,
    "malignant": <float>
  }
}
```

Probabilities per side must sum to 1.0. The `scripts/inference.py` script generates this format automatically.

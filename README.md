# Breast MRI Classification

3D breast MRI classification using PyTorch and MONAI. Classifies each breast as **normal**, **benign**, or **malignant** from multi-sequence MRI.

## Approach

- **Unilateral classifier**: A single 3D DenseNet121 is trained on individual breasts
- **Bilateral output**: At inference, left and right predictions are combined into study-level JSON files
- **Input**: 4 MRI sequences per breast — Pre-contrast, Post-contrast 1, Post-contrast 2, T2-weighted
- **Improvements**: Temperature scaling (calibration), class-weighted loss (imbalance), sequence ablation (analysis)

## Project Structure

```
configs/
  default.yaml              # All hyperparameters

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
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with PyTorch, MONAI, scikit-learn, and standard scientific Python packages.

## Data Preparation

1. Place your data under `data/` following the layout: `data/<Institution>/data_unilateral/<UID>/`
2. Each UID folder should contain: `Pre.nii.gz`, `Post_1.nii.gz`, `Post_2.nii.gz`, `T2.nii.gz`
3. Place `split_unilateral.csv` in the project root (columns: `UID`, `Fold`, `Split`, `Institution`)
4. Place your label CSV and update `data.label_csv` in `configs/default.yaml`

The label CSV should have either:
- **Unilateral format**: columns `UID`, `Lesion` (0=normal, 1=benign, 2=malignant)
- **Bilateral format**: columns `studyID`, `Lesion_Left`, `Lesion_Right`

Both formats are auto-detected.

## Usage

### Train

```bash
python scripts/train.py --config configs/default.yaml
```

Override any parameter from the command line:

```bash
python scripts/train.py --config configs/default.yaml training.epochs=50 data.batch_size=2
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

Trains with different MRI sequence subsets and reports a comparison table.

## Evaluation Metrics

- **Accuracy** and **confusion matrix**
- **Per-class precision, recall, F1**
- **Malignant-vs-rest AUROC**
- **Specificity @ 90% sensitivity**
- **Sensitivity @ 90% specificity**
- **ECE** (Expected Calibration Error)

## Configuration

All settings are in `configs/default.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `data.sequences` | `[Pre, Post_1, Post_2, T2]` | MRI sequences to use |
| `data.spatial_size` | `[128, 128, 32]` | Volume resize target |
| `model.architecture` | `densenet121` | Backbone (`densenet121`, `resnet18`, `resnet50`) |
| `training.epochs` | `100` | Max training epochs |
| `training.learning_rate` | `1e-4` | Initial learning rate |
| `training.mixed_precision` | `true` | Use AMP on CUDA |
| `training.class_weights` | `null` | Auto-compute from train set, or specify `[w0, w1, w2]` |

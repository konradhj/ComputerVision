#!/usr/bin/env python3
"""
Leave-One-Institution-Out (LOO) training.

Trains 4 models, each holding out a different institution for validation.
Then generates ensemble predictions for RSH.

Usage:
    python scripts/train_loo.py --config configs/idun_v13.yaml
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BreastMRIDataset, SampleInfo, build_sample_list, _resolve_data_path
from src.data.label_mapping import compute_class_weights, load_labels_from_institutions
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.classifier import build_model, load_model_checkpoint
from src.training.losses import get_loss_function
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import get_device, seed_everything

# Institutions with labeled data
INSTITUTIONS = ["CAM", "MHA", "RUMC", "UKA"]


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )


def build_scheduler(optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.scheduler_patience, factor=0.5
        )
    return None


def train_one_loo(cfg, val_institution, device, logger):
    """Train one LOO model with val_institution held out."""
    logger.info(f"\n{'='*60}")
    logger.info(f"LOO: Validate on {val_institution}, train on {[i for i in INSTITUTIONS if i != val_institution]}")
    logger.info(f"{'='*60}")

    # Override output paths for this LOO split
    loo_dir = Path(cfg.paths.output_dir) / f"loo_{val_institution}"
    loo_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = loo_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.checkpoint_dir = str(ckpt_dir)
    cfg.paths.log_dir = str(loo_dir)

    # Build transforms
    use_pct = getattr(cfg.augmentation, 'use_percentile_norm', False)
    noise_prob = getattr(cfg.augmentation, 'rand_gaussian_noise_prob', 0.0)
    noise_std = getattr(cfg.augmentation, 'rand_gaussian_noise_std', 0.05)
    crop_fg = getattr(cfg.augmentation, 'crop_foreground', False)

    train_transforms = get_train_transforms(
        sequences=cfg.data.sequences,
        spatial_size=cfg.data.spatial_size,
        rand_flip_prob=cfg.augmentation.rand_flip_prob,
        rand_rotate90_prob=cfg.augmentation.rand_rotate90_prob,
        rand_affine_prob=cfg.augmentation.rand_affine_prob,
        rand_affine_rotate_range=cfg.augmentation.rand_affine_rotate_range,
        rand_affine_scale_range=cfg.augmentation.rand_affine_scale_range,
        rand_intensity_shift=cfg.augmentation.rand_intensity_shift,
        rand_intensity_scale=cfg.augmentation.rand_intensity_scale,
        use_percentile_norm=use_pct,
        rand_gaussian_noise_prob=noise_prob,
        rand_gaussian_noise_std=noise_std,
        crop_foreground=crop_fg,
    )
    val_transforms = get_val_transforms(
        sequences=cfg.data.sequences,
        spatial_size=cfg.data.spatial_size,
        use_percentile_norm=use_pct,
        crop_foreground=crop_fg,
    )

    # Build sample lists with LOO filtering
    # Train: all institutions except val_institution, all splits combined
    train_samples = build_sample_list(
        cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
        cfg.data.sequences, cfg.data.fold, "train",
        exclude_institution=val_institution,
    )
    # Also include val/test splits from training institutions (maximize data)
    for extra_split in ["val", "test"]:
        extra = build_sample_list(
            cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
            cfg.data.sequences, cfg.data.fold, extra_split,
            exclude_institution=val_institution,
        )
        train_samples.extend(extra)

    # Validation: only the held-out institution
    val_samples = build_sample_list(
        cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
        cfg.data.sequences, cfg.data.fold, "train",
        only_institution=val_institution,
    )
    # Also include val/test splits from held-out institution
    for extra_split in ["val", "test"]:
        extra = build_sample_list(
            cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
            cfg.data.sequences, cfg.data.fold, extra_split,
            only_institution=val_institution,
        )
        val_samples.extend(extra)

    logger.info(f"LOO train: {len(train_samples)} samples, LOO val ({val_institution}): {len(val_samples)} samples")

    if len(val_samples) < 2:
        logger.warning(f"Not enough val samples for {val_institution}, skipping")
        return None

    # Build datasets and dataloaders
    train_ds = BreastMRIDataset(train_samples, train_transforms)
    val_ds = BreastMRIDataset(val_samples, val_transforms)

    # Oversampling
    use_oversampling = getattr(cfg.training, 'oversample', False)
    if use_oversampling:
        from torch.utils.data import WeightedRandomSampler
        train_labels = [s.label for s in train_samples if s.label is not None]
        class_counts = np.bincount(train_labels, minlength=3)
        sample_weights = [1.0 / class_counts[s.label] for s in train_samples if s.label is not None]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.data.batch_size, sampler=sampler,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.data.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    # Build model
    model = build_model(cfg.model, device)

    # Optimizer and scheduler
    optimizer = build_optimizer(model, cfg.training)
    scheduler = build_scheduler(optimizer, cfg.training)

    # Loss
    class_weights = None
    if cfg.training.class_weights is not None:
        class_weights = torch.tensor(cfg.training.class_weights, dtype=torch.float32)
    loss_type = getattr(cfg.training, 'loss_type', 'cross_entropy')
    focal_gamma = getattr(cfg.training, 'focal_gamma', 2.0)
    criterion = get_loss_function(
        class_weights=class_weights,
        label_smoothing=cfg.training.label_smoothing,
        device=device,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, criterion, device, cfg)
    best_ckpt = trainer.fit()

    return best_ckpt


def generate_rsh_predictions(cfg, checkpoint_path, device, logger):
    """Generate RSH predictions from a single model."""
    model = load_model_checkpoint(cfg.model, checkpoint_path, device)
    model.eval()

    # Load RSH samples
    rsh_dir = Path(cfg.data.root_dir) / "RSH" / "data_unilateral"
    samples = []
    for uid_dir in sorted(rsh_dir.iterdir()):
        if not uid_dir.is_dir():
            continue
        image_paths = {}
        all_exist = True
        for seq in cfg.data.sequences:
            p = uid_dir / f"{seq}.nii.gz"
            image_paths[seq] = str(p)
            if not p.exists():
                all_exist = False
        if all_exist:
            samples.append(SampleInfo(uid=uid_dir.name, image_paths=image_paths, label=None, institution="RSH"))

    use_pct = getattr(cfg.augmentation, 'use_percentile_norm', False)
    crop_fg = getattr(cfg.augmentation, 'crop_foreground', False)
    transforms = get_val_transforms(cfg.data.sequences, cfg.data.spatial_size,
                                     use_percentile_norm=use_pct, crop_foreground=crop_fg)
    dataset = BreastMRIDataset(samples, transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    all_logits, all_uids = [], []
    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"
    amp_dtype = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            with torch.amp.autocast(amp_dtype, enabled=amp_enabled):
                logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_uids.extend(batch["uid"])

    return np.concatenate(all_logits), all_uids


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out training + ensemble")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <output_dir>/predictions_loo_ensemble.csv)")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides if overrides else None)
    logger = setup_logging(cfg.paths.log_dir)
    device = get_device(cfg.device)

    output_path = args.output or str(Path(cfg.paths.output_dir) / "predictions_loo_ensemble.csv")

    # Train 4 LOO models
    checkpoints = {}
    for val_inst in INSTITUTIONS:
        seed_everything(cfg.seed)
        ckpt = train_one_loo(cfg, val_inst, device, logger)
        if ckpt:
            checkpoints[val_inst] = ckpt

    logger.info(f"\nTrained {len(checkpoints)} LOO models")

    # Generate ensemble predictions for RSH
    all_model_logits = []
    uids = None
    for val_inst, ckpt_path in checkpoints.items():
        logger.info(f"Generating RSH predictions from LOO-{val_inst} model")
        logits, model_uids = generate_rsh_predictions(cfg, ckpt_path, device, logger)
        all_model_logits.append(logits)
        if uids is None:
            uids = model_uids

    # Average logits across models, then softmax
    ensemble_logits = np.mean(all_model_logits, axis=0)
    probs = softmax(ensemble_logits, axis=1)

    # Write CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "normal", "benign", "malignant"])
        for uid, prob in zip(uids, probs):
            writer.writerow([uid, f"{prob[0]:.4f}", f"{prob[1]:.4f}", f"{prob[2]:.4f}"])

    print(f"\nSaved LOO ensemble predictions ({len(checkpoints)} models) to {output_path}")
    print(f"Preview:")
    with open(output_path) as f:
        for i, line in enumerate(f):
            print(f"  {line.strip()}")
            if i >= 5:
                break


if __name__ == "__main__":
    main()

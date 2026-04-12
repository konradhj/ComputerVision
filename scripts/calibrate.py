#!/usr/bin/env python3
"""
Fit temperature scaling on the validation set for probability calibration.

Usage:
    python scripts/calibrate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.temperature_scaling import TemperatureScaler, compute_ece
from src.data.dataset import BreastMRIDataset, build_sample_list
from src.data.transforms import get_val_transforms
from src.evaluation.metrics import compute_metrics, print_metrics_report
from src.models.classifier import load_model_checkpoint
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import get_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Calibrate model with temperature scaling")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides if overrides else None)
    logger = setup_logging(cfg.paths.log_dir)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    # Load model
    model = load_model_checkpoint(cfg.model, args.checkpoint, device)

    # Build validation dataloader
    samples = build_sample_list(
        cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
        cfg.data.sequences, cfg.data.fold, "val",
    )
    transforms = get_val_transforms(cfg.data.sequences, cfg.data.spatial_size)
    dataset = BreastMRIDataset(samples, transforms)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # Collect logits for ECE comparison
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)

    # ECE before calibration
    ece_before = compute_ece(logits_np, labels_np, temperature=1.0)
    print(f"\n{'='*50}")
    print(f"Before calibration:")
    print(f"  ECE: {ece_before:.4f}")

    # Metrics before
    metrics_before = compute_metrics(logits_np, labels_np)
    print(f"  Accuracy: {metrics_before['accuracy']:.4f}")
    if metrics_before.get("malignant_auroc"):
        print(f"  AUROC: {metrics_before['malignant_auroc']:.4f}")

    # Fit temperature
    scaler = TemperatureScaler(init_temperature=cfg.calibration.temperature_init)
    learned_temp = scaler.fit(model, val_loader, device)

    # ECE after calibration
    ece_after = compute_ece(logits_np, labels_np, temperature=learned_temp)
    print(f"\nAfter calibration (T={learned_temp:.4f}):")
    print(f"  ECE: {ece_after:.4f}")
    print(f"  ECE improvement: {ece_before - ece_after:.4f}")

    # Metrics after (AUROC unchanged, accuracy may change slightly)
    scaled_logits = logits_np / learned_temp
    metrics_after = compute_metrics(scaled_logits, labels_np)
    print(f"  Accuracy: {metrics_after['accuracy']:.4f}")
    if metrics_after.get("malignant_auroc"):
        print(f"  AUROC: {metrics_after['malignant_auroc']:.4f}")
    print(f"{'='*50}\n")

    # Save temperature
    cal_dir = Path(cfg.paths.output_dir) / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(str(cal_dir / "temperature.pt"))

    print(f"Temperature saved to {cal_dir / 'temperature.pt'}")
    print(f"Use with: --temperature {cal_dir / 'temperature.pt'}")


if __name__ == "__main__":
    main()

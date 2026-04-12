#!/usr/bin/env python3
"""
Evaluate a trained breast MRI classifier on validation or test data.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
    python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --split test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.temperature_scaling import compute_ece
from src.data.dataset import BreastMRIDataset, build_sample_list
from src.data.transforms import get_val_transforms
from src.evaluation.metrics import (
    compute_metrics,
    compute_metrics_per_institution,
    print_metrics_report,
)
from src.models.classifier import load_model_checkpoint
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import get_device, seed_everything


@torch.no_grad()
def run_inference(model, dataloader, device, amp_enabled=False):
    """Run model on a dataloader, collecting logits, labels, UIDs, and institutions."""
    model.eval()
    all_logits, all_labels, all_uids = [], [], []

    amp_dtype = "cuda" if device.type == "cuda" else "cpu"

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        with torch.amp.autocast(amp_dtype, enabled=amp_enabled):
            logits = model(images)

        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_uids.extend(batch["uid"])

    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        all_uids,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate breast MRI classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--temperature", type=str, default=None,
                        help="Path to temperature calibration file")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides if overrides else None)
    logger = setup_logging(cfg.paths.log_dir)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    # Load model
    model = load_model_checkpoint(cfg.model, args.checkpoint, device)

    # Build dataloader
    samples = build_sample_list(
        cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
        cfg.data.sequences, cfg.data.fold, args.split,
    )
    transforms = get_val_transforms(cfg.data.sequences, cfg.data.spatial_size)
    dataset = BreastMRIDataset(samples, transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # Run inference
    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"
    logits, labels, uids = run_inference(model, dataloader, device, amp_enabled)

    # Filter out unlabeled samples (label == -1)
    labeled_mask = labels >= 0
    if not labeled_mask.all():
        logger.info(f"Filtering {(~labeled_mask).sum()} unlabeled samples")
        logits_labeled = logits[labeled_mask]
        labels_labeled = labels[labeled_mask]
    else:
        logits_labeled = logits
        labels_labeled = labels

    # Compute metrics
    metrics = compute_metrics(
        logits_labeled, labels_labeled,
        sensitivity_threshold=cfg.evaluation.sensitivity_threshold,
        specificity_threshold=cfg.evaluation.specificity_threshold,
    )
    print_metrics_report(metrics)

    # ECE
    ece_before = compute_ece(logits_labeled, labels_labeled, temperature=1.0)
    print(f"ECE (no calibration): {ece_before:.4f}")

    # Optional temperature scaling
    if args.temperature is not None:
        from src.calibration.temperature_scaling import TemperatureScaler
        scaler = TemperatureScaler()
        scaler.load(args.temperature)
        temp = scaler.temperature.item()

        ece_after = compute_ece(logits_labeled, labels_labeled, temperature=temp)
        print(f"ECE (T={temp:.4f}): {ece_after:.4f}")

    # Per-institution breakdown
    institutions = [s.institution for s in samples]
    if labeled_mask is not None:
        institutions_labeled = [inst for inst, m in zip(institutions, labeled_mask) if m]
    else:
        institutions_labeled = institutions

    if len(set(institutions_labeled)) > 1:
        print("\nPer-institution breakdown:")
        inst_metrics = compute_metrics_per_institution(
            logits_labeled, labels_labeled, institutions_labeled
        )
        for inst, m in inst_metrics.items():
            print(f"  {inst}: accuracy={m['accuracy']:.4f}, n={m['n_samples']}")
            if m.get("malignant_auroc") is not None:
                print(f"    AUROC={m['malignant_auroc']:.4f}")

    # Save results
    output_path = Path(cfg.paths.output_dir) / f"{args.split}_metrics.json"
    serializable = {k: v for k, v in metrics.items()}
    serializable["ece"] = ece_before
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate predictions.csv for leaderboard submission.

Runs inference on RSH samples (or any specified institution) and outputs
a CSV with columns: ID, normal, benign, malignant

Usage:
    python scripts/generate_submission.py --config configs/idun_v2.yaml \
        --checkpoint /path/to/best.pt \
        --output predictions.csv

    With temperature scaling:
    python scripts/generate_submission.py --config configs/idun_v2.yaml \
        --checkpoint /path/to/best.pt \
        --temperature /path/to/temperature.pt \
        --output predictions.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BreastMRIDataset, SampleInfo, _resolve_data_path
from src.data.transforms import get_val_transforms
from src.models.classifier import load_model_checkpoint
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import get_device, seed_everything


def get_rsh_samples(data_root: str, sequences: list) -> list:
    """Find all RSH samples that have data on disk."""
    rsh_dir = Path(data_root) / "RSH" / "data_unilateral"
    if not rsh_dir.exists():
        raise FileNotFoundError(f"RSH data directory not found: {rsh_dir}")

    samples = []
    for uid_dir in sorted(rsh_dir.iterdir()):
        if not uid_dir.is_dir():
            continue
        uid = uid_dir.name

        # Check all sequences exist
        image_paths = {}
        all_exist = True
        for seq in sequences:
            seq_path = uid_dir / f"{seq}.nii.gz"
            image_paths[seq] = str(seq_path)
            if not seq_path.exists():
                all_exist = False

        if all_exist:
            samples.append(SampleInfo(
                uid=uid,
                image_paths=image_paths,
                label=None,
                institution="RSH",
            ))

    return samples


@torch.no_grad()
def predict_all(model, dataloader, device, amp_enabled=False):
    """Run model on all samples, return logits and UIDs."""
    model.eval()
    all_logits = []
    all_uids = []

    amp_dtype = "cuda" if device.type == "cuda" else "cpu"

    for batch in tqdm(dataloader, desc="Running inference"):
        images = batch["image"].to(device)
        with torch.amp.autocast(amp_dtype, enabled=amp_enabled):
            logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_uids.extend(batch["uid"])

    return np.concatenate(all_logits), all_uids


def main():
    parser = argparse.ArgumentParser(description="Generate submission CSV")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperature", type=str, default=None)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--institution", type=str, default="RSH",
                        help="Which institution to predict on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg.paths.log_dir)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    # Load model
    model = load_model_checkpoint(cfg.model, args.checkpoint, device)

    # Get RSH samples directly from disk
    samples = get_rsh_samples(cfg.data.root_dir, cfg.data.sequences)
    logger.info(f"Found {len(samples)} {args.institution} samples with all sequences")

    # Build dataloader (use same normalization as training)
    use_pct = getattr(cfg.augmentation, 'use_percentile_norm', False)
    transforms = get_val_transforms(cfg.data.sequences, cfg.data.spatial_size, use_percentile_norm=use_pct)
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
    logits, uids = predict_all(model, dataloader, device, amp_enabled)

    # Optional temperature scaling
    if args.temperature is not None:
        from src.calibration.temperature_scaling import TemperatureScaler
        scaler = TemperatureScaler()
        scaler.load(args.temperature)
        temp = scaler.temperature.item()
        logger.info(f"Applying temperature scaling: T={temp:.4f}")
        logits = logits / temp

    # Convert to probabilities
    probs = softmax(logits, axis=1)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "normal", "benign", "malignant"])
        for uid, prob in zip(uids, probs):
            writer.writerow([uid, f"{prob[0]:.4f}", f"{prob[1]:.4f}", f"{prob[2]:.4f}"])

    logger.info(f"Saved {len(uids)} predictions to {output_path}")
    print(f"\nSaved {len(uids)} predictions to {output_path}")
    print(f"Preview:")
    with open(output_path) as f:
        for i, line in enumerate(f):
            print(f"  {line.strip()}")
            if i >= 5:
                print(f"  ... ({len(uids)} rows total)")
                break


if __name__ == "__main__":
    main()

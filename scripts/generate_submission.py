#!/usr/bin/env python3
"""
Generate predictions.csv for leaderboard submission.

Runs inference on RSH samples and outputs a CSV with columns: ID, normal, benign, malignant.
Supports test-time augmentation (TTA) for more robust predictions.

Usage:
    python scripts/generate_submission.py --config configs/idun_v5.yaml \
        --checkpoint /path/to/best.pt --output predictions.csv

    With TTA (8 augmented passes):
    python scripts/generate_submission.py --config configs/idun_v5.yaml \
        --checkpoint /path/to/best.pt --output predictions.csv --tta 8

    With temperature scaling:
    python scripts/generate_submission.py --config configs/idun_v5.yaml \
        --checkpoint /path/to/best.pt --temperature /path/to/temperature.pt \
        --output predictions.csv --tta 8
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

from src.data.dataset import BreastMRIDataset, SampleInfo
from src.data.transforms import get_train_transforms, get_val_transforms
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


@torch.no_grad()
def predict_with_tta(model, samples, cfg, device, n_tta=8):
    """
    Test-Time Augmentation: run N forward passes with random augmentations,
    average the logits for more robust predictions.
    """
    model.eval()
    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"
    amp_dtype = "cuda" if device.type == "cuda" else "cpu"
    use_pct = getattr(cfg.augmentation, 'use_percentile_norm', False)

    # First pass: no augmentation (deterministic)
    val_transforms = get_val_transforms(cfg.data.sequences, cfg.data.spatial_size, use_percentile_norm=use_pct)
    dataset = BreastMRIDataset(samples, val_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )
    logits_base, uids = predict_all(model, dataloader, device, amp_enabled)
    all_logits = [logits_base]

    # Augmented passes
    noise_prob = getattr(cfg.augmentation, 'rand_gaussian_noise_prob', 0.0)
    noise_std = getattr(cfg.augmentation, 'rand_gaussian_noise_std', 0.05)

    for i in range(n_tta - 1):
        print(f"TTA pass {i+2}/{n_tta}")
        tta_transforms = get_train_transforms(
            sequences=cfg.data.sequences,
            spatial_size=cfg.data.spatial_size,
            rand_flip_prob=0.5,
            rand_rotate90_prob=0.5,
            rand_affine_prob=0.0,       # Skip affine for TTA (too aggressive)
            rand_affine_rotate_range=0.0,
            rand_affine_scale_range=[1.0, 1.0],
            rand_intensity_shift=0.05,  # Mild intensity augmentation
            rand_intensity_scale=0.05,
            use_percentile_norm=use_pct,
            rand_gaussian_noise_prob=0.0,
            rand_gaussian_noise_std=0.0,
        )
        dataset = BreastMRIDataset(samples, tta_transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True,
        )
        logits_aug, _ = predict_all(model, dataloader, device, amp_enabled)
        all_logits.append(logits_aug)

    # Average logits across all passes
    avg_logits = np.mean(all_logits, axis=0)
    print(f"TTA complete: averaged {n_tta} passes")
    return avg_logits, uids


def main():
    parser = argparse.ArgumentParser(description="Generate submission CSV")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperature", type=str, default=None)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--tta", type=int, default=0,
                        help="Number of TTA passes (0=disabled, 8=recommended)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg.paths.log_dir)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    # Load model
    model = load_model_checkpoint(cfg.model, args.checkpoint, device)

    # Get RSH samples directly from disk
    samples = get_rsh_samples(cfg.data.root_dir, cfg.data.sequences)
    logger.info(f"Found {len(samples)} RSH samples with all sequences")

    # Run inference (with or without TTA)
    if args.tta > 1:
        logger.info(f"Running inference with TTA ({args.tta} passes)")
        logits, uids = predict_with_tta(model, samples, cfg, device, n_tta=args.tta)
    else:
        use_pct = getattr(cfg.augmentation, 'use_percentile_norm', False)
        transforms = get_val_transforms(cfg.data.sequences, cfg.data.spatial_size, use_percentile_norm=use_pct)
        dataset = BreastMRIDataset(samples, transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True,
        )
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

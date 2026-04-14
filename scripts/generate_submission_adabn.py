#!/usr/bin/env python3
"""
Generate submission with Adaptive BatchNorm (AdaBN).

Adapts the model's BatchNorm statistics to the RSH test distribution
before running inference. Uses unlabeled RSH images only — no labels needed.

This bridges the domain gap between training institutions and RSH by
updating the running mean/variance in BatchNorm layers to match RSH's
scanner characteristics.

Reference: Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation" (2016)

Usage:
    python scripts/generate_submission_adabn.py \
        --config configs/idun_v14.yaml \
        --checkpoint /path/to/best.pt \
        --output predictions_adabn.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BreastMRIDataset, SampleInfo
from src.data.transforms import get_val_transforms
from src.models.classifier import load_model_checkpoint
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import get_device, seed_everything


def get_rsh_samples(data_root: str, sequences: list) -> list:
    """Find all RSH samples that have data on disk."""
    rsh_dir = Path(data_root) / "RSH" / "data_unilateral"
    samples = []
    for uid_dir in sorted(rsh_dir.iterdir()):
        if not uid_dir.is_dir():
            continue
        image_paths = {}
        all_exist = True
        for seq in sequences:
            p = uid_dir / f"{seq}.nii.gz"
            image_paths[seq] = str(p)
            if not p.exists():
                all_exist = False
        if all_exist:
            samples.append(SampleInfo(uid=uid_dir.name, image_paths=image_paths, label=None, institution="RSH"))
    return samples


def set_bn_training(model: nn.Module):
    """Set all BatchNorm layers to training mode (updates running stats)."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.training = True
            module.reset_running_stats()


def set_bn_eval(model: nn.Module):
    """Set all BatchNorm layers back to eval mode."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.training = False


@torch.no_grad()
def adapt_bn_to_target(model, dataloader, device, amp_enabled=False):
    """
    Adaptive BatchNorm: update BatchNorm statistics using target domain data.

    1. Reset BatchNorm running stats
    2. Run forward passes on target data (RSH) with BN in training mode
    3. BN layers accumulate RSH-specific mean/variance
    """
    # Set BN layers to training mode, everything else stays eval
    model.eval()
    set_bn_training(model)

    amp_dtype = "cuda" if device.type == "cuda" else "cpu"

    print("Adapting BatchNorm to RSH distribution...")
    # Run multiple passes to get stable statistics
    for pass_num in range(3):
        for batch in tqdm(dataloader, desc=f"AdaBN pass {pass_num+1}/3"):
            images = batch["image"].to(device)
            with torch.amp.autocast(amp_dtype, enabled=amp_enabled):
                _ = model(images)

    # Set BN back to eval (use the adapted statistics)
    set_bn_eval(model)
    print("BatchNorm adapted to RSH distribution")


@torch.no_grad()
def predict_all(model, dataloader, device, amp_enabled=False):
    """Run inference and collect logits."""
    model.eval()
    all_logits, all_uids = [], []
    amp_dtype = "cuda" if device.type == "cuda" else "cpu"

    for batch in tqdm(dataloader, desc="Running inference"):
        images = batch["image"].to(device)
        with torch.amp.autocast(amp_dtype, enabled=amp_enabled):
            logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_uids.extend(batch["uid"])

    return np.concatenate(all_logits), all_uids


def main():
    parser = argparse.ArgumentParser(description="Generate submission with AdaBN")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions_adabn.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg.paths.log_dir)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    # Load model
    model = load_model_checkpoint(cfg.model, args.checkpoint, device)

    # Get RSH samples
    samples = get_rsh_samples(cfg.data.root_dir, cfg.data.sequences)
    logger.info(f"Found {len(samples)} RSH samples")

    # Build dataloader
    use_pct = getattr(cfg.augmentation, 'use_percentile_norm', False)
    crop_fg = getattr(cfg.augmentation, 'crop_foreground', False)
    transforms = get_val_transforms(cfg.data.sequences, cfg.data.spatial_size,
                                     use_percentile_norm=use_pct, crop_foreground=crop_fg)
    dataset = BreastMRIDataset(samples, transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"

    # Step 1: Adapt BatchNorm statistics to RSH distribution
    adapt_bn_to_target(model, dataloader, device, amp_enabled)

    # Step 2: Run inference with adapted model
    logits, uids = predict_all(model, dataloader, device, amp_enabled)
    probs = softmax(logits, axis=1)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "normal", "benign", "malignant"])
        for uid, prob in zip(uids, probs):
            writer.writerow([uid, f"{prob[0]:.4f}", f"{prob[1]:.4f}", f"{prob[2]:.4f}"])

    print(f"\nSaved {len(uids)} AdaBN predictions to {output_path}")
    print("Preview:")
    with open(output_path) as f:
        for i, line in enumerate(f):
            print(f"  {line.strip()}")
            if i >= 5:
                break


if __name__ == "__main__":
    main()

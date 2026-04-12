#!/usr/bin/env python3
"""
Run inference and generate bilateral JSON submission files.

This script:
1. Loads a trained unilateral classifier
2. Predicts on all test samples (left and right breasts)
3. Groups predictions by study
4. Writes bilateral-breast-classification-likelihoods.json per study

Usage:
    python scripts/inference.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
    python scripts/inference.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --output_dir submissions/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BreastMRIDataset, build_sample_list
from src.data.transforms import get_val_transforms
from src.evaluation.bilateral import assemble_bilateral_predictions, save_bilateral_json
from src.models.classifier import load_model_checkpoint
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import get_device, seed_everything


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
    parser = argparse.ArgumentParser(description="Generate bilateral submission")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="submissions/")
    parser.add_argument("--temperature", type=str, default=None,
                        help="Path to temperature calibration file")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides if overrides else None)
    logger = setup_logging(cfg.paths.log_dir)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    # Load model
    model = load_model_checkpoint(cfg.model, args.checkpoint, device)

    # Build dataloader (no labels needed)
    samples = build_sample_list(
        cfg.data.split_csv,
        label_csv=None,  # No labels needed for inference
        root_dir=cfg.data.root_dir,
        sequences=cfg.data.sequences,
        fold=cfg.data.fold,
        split=args.split,
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

    logger.info(f"Running inference on {len(dataset)} samples ({args.split} split)")

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

    # Assemble bilateral predictions
    bilateral_preds = assemble_bilateral_predictions(uids, logits, apply_softmax=True)

    # Save JSON files
    saved_paths = save_bilateral_json(bilateral_preds, args.output_dir)

    # Summary
    print(f"\nGenerated {len(saved_paths)} bilateral prediction files in {args.output_dir}")
    print(f"\nExample prediction for first study:")
    first_study = list(bilateral_preds.keys())[0]
    import json
    print(json.dumps({first_study: bilateral_preds[first_study]}, indent=2))


if __name__ == "__main__":
    main()

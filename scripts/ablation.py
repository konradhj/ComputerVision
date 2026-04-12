#!/usr/bin/env python3
"""
Sequence ablation study: train with different MRI sequence subsets
to measure each sequence's contribution.

Usage:
    python scripts/ablation.py --config configs/default.yaml
    python scripts/ablation.py --config configs/default.yaml --epochs 30
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_dataloaders, build_sample_list
from src.data.label_mapping import compute_class_weights
from src.evaluation.metrics import compute_metrics
from src.models.classifier import build_model
from src.training.losses import get_loss_function
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import get_device, seed_everything

# Sequence subsets to test
ABLATION_SETS = [
    ["Pre", "Post_1", "Post_2", "T2"],  # Full baseline
    ["Pre", "Post_1", "Post_2"],         # No T2
    ["Pre", "Post_1", "T2"],             # No Post_2
    ["Pre", "T2"],                       # Minimal: pre-contrast + T2
    ["Post_1", "Post_2"],                # Only post-contrast
    ["T2"],                              # T2 only
    ["Pre"],                             # Pre only
]


def run_one_ablation(cfg, sequences, device, logger):
    """Train and evaluate with a specific sequence subset."""
    # Override sequences and in_channels
    cfg.data.sequences = sequences
    cfg.model.in_channels = len(sequences)

    logger.info(f"\n{'='*60}")
    logger.info(f"Ablation: sequences={sequences} ({len(sequences)} channels)")
    logger.info(f"{'='*60}")

    # Build dataloaders
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Class weights
    train_samples = build_sample_list(
        cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
        cfg.data.sequences, cfg.data.fold, "train",
    )
    train_labels = [s.label for s in train_samples if s.label is not None]
    class_weights = compute_class_weights(train_labels) if train_labels else None

    # Build model
    model = build_model(cfg.model, device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    criterion = get_loss_function(class_weights, cfg.training.label_smoothing, device)

    # Train
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, criterion, device, cfg)
    best_ckpt = trainer.fit()

    # Evaluate with best checkpoint
    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_logits, val_labels, _ = trainer.validate(epoch=0)

    metrics = compute_metrics(val_logits, val_labels)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Sequence ablation study")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per ablation run")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides if overrides else None)
    cfg.training.epochs = args.epochs  # Shorter training for ablation

    logger = setup_logging(cfg.paths.log_dir)
    device = get_device(cfg.device)

    results = []

    for sequences in ABLATION_SETS:
        seed_everything(cfg.seed)  # Reset seed for fair comparison

        # Set unique output dir for each run
        seq_name = "+".join(sequences)
        cfg.paths.output_dir = f"outputs/ablation/{seq_name}/"
        cfg.paths.checkpoint_dir = f"outputs/ablation/{seq_name}/checkpoints/"
        cfg.paths.log_dir = f"outputs/ablation/{seq_name}/logs/"
        Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)

        try:
            metrics = run_one_ablation(cfg, sequences, device, logger)
            results.append({
                "sequences": seq_name,
                "n_channels": len(sequences),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "malignant_auroc": metrics.get("malignant_auroc"),
            })
        except Exception as e:
            logger.error(f"Ablation failed for {sequences}: {e}")
            results.append({
                "sequences": seq_name,
                "n_channels": len(sequences),
                "accuracy": None,
                "macro_f1": None,
                "malignant_auroc": None,
            })

    # Print comparison table
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print(f"{'Sequences':<35} {'Channels':>8} {'Accuracy':>10} {'Macro F1':>10} {'AUROC':>10}")
    print("-" * 80)
    for r in results:
        acc = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "FAILED"
        f1 = f"{r['macro_f1']:.4f}" if r['macro_f1'] is not None else "FAILED"
        auroc = f"{r['malignant_auroc']:.4f}" if r.get('malignant_auroc') is not None else "N/A"
        print(f"{r['sequences']:<35} {r['n_channels']:>8} {acc:>10} {f1:>10} {auroc:>10}")
    print("=" * 80)

    # Save results
    output_path = Path("outputs/ablation/ablation_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

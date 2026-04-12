#!/usr/bin/env python3
"""
Train a unilateral breast MRI classifier.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml training.epochs=50 data.batch_size=2
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_dataloaders, build_sample_list
from src.data.label_mapping import compute_class_weights
from src.evaluation.metrics import compute_metrics, print_metrics_report
from src.models.classifier import build_model
from src.training.losses import get_loss_function
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import get_device, seed_everything


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    """Build learning rate scheduler from config."""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.scheduler_patience, factor=0.5
        )
    elif cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif cfg.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def main():
    parser = argparse.ArgumentParser(description="Train breast MRI classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    args, overrides = parser.parse_known_args()

    # Load config with CLI overrides
    cfg = load_config(args.config, overrides if overrides else None)

    # Setup
    logger = setup_logging(cfg.paths.log_dir)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    logger.info(f"Config: {args.config}")
    logger.info(f"Overrides: {overrides}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {cfg.seed}")

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # Compute class weights from training labels if needed
    class_weights = None
    if cfg.training.class_weights is not None:
        class_weights = torch.tensor(cfg.training.class_weights, dtype=torch.float32)
    else:
        # Auto-compute from training set
        train_samples = build_sample_list(
            cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
            cfg.data.sequences, cfg.data.fold, "train",
        )
        train_labels = [s.label for s in train_samples if s.label is not None]
        if train_labels:
            class_weights = compute_class_weights(train_labels)

    # Build model
    model = build_model(cfg.model, device)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg.training)
    scheduler = build_scheduler(optimizer, cfg.training)

    # Build loss function
    criterion = get_loss_function(
        class_weights=class_weights,
        label_smoothing=cfg.training.label_smoothing,
        device=device,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        cfg=cfg,
    )

    best_ckpt = trainer.fit()

    # Final evaluation on validation set using best checkpoint
    logger.info("Running final evaluation on validation set...")
    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_logits, val_labels, val_uids = trainer.validate(epoch=0)

    metrics = compute_metrics(
        val_logits, val_labels,
        sensitivity_threshold=cfg.evaluation.sensitivity_threshold,
        specificity_threshold=cfg.evaluation.specificity_threshold,
    )
    print_metrics_report(metrics)

    # Save final metrics
    metrics_path = Path(cfg.paths.output_dir) / "final_val_metrics.json"
    serializable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    serializable["confusion_matrix"] = metrics["confusion_matrix"]
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save config used for this run
    import yaml
    config_save_path = Path(cfg.paths.output_dir) / "config_used.yaml"
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    with open(config_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


if __name__ == "__main__":
    main()

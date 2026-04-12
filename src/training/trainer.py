"""
Training and validation loops for breast MRI classification.

Plain PyTorch loop with mixed precision, early stopping, and checkpointing.
No frameworks — easy to understand and present.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.config import Config
from ..utils.logging_utils import MetricLogger

logger = logging.getLogger("breast_mri")


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True
        return False


class Trainer:
    """
    Handles the full training lifecycle: training loop, validation,
    checkpointing, early stopping, and metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: nn.Module,
        device: torch.device,
        cfg: Config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = cfg

        # Mixed precision
        use_amp = cfg.training.mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.amp_enabled = use_amp
        self.amp_device_type = "cuda" if use_amp else "cpu"

        # Early stopping
        self.early_stopping = EarlyStopping(patience=cfg.training.early_stopping_patience)

        # Metric logging
        self.metric_logger = MetricLogger(cfg.paths.output_dir)

        # Checkpoint directory
        self.ckpt_dir = Path(cfg.paths.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch. Returns average loss and accuracy."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(self.amp_device_type, enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

            # Log batch metrics
            self.metric_logger.update("train", {
                "loss": loss.item(),
                "accuracy": (preds == labels).float().mean().item(),
            }, batch_size=batch_size)

        avg_loss = total_loss / total
        avg_acc = correct / total

        return {"loss": avg_loss, "accuracy": avg_acc}

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[str]]:
        """
        Run validation. Returns metrics dict plus all logits, labels, and UIDs
        for epoch-level metric computation (e.g., AUROC).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_logits = []
        all_labels = []
        all_uids = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast(self.amp_device_type, enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_uids.extend(batch["uid"])

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

            self.metric_logger.update("val", {
                "loss": loss.item(),
                "accuracy": (preds == labels).float().mean().item(),
            }, batch_size=batch_size)

        avg_loss = total_loss / total
        avg_acc = correct / total

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = {"loss": avg_loss, "accuracy": avg_acc}
        return metrics, all_logits, all_labels, all_uids

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        # Always save latest
        torch.save(state, self.ckpt_dir / "latest.pt")

        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")
            logger.info(f"  Saved best checkpoint (val_loss={val_loss:.4f})")

    def fit(self) -> str:
        """
        Main training loop. Returns path to the best checkpoint.

        Runs for cfg.training.epochs or until early stopping triggers.
        """
        logger.info(f"Starting training for up to {self.cfg.training.epochs} epochs")
        logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Val samples:   {len(self.val_loader.dataset)}")
        logger.info(f"  Batch size:    {self.cfg.data.batch_size}")
        logger.info(f"  Mixed precision: {self.amp_enabled}")

        best_ckpt_path = str(self.ckpt_dir / "best.pt")

        for epoch in range(self.cfg.training.epochs):
            start = time.time()

            # Train
            train_metrics = self.train_one_epoch(epoch)
            train_summary = self.metric_logger.epoch_summary("train", epoch)

            # Validate
            val_metrics, val_logits, val_labels, val_uids = self.validate(epoch)
            val_summary = self.metric_logger.epoch_summary("val", epoch)

            # Step the learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start

            # Check if best
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics["loss"], is_best)

            # Log
            logger.info(
                f"Epoch {epoch+1}/{self.cfg.training.epochs} "
                f"({elapsed:.0f}s) — "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"lr={current_lr:.2e}"
                + (" *" if is_best else "")
            )

            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                break

        # Save training history
        self.metric_logger.save_history()
        logger.info(f"Training complete. Best val_loss={self.best_val_loss:.4f}")
        logger.info(f"Best checkpoint: {best_ckpt_path}")

        return best_ckpt_path

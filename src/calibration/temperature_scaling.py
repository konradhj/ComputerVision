"""
Temperature scaling for post-hoc calibration.

After training, learns a single scalar T on the validation set such that
calibrated probabilities = softmax(logits / T) are better calibrated.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from torch.utils.data import DataLoader

logger = logging.getLogger("breast_mri")


class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T to calibrate model outputs.

    Higher T → softer (less confident) probabilities.
    Lower T  → sharper (more confident) probabilities.
    T = 1.0  → no change from raw softmax.
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def calibrated_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling then softmax."""
        return torch.softmax(self.forward(logits), dim=1)

    @torch.no_grad()
    def _collect_logits_and_labels(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> tuple:
        """Collect all logits and labels from a dataloader."""
        model.eval()
        all_logits = []
        all_labels = []

        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        return torch.cat(all_logits), torch.cat(all_labels)

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """
        Optimize temperature on the validation set.

        Args:
            model: Trained classifier (used in eval mode).
            val_loader: Validation dataloader.
            device: Compute device.
            max_iter: Maximum optimization steps.
            lr: Learning rate for temperature optimization.

        Returns:
            Learned temperature value.
        """
        # Collect all validation logits
        logits, labels = self._collect_logits_and_labels(model, val_loader, device)

        # Optimize temperature
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _eval():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)

        learned_temp = self.temperature.item()
        logger.info(f"Learned temperature: {learned_temp:.4f}")
        return learned_temp

    def save(self, path: str) -> None:
        """Save temperature parameter to file."""
        torch.save({"temperature": self.temperature.item()}, path)
        logger.info(f"Saved temperature ({self.temperature.item():.4f}) to {path}")

    def load(self, path: str) -> None:
        """Load temperature parameter from file."""
        state = torch.load(path, weights_only=True)
        self.temperature = nn.Parameter(torch.tensor(state["temperature"]))
        logger.info(f"Loaded temperature ({self.temperature.item():.4f}) from {path}")


def compute_ece(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    temperature: float = 1.0,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual accuracy.
    Lower is better. Perfect calibration = 0.0.

    Args:
        logits: Raw model outputs, shape [N, 3].
        labels: Ground truth labels, shape [N].
        n_bins: Number of confidence bins.
        temperature: Temperature for scaling (1.0 = no scaling).

    Returns:
        ECE value (float).
    """
    probs = softmax(logits / temperature, axis=1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_weight = mask.sum() / len(labels)
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)

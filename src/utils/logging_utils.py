"""
Logging utilities: metric accumulation and CSV logging.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    """Configure root logger with console and file handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("breast_mri")
    logger.setLevel(getattr(logging, level))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(Path(log_dir) / "training.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class MetricLogger:
    """
    Accumulates per-batch metrics, computes epoch averages, and saves to CSV.

    Usage:
        logger = MetricLogger("outputs/")
        logger.update("train", {"loss": 0.5, "accuracy": 0.8}, batch_size=4)
        summary = logger.epoch_summary("train", epoch=1)
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._running: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._weights: Dict[str, list] = defaultdict(list)
        self.history: list = []

    def update(self, phase: str, metrics: Dict[str, float], batch_size: int = 1) -> None:
        """Record metrics for one batch."""
        for key, value in metrics.items():
            self._running[phase][key].append(value * batch_size)
        self._weights[phase].append(batch_size)

    def epoch_summary(self, phase: str, epoch: int) -> Dict[str, float]:
        """Compute weighted averages for the epoch and reset accumulators."""
        total_weight = sum(self._weights[phase])
        summary = {"epoch": epoch, "phase": phase}

        for key, values in self._running[phase].items():
            summary[key] = sum(values) / total_weight if total_weight > 0 else 0.0

        self.history.append(summary)
        self._running[phase].clear()
        self._weights[phase].clear()

        return summary

    def save_history(self, filename: Optional[str] = None) -> None:
        """Write accumulated history to CSV."""
        if not self.history:
            return

        path = self.output_dir / (filename or "training_history.csv")
        keys = list(self.history[0].keys())
        # Collect all keys across history
        for entry in self.history:
            for k in entry:
                if k not in keys:
                    keys.append(k)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)

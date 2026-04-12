"""
Reproducibility helpers: seeding and device selection.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """
    Select compute device.

    Args:
        preference: "auto" picks the best available, or specify "cuda", "mps", "cpu".
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)

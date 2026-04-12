"""
Loss functions for breast MRI classification.
"""

from typing import Optional

import torch
import torch.nn as nn


def get_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> nn.CrossEntropyLoss:
    """
    Build CrossEntropyLoss with optional class weighting and label smoothing.

    Args:
        class_weights: Tensor of per-class weights, or None for uniform.
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
        device: Device to place the weight tensor on.

    Returns:
        Configured CrossEntropyLoss.
    """
    weight = class_weights.to(device) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

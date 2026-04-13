"""
Loss functions for breast MRI classification.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance.

    Focuses training on hard, misclassified examples by down-weighting
    easy/confident predictions. Eliminates the need for manual class weights.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to CrossEntropyLoss.
               gamma=2 is the standard choice (reduces easy sample weight by ~100x).
        alpha: Per-class weights (optional). Tensor of shape [num_classes].
        label_smoothing: Label smoothing factor.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw model outputs
            targets: [B] integer class labels
        """
        num_classes = logits.size(1)

        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        # Compute probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_weight = alpha.unsqueeze(0).expand_as(logits)  # [B, C]
            focal_weight = focal_weight * alpha_weight

        # Focal loss = -alpha * (1-p)^gamma * log(p) * target
        loss = -focal_weight * log_probs * smooth_targets
        return loss.sum(dim=1).mean()


def get_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    device: torch.device = torch.device("cpu"),
    loss_type: str = "cross_entropy",
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Build loss function.

    Args:
        class_weights: Per-class weights, or None for uniform.
        label_smoothing: Label smoothing factor.
        device: Device to place weight tensors on.
        loss_type: "cross_entropy" or "focal".
        focal_gamma: Gamma parameter for focal loss (only used if loss_type="focal").

    Returns:
        Configured loss function.
    """
    if loss_type == "focal":
        alpha = class_weights.to(device) if class_weights is not None else None
        return FocalLoss(gamma=focal_gamma, alpha=alpha, label_smoothing=label_smoothing)
    else:
        weight = class_weights.to(device) if class_weights is not None else None
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

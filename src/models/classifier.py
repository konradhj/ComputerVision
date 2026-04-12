"""
3D breast MRI classifier using MONAI backbones.

Supports DenseNet121 (default), ResNet18, and ResNet50.
All models work with 3D multi-channel input and output 3-class logits.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet18, resnet50

from ..utils.config import ModelConfig

logger = logging.getLogger("breast_mri")


class BreastClassifier(nn.Module):
    """
    Wrapper around MONAI 3D classification backbones.

    Input:  [B, C, D, H, W] where C = number of MRI sequences
    Output: [B, num_classes] raw logits (no softmax)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.architecture = cfg.architecture

        if cfg.architecture == "densenet121":
            self.backbone = DenseNet121(
                spatial_dims=3,
                in_channels=cfg.in_channels,
                out_channels=cfg.num_classes,
                dropout_prob=cfg.dropout,
            )
        elif cfg.architecture == "resnet18":
            self.backbone = resnet18(
                spatial_dims=3,
                n_input_channels=cfg.in_channels,
                num_classes=cfg.num_classes,
            )
        elif cfg.architecture == "resnet50":
            self.backbone = resnet50(
                spatial_dims=3,
                n_input_channels=cfg.in_channels,
                num_classes=cfg.num_classes,
            )
        else:
            raise ValueError(
                f"Unknown architecture: '{cfg.architecture}'. "
                f"Choose from: densenet121, resnet18, resnet50"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits [B, num_classes]."""
        return self.backbone(x)


def build_model(cfg: ModelConfig, device: torch.device) -> BreastClassifier:
    """
    Build the classifier and move it to the specified device.

    Logs the model architecture and parameter count.
    """
    model = BreastClassifier(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {cfg.architecture}")
    logger.info(f"  Input channels: {cfg.in_channels}")
    logger.info(f"  Output classes: {cfg.num_classes}")
    logger.info(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")
    logger.info(f"  Device: {device}")

    return model


def load_model_checkpoint(
    cfg: ModelConfig,
    checkpoint_path: str,
    device: torch.device,
) -> BreastClassifier:
    """Load a model from a saved checkpoint."""
    model = build_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Support both direct state_dict and wrapped checkpoint
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model

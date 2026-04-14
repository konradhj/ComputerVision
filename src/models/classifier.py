"""
Breast MRI classifiers.

Supports:
  - 3D models: DenseNet121, ResNet18, ResNet50 (MONAI, trained from scratch)
  - 2D pretrained: SliceClassifier (ImageNet-pretrained ResNet50, processes each slice)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet18, resnet50

from ..utils.config import ModelConfig

logger = logging.getLogger("breast_mri")


class SliceClassifier(nn.Module):
    """
    2D pretrained slice-based classifier for 3D volumes.

    Processes each 2D slice independently through an ImageNet-pretrained backbone,
    then aggregates slice features for volume-level classification.

    This leverages powerful pretrained features (edges, textures, patterns) learned
    from millions of ImageNet images, requiring only fine-tuning on our small dataset.

    Input:  [B, C, D, H, W]  (e.g., [B, 1, 32, 224, 224])
    Output: [B, num_classes]  raw logits
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        import torchvision.models as models

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)

        # Remove the final FC layer — we'll add our own
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # Output: [B, 2048, 1, 1]
        self.feature_dim = 2048

        # Freeze early layers (learn general features), fine-tune later layers
        # conv1, bn1, relu, maxpool, layer1, layer2 = frozen
        # layer3, layer4 = fine-tuned
        frozen_layers = ['0', '1', '2', '3', '4', '5']  # indices in Sequential
        for name, param in self.features.named_parameters():
            layer_idx = name.split('.')[0]
            if layer_idx in frozen_layers:
                param.requires_grad = False

        # Aggregation across slices + classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(self.feature_dim, cfg.num_classes),
        )

        self.in_channels = cfg.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] — 3D volume with C channels

        Returns:
            [B, num_classes] logits
        """
        B, C, D, H, W = x.shape

        # Rearrange: treat each slice as a separate 2D image
        # [B, C, D, H, W] → [B, D, C, H, W] → [B*D, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * D, C, H, W)

        # Repeat single channel to 3 channels (match ImageNet RGB input)
        if C == 1:
            x = x.repeat(1, 3, 1, 1)  # [B*D, 3, H, W]
        elif C != 3:
            # For multi-channel input, use first 3 or pad
            if C < 3:
                x = torch.cat([x, x[:, :3-C]], dim=1)
            else:
                x = x[:, :3]

        # Extract features per slice
        slice_features = self.features(x)  # [B*D, 2048, 1, 1]
        slice_features = slice_features.squeeze(-1).squeeze(-1)  # [B*D, 2048]

        # Reshape back to [B, D, 2048]
        slice_features = slice_features.view(B, D, self.feature_dim)

        # Aggregate across slices (average pooling)
        volume_features = slice_features.mean(dim=1)  # [B, 2048]

        # Classify
        logits = self.classifier(volume_features)  # [B, num_classes]
        return logits


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
                f"Choose from: densenet121, resnet18, resnet50, slice_resnet50"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits [B, num_classes]."""
        return self.backbone(x)


def build_model(cfg: ModelConfig, device: torch.device) -> nn.Module:
    """
    Build the classifier and move it to the specified device.

    Logs the model architecture and parameter count.
    """
    if cfg.architecture == "slice_resnet50":
        model = SliceClassifier(cfg).to(device)
    else:
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
) -> nn.Module:
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

"""
MedicalNet 3D ResNet with pretrained weights for medical image classification.

Uses pretrained 3D-ResNet50 weights from Tencent's MedicalNet project,
trained on diverse medical imaging datasets. Fine-tuned for breast cancer
classification.

Reference: Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis" (2019)
Weights: https://github.com/Tencent/MedicalNet
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("breast_mri")


class MedicalNetConv3d(nn.Module):
    """3D convolution with optional batch norm and ReLU (matching MedicalNet)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MedicalNetResNet(nn.Module):
    """
    3D ResNet from MedicalNet with classification head.

    Backbone pretrained on diverse medical imaging datasets.
    Classification head trained from scratch for breast cancer.
    """

    def __init__(self, block, layers, num_classes=3, in_channels=1, dropout=0.3):
        super().__init__()
        self.inplanes = 64

        # Backbone (matches MedicalNet architecture exactly)
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(2, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        # Classification head (trained from scratch)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * block.expansion, num_classes),
        )

        # Initialize classification head
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def build_medicalnet_resnet50(num_classes=3, in_channels=1, dropout=0.3,
                               pretrain_path=None, device=None):
    """
    Build MedicalNet ResNet50 with optional pretrained weights.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        dropout: Dropout rate before classification head.
        pretrain_path: Path to MedicalNet pretrained weights (.pth file).
        device: Target device.

    Returns:
        Model with pretrained backbone + fresh classification head.
    """
    model = MedicalNetResNet(Bottleneck, [3, 4, 6, 3],
                              num_classes=num_classes,
                              in_channels=in_channels,
                              dropout=dropout)

    if pretrain_path and Path(pretrain_path).exists():
        logger.info(f"Loading MedicalNet pretrained weights from {pretrain_path}")
        pretrain = torch.load(pretrain_path, map_location='cpu', weights_only=False)

        # MedicalNet stores weights under 'state_dict' key
        if 'state_dict' in pretrain:
            pretrain_dict = pretrain['state_dict']
        else:
            pretrain_dict = pretrain

        # Filter: only load weights that match our model (skip segmentation head)
        model_dict = model.state_dict()
        matched = {}
        skipped = []
        for k, v in pretrain_dict.items():
            # Remove 'module.' prefix if present (from DataParallel)
            clean_key = k.replace('module.', '')
            if clean_key in model_dict and v.shape == model_dict[clean_key].shape:
                matched[clean_key] = v
            else:
                skipped.append(k)

        model_dict.update(matched)
        model.load_state_dict(model_dict)

        logger.info(f"  Loaded {len(matched)}/{len(model_dict)} layers from pretrained weights")
        logger.info(f"  Skipped {len(skipped)} layers (segmentation head / shape mismatch)")

        # Freeze early layers, fine-tune later layers + classifier
        frozen = ['conv1', 'bn1', 'layer1', 'layer2']
        for name, param in model.named_parameters():
            if any(name.startswith(f) for f in frozen):
                param.requires_grad = False

        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        n_total = sum(1 for p in model.parameters())
        logger.info(f"  Frozen {n_frozen}/{n_total} parameter groups (conv1, bn1, layer1, layer2)")

    if device:
        model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MedicalNet ResNet50: {n_params:,} total, {n_trainable:,} trainable")

    return model

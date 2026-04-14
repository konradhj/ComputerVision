"""
MONAI transform pipelines for breast MRI preprocessing and augmentation.

Each sample has multiple MRI sequences (e.g., Pre, Post_1, Post_2, T2) stored as
separate NIfTI files. The pipeline loads them individually, normalizes each,
concatenates into a multi-channel volume, resizes, and optionally augments.

Optionally computes derived clinical features from DCE dynamics:
  - Sub_2 = Post_2 - Pre (late subtraction)
  - Washout = Post_1 - Post_2 (washout map: positive = malignant sign)
"""

from typing import Dict, Hashable, List, Mapping, Tuple

import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRangePercentilesd,
)


class ComputeDerivedChannelsd(MapTransform):
    """
    Compute clinically motivated derived channels from DCE-MRI sequences.

    Derived channels (computed BEFORE normalization, from raw loaded data):
      - Sub_2:    Post_2 - Pre   (late subtraction — enhancement at later timepoint)
      - Washout:  Post_1 - Post_2 (washout map — positive values indicate washout,
                                   a hallmark of malignancy)

    These features capture the temporal dynamics of contrast enhancement,
    which is the primary diagnostic criterion for breast cancer on DCE-MRI.
    """

    def __init__(self, derive_sub2: bool = True, derive_washout: bool = True):
        # Not calling super().__init__ with keys since we handle keys manually
        self.derive_sub2 = derive_sub2
        self.derive_washout = derive_washout

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)

        if self.derive_sub2 and "Post_2" in d and "Pre" in d:
            d["Sub_2"] = d["Post_2"] - d["Pre"]

        if self.derive_washout and "Post_1" in d and "Post_2" in d:
            d["Washout"] = d["Post_1"] - d["Post_2"]

        return d


def _get_normalization(sequences: List[str], use_percentile: bool = False):
    """Get normalization transform — percentile-based (robust) or z-score."""
    if use_percentile:
        return ScaleIntensityRangePercentilesd(
            keys=sequences,
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    else:
        return NormalizeIntensityd(keys=sequences, nonzero=True, channel_wise=True)


def _build_sequence_list(
    base_sequences: List[str],
    derive_sub2: bool = False,
    derive_washout: bool = False,
) -> List[str]:
    """Build the full list of channel keys including derived features."""
    all_keys = list(base_sequences)
    if derive_sub2:
        all_keys.append("Sub_2")
    if derive_washout:
        all_keys.append("Washout")
    return all_keys


def get_train_transforms(
    sequences: List[str],
    spatial_size: Tuple[int, int, int],
    rand_flip_prob: float = 0.5,
    rand_rotate90_prob: float = 0.5,
    rand_affine_prob: float = 0.3,
    rand_affine_rotate_range: float = 0.1745,
    rand_affine_scale_range: List[float] = None,
    rand_intensity_shift: float = 0.1,
    rand_intensity_scale: float = 0.1,
    use_percentile_norm: bool = False,
    rand_gaussian_noise_prob: float = 0.0,
    rand_gaussian_noise_std: float = 0.05,
    derive_sub2: bool = False,
    derive_washout: bool = False,
    crop_foreground: bool = False,
) -> Compose:
    """
    Training transform pipeline with augmentation.

    Flow:
        1. Load each sequence NIfTI -> separate arrays
        2. Ensure channel-first: each becomes [1, D, H, W]
        2b. (Optional) Crop to foreground bounding box — removes background/air
        3. (Optional) Compute derived channels (Sub_2, Washout)
        4. Normalize intensity per sequence
        5. Concatenate -> single "image" tensor [C, D, H, W]
        6. Resize to target spatial size
        7. Random augmentations
        8. Ensure output types
    """
    if rand_affine_scale_range is None:
        rand_affine_scale_range = [0.9, 1.1]

    scale_offset = (rand_affine_scale_range[0] - 1.0, rand_affine_scale_range[1] - 1.0)

    # All channel keys (loaded + derived)
    all_keys = _build_sequence_list(sequences, derive_sub2, derive_washout)

    transforms = [
        # 1. Load NIfTI files
        LoadImaged(keys=sequences, image_only=True),

        # 2. Ensure each has a channel dimension: [1, D, H, W]
        EnsureChannelFirstd(keys=sequences),
    ]

    # 3. Compute derived channels (before normalization, from raw intensities)
    if derive_sub2 or derive_washout:
        transforms.append(ComputeDerivedChannelsd(derive_sub2=derive_sub2, derive_washout=derive_washout))

    transforms.extend([
        # 4. Per-sequence intensity normalization (on all channels)
        _get_normalization(all_keys, use_percentile=use_percentile_norm),

        # 5. Concatenate all channels -> "image"
        ConcatItemsd(keys=all_keys, name="image", dim=0),
    ])

    # 5b. Crop foreground on concatenated image — removes air/background
    # Done after concat so all channels are cropped identically
    if crop_foreground:
        transforms.append(
            CropForegroundd(keys=["image"], source_key="image", margin=5)
        )

    transforms.extend([
        # 6. Resize to target spatial size (ensures uniform size after crop)
        Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),

        # 7. Augmentation
        RandFlipd(keys=["image"], prob=rand_flip_prob, spatial_axis=0),
        RandFlipd(keys=["image"], prob=rand_flip_prob, spatial_axis=1),
        RandRotate90d(keys=["image"], prob=rand_rotate90_prob, spatial_axes=(0, 1)),
        RandAffined(
            keys=["image"],
            prob=rand_affine_prob,
            rotate_range=[rand_affine_rotate_range] * 3,
            scale_range=[scale_offset] * 3,
            mode="bilinear",
            padding_mode="zeros",
        ),
        RandScaleIntensityd(keys=["image"], factors=rand_intensity_scale, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=rand_intensity_shift, prob=0.5),
    ])

    if rand_gaussian_noise_prob > 0:
        transforms.append(
            RandGaussianNoised(keys=["image"], prob=rand_gaussian_noise_prob, std=rand_gaussian_noise_std)
        )

    # 8. Ensure correct tensor type
    transforms.append(EnsureTyped(keys=["image"]))

    return Compose(transforms)


def get_val_transforms(
    sequences: List[str],
    spatial_size: Tuple[int, int, int],
    use_percentile_norm: bool = False,
    derive_sub2: bool = False,
    derive_washout: bool = False,
    crop_foreground: bool = False,
) -> Compose:
    """
    Validation/test transform pipeline (no augmentation).
    """
    all_keys = _build_sequence_list(sequences, derive_sub2, derive_washout)

    transforms = [
        LoadImaged(keys=sequences, image_only=True),
        EnsureChannelFirstd(keys=sequences),
    ]

    if derive_sub2 or derive_washout:
        transforms.append(ComputeDerivedChannelsd(derive_sub2=derive_sub2, derive_washout=derive_washout))

    transforms.extend([
        _get_normalization(all_keys, use_percentile=use_percentile_norm),
        ConcatItemsd(keys=all_keys, name="image", dim=0),
    ])

    if crop_foreground:
        transforms.append(
            CropForegroundd(keys=["image"], source_key="image", margin=5)
        )

    transforms.extend([
        Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
        EnsureTyped(keys=["image"]),
    ])

    return Compose(transforms)

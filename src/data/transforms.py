"""
MONAI transform pipelines for breast MRI preprocessing and augmentation.

Each sample has multiple MRI sequences (e.g., Pre, Post_1, Post_2, T2) stored as
separate NIfTI files. The pipeline loads them individually, normalizes each,
concatenates into a multi-channel volume, resizes, and optionally augments.
"""

from typing import List, Tuple

from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
)


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
) -> Compose:
    """
    Training transform pipeline with augmentation.

    Flow:
        1. Load each sequence NIfTI → separate arrays
        2. Ensure channel-first: each becomes [1, D, H, W]
        3. Normalize intensity per sequence (z-score, ignoring zeros)
        4. Concatenate → single "image" tensor [C, D, H, W]
        5. Resize to target spatial size
        6. Random augmentations
        7. Ensure output types
    """
    if rand_affine_scale_range is None:
        rand_affine_scale_range = [0.9, 1.1]

    # Scale range for MONAI: specified as (min_offset, max_offset) from 1.0
    scale_offset = (rand_affine_scale_range[0] - 1.0, rand_affine_scale_range[1] - 1.0)

    return Compose([
        # 1. Load NIfTI files
        LoadImaged(keys=sequences, image_only=True),

        # 2. Ensure each has a channel dimension: [1, D, H, W]
        EnsureChannelFirstd(keys=sequences),

        # 3. Per-sequence intensity normalization
        NormalizeIntensityd(keys=sequences, nonzero=True, channel_wise=True),

        # 4. Concatenate sequences → "image" with C channels
        ConcatItemsd(keys=sequences, name="image", dim=0),

        # 5. Resize to target spatial size
        Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),

        # 6. Augmentation
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

        # 7. Ensure correct tensor type
        EnsureTyped(keys=["image"]),
    ])


def get_val_transforms(
    sequences: List[str],
    spatial_size: Tuple[int, int, int],
) -> Compose:
    """
    Validation/test transform pipeline (no augmentation).

    Same loading, normalization, concatenation, and resizing as training,
    but without any random augmentations.
    """
    return Compose([
        LoadImaged(keys=sequences, image_only=True),
        EnsureChannelFirstd(keys=sequences),
        NormalizeIntensityd(keys=sequences, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=sequences, name="image", dim=0),
        Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
        EnsureTyped(keys=["image"]),
    ])

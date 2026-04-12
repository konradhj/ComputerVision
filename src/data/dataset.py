"""
Dataset class and factory functions for breast MRI classification.

Each sample is a unilateral breast with multiple MRI sequences stored as NIfTI files.
The dataset works with MONAI dictionary transforms.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from monai.data import DataLoader, Dataset

from ..utils.config import Config
from .label_mapping import load_labels, load_labels_from_institutions
from .transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger("breast_mri")


@dataclass
class SampleInfo:
    """Metadata for one unilateral breast sample."""
    uid: str
    image_paths: Dict[str, str]  # sequence_name -> path to .nii.gz
    label: Optional[int]         # None if labels not available (test set)
    institution: str = ""


def _resolve_data_path(root_dir: str, institution: str, uid: str) -> Path:
    """
    Resolve the path to a unilateral breast folder.

    ODELIA2025 layout on IDUN:
        data/<Institution>/data_unilateral/<UID>/

    Also tries fallback patterns for local setups.
    """
    # Primary: ODELIA2025 layout
    primary = Path(root_dir) / institution / "data_unilateral" / uid
    if primary.is_dir():
        return primary

    # Fallbacks for other layouts
    for candidate in [
        Path(root_dir) / "data" / institution / "data_unilateral" / uid,
        Path(root_dir) / uid,
        Path(root_dir) / "data_unilateral" / uid,
    ]:
        if candidate.is_dir():
            return candidate

    return primary


def build_sample_list(
    split_csv: str,
    label_csv: Optional[str],
    root_dir: str,
    sequences: List[str],
    fold: int,
    split: str,
) -> List[SampleInfo]:
    """
    Build a list of SampleInfo from the split CSV and labels.

    Args:
        split_csv: Path to split_unilateral.csv (columns: UID, Fold, Split, Institution).
        label_csv: Path to label CSV, or None for label-free inference.
        root_dir: Root directory containing the data.
        sequences: List of sequence names to load (e.g., ["Pre", "Post_1", "Post_2", "T2"]).
        fold: Which fold to use.
        split: One of "train", "val", "test".

    Returns:
        List of SampleInfo objects.
    """
    df = pd.read_csv(split_csv)

    # Filter by fold and split
    mask = (df["Split"] == split)
    if "Fold" in df.columns:
        mask = mask & (df["Fold"] == fold)
    df_split = df[mask].reset_index(drop=True)

    logger.info(f"Found {len(df_split)} samples for split='{split}', fold={fold}")

    # Load labels if available
    labels: Dict[str, int] = {}
    if label_csv is not None and label_csv == "auto":
        # Auto-load from per-institution annotation CSVs
        institutions = df_split["Institution"].unique().tolist() if "Institution" in df_split.columns else []
        if institutions:
            labels = load_labels_from_institutions(root_dir, institutions)
    elif label_csv is not None and Path(label_csv).exists():
        labels = load_labels(label_csv)
    elif label_csv is not None and label_csv != "auto":
        logger.warning(f"Label file not found: {label_csv}. Proceeding without labels.")

    samples = []
    missing_paths = 0
    missing_labels = 0

    for _, row in df_split.iterrows():
        uid = str(row["UID"])
        institution = str(row.get("Institution", ""))

        # Resolve data directory
        data_dir = _resolve_data_path(root_dir, institution, uid)

        # Build image paths for each sequence
        image_paths = {}
        all_exist = True
        for seq in sequences:
            seq_path = data_dir / f"{seq}.nii.gz"
            image_paths[seq] = str(seq_path)
            if not seq_path.exists():
                all_exist = False

        if not all_exist:
            missing_paths += 1
            continue  # Skip samples with missing sequence files

        # Get label
        label = labels.get(uid, None)
        if label is None and labels:
            missing_labels += 1
            if split in ("train", "val"):
                continue  # Skip unlabeled samples for training/validation

        samples.append(SampleInfo(
            uid=uid,
            image_paths=image_paths,
            label=label,
            institution=institution,
        ))

    total_in_csv = len(df_split)
    if missing_paths > 0:
        logger.warning(f"Skipped {missing_paths}/{total_in_csv} samples with missing sequence files")
    if missing_labels > 0 and labels:
        logger.warning(f"Skipped {missing_labels}/{total_in_csv} unlabeled samples (split={split})")
    logger.info(f"Using {len(samples)}/{total_in_csv} samples for split='{split}'")

    return samples


def _sample_to_dict(sample: SampleInfo) -> Dict:
    """Convert a SampleInfo to a dict that MONAI transforms expect."""
    d = {}
    # Image paths keyed by sequence name (LoadImaged will load these)
    for seq_name, seq_path in sample.image_paths.items():
        d[seq_name] = seq_path
    # Metadata
    d["label"] = sample.label if sample.label is not None else -1
    d["uid"] = sample.uid
    return d


class BreastMRIDataset(Dataset):
    """
    Dataset for unilateral breast MRI classification.

    Each __getitem__ returns a dict with:
        "image": torch.Tensor of shape [C, D, H, W] (after transforms)
        "label": int (0=normal, 1=benign, 2=malignant, -1=unknown)
        "uid": str
    """

    def __init__(self, samples: List[SampleInfo], transform):
        data = [_sample_to_dict(s) for s in samples]
        super().__init__(data=data, transform=transform)


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Build train, validation, and (optional) test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader_or_None)
    """
    # Build transform pipelines
    train_transforms = get_train_transforms(
        sequences=cfg.data.sequences,
        spatial_size=cfg.data.spatial_size,
        rand_flip_prob=cfg.augmentation.rand_flip_prob,
        rand_rotate90_prob=cfg.augmentation.rand_rotate90_prob,
        rand_affine_prob=cfg.augmentation.rand_affine_prob,
        rand_affine_rotate_range=cfg.augmentation.rand_affine_rotate_range,
        rand_affine_scale_range=cfg.augmentation.rand_affine_scale_range,
        rand_intensity_shift=cfg.augmentation.rand_intensity_shift,
        rand_intensity_scale=cfg.augmentation.rand_intensity_scale,
    )
    val_transforms = get_val_transforms(
        sequences=cfg.data.sequences,
        spatial_size=cfg.data.spatial_size,
    )

    # Build sample lists
    train_samples = build_sample_list(
        cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
        cfg.data.sequences, cfg.data.fold, "train",
    )
    val_samples = build_sample_list(
        cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
        cfg.data.sequences, cfg.data.fold, "val",
    )

    # Build datasets
    train_ds = BreastMRIDataset(train_samples, train_transforms)
    val_ds = BreastMRIDataset(val_samples, val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # Optional test set
    test_loader = None
    try:
        test_samples = build_sample_list(
            cfg.data.split_csv, cfg.data.label_csv, cfg.data.root_dir,
            cfg.data.sequences, cfg.data.fold, "test",
        )
        if test_samples:
            test_ds = BreastMRIDataset(test_samples, val_transforms)
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=True,
            )
    except Exception as e:
        logger.warning(f"Could not build test dataloader: {e}")

    return train_loader, val_loader, test_loader

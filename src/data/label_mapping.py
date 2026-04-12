"""
Label loading and UID parsing for breast MRI classification.

Handles three label sources:
  1. Per-institution annotation CSVs: data/<Institution>/metadata_unilateral/annotation.csv
     Columns: UID, PatientID, Age, Lesion
  2. Single unilateral CSV: columns (UID, Lesion)
  3. Bilateral CSV: columns (studyID, Lesion_Left, Lesion_Right)

Label mapping: 0 = normal, 1 = benign, 2 = malignant
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

CLASS_NAMES = {0: "normal", 1: "benign", 2: "malignant"}
NUM_CLASSES = 3

logger = logging.getLogger("breast_mri")


def extract_study_id_and_side(uid: str) -> Tuple[str, str]:
    """
    Parse a unilateral UID into (study_id, side).

    Example:
        'ODELIA_BRAID1_0158_1_left' -> ('ODELIA_BRAID1_0158_1', 'left')

    The side is always the last segment after the final underscore.
    """
    last_underscore = uid.rfind("_")
    if last_underscore == -1:
        raise ValueError(f"Cannot parse UID '{uid}': no underscore found")

    study_id = uid[:last_underscore]
    side = uid[last_underscore + 1:]

    if side not in ("left", "right"):
        raise ValueError(f"Cannot parse UID '{uid}': expected 'left' or 'right', got '{side}'")

    return study_id, side


def load_labels_from_institutions(data_root: str, institutions: List[str]) -> Dict[str, int]:
    """
    Load labels from per-institution annotation CSVs.

    Looks for: data_root/<institution>/metadata_unilateral/annotation.csv
    Each CSV has columns: UID, PatientID, Age, Lesion

    Institutions without an annotation.csv (e.g., RSH) are skipped —
    their samples will have no label (used for test/inference only).

    Args:
        data_root: Root data directory (e.g., /cluster/.../ODELIA2025/data/).
        institutions: List of institution names (e.g., ["CAM", "MHA", "RSH", "RUMC", "UKA"]).

    Returns:
        Dict mapping UID -> integer label (0, 1, or 2).
    """
    labels = {}
    for inst in institutions:
        csv_path = Path(data_root) / inst / "metadata_unilateral" / "annotation.csv"
        if not csv_path.exists():
            logger.info(f"No annotation.csv for {inst} — skipping (test/hidden labels)")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            labels[str(row["UID"])] = int(row["Lesion"])
        logger.info(f"Loaded {len(df)} labels from {inst}")

    logger.info(f"Total labels loaded: {len(labels)}")
    return labels


def load_labels(label_csv: str) -> Dict[str, int]:
    """
    Load labels from a single CSV file. Auto-detects format based on column names.

    Returns:
        Dict mapping unilateral UID -> integer label (0, 1, or 2).
    """
    df = pd.read_csv(label_csv)
    columns = set(df.columns)
    labels = {}

    if "Lesion" in columns and "UID" in columns:
        # Unilateral format: UID, Lesion (also handles UID, PatientID, Age, Lesion)
        for _, row in df.iterrows():
            labels[str(row["UID"])] = int(row["Lesion"])
        logger.info(f"Loaded {len(labels)} unilateral labels from {label_csv}")

    elif "Lesion_Left" in columns and "Lesion_Right" in columns:
        # Bilateral format: studyID, Lesion_Left, Lesion_Right
        id_col = "studyID" if "studyID" in columns else df.columns[0]
        for _, row in df.iterrows():
            study_id = str(row[id_col])
            labels[f"{study_id}_left"] = int(row["Lesion_Left"])
            labels[f"{study_id}_right"] = int(row["Lesion_Right"])
        logger.info(f"Loaded {len(labels)} labels (bilateral->unilateral) from {label_csv}")

    else:
        raise ValueError(
            f"Cannot detect label format in {label_csv}. "
            f"Expected columns (UID, Lesion) or (studyID, Lesion_Left, Lesion_Right). "
            f"Found: {columns}"
        )

    # Validate label values
    unique_labels = set(labels.values())
    invalid = unique_labels - {0, 1, 2}
    if invalid:
        logger.warning(f"Unexpected label values found: {invalid}. Expected 0, 1, 2.")

    return labels


def compute_class_weights(labels: List[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for balanced loss.

    Formula: weight_c = N_total / (num_classes * N_c)
    Same as sklearn's compute_class_weight("balanced").

    Args:
        labels: List of integer labels for the training set.
        num_classes: Number of classes.

    Returns:
        Tensor of shape [num_classes] with class weights.
    """
    counts = Counter(labels)
    n_total = len(labels)
    weights = []

    for c in range(num_classes):
        n_c = counts.get(c, 1)  # Avoid division by zero
        weights.append(n_total / (num_classes * n_c))

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class distribution: {dict(counts)}")
    logger.info(f"Class weights: {weights}")

    return weight_tensor

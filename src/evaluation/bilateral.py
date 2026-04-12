"""
Bilateral prediction assembly.

Converts unilateral breast predictions into study-level bilateral JSON files
matching the evaluator's expected format.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import softmax

from ..data.label_mapping import CLASS_NAMES, extract_study_id_and_side

logger = logging.getLogger("breast_mri")

# Uniform probability for missing sides
UNIFORM_PROB = {CLASS_NAMES[0]: 1.0 / 3, CLASS_NAMES[1]: 1.0 / 3, CLASS_NAMES[2]: 1.0 / 3}


def _probs_to_dict(probs: np.ndarray) -> Dict[str, float]:
    """Convert a probability array [3] to a {class_name: probability} dict."""
    return {
        CLASS_NAMES[0]: float(probs[0]),  # normal
        CLASS_NAMES[1]: float(probs[1]),  # benign
        CLASS_NAMES[2]: float(probs[2]),  # malignant
    }


def assemble_bilateral_predictions(
    uids: List[str],
    logits: np.ndarray,
    apply_softmax: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Group unilateral predictions into bilateral study-level predictions.

    Args:
        uids: List of unilateral UIDs (e.g., "ODELIA_BRAID1_0158_1_left").
        logits: Model outputs, shape [N, 3]. Can be logits or probabilities.
        apply_softmax: If True, apply softmax to convert logits to probabilities.

    Returns:
        Dict mapping study_id to bilateral prediction:
        {
            "study_id": {
                "left":  {"normal": 0.1, "benign": 0.2, "malignant": 0.7},
                "right": {"normal": 0.8, "benign": 0.1, "malignant": 0.1}
            }
        }
    """
    if apply_softmax:
        probs = softmax(logits, axis=1)
    else:
        probs = logits

    # Group by study
    study_predictions: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    for uid, prob in zip(uids, probs):
        study_id, side = extract_study_id_and_side(uid)
        study_predictions[study_id][side] = _probs_to_dict(prob)

    # Ensure both sides exist for each study
    missing_sides = 0
    for study_id in study_predictions:
        for side in ["left", "right"]:
            if side not in study_predictions[study_id]:
                study_predictions[study_id][side] = UNIFORM_PROB.copy()
                missing_sides += 1

    if missing_sides > 0:
        logger.warning(
            f"{missing_sides} missing sides filled with uniform probabilities "
            f"across {len(study_predictions)} studies"
        )

    logger.info(f"Assembled bilateral predictions for {len(study_predictions)} studies")
    return dict(study_predictions)


def save_bilateral_json(
    predictions: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
) -> List[str]:
    """
    Save bilateral predictions as JSON files in the evaluator's expected format.

    Creates one JSON file per study:
        output_dir/<study_id>/bilateral-breast-classification-likelihoods.json

    Args:
        predictions: Output from assemble_bilateral_predictions().
        output_dir: Root output directory.

    Returns:
        List of paths to saved JSON files.
    """
    output_path = Path(output_dir)
    saved_paths = []

    for study_id, bilateral_pred in predictions.items():
        study_dir = output_path / study_id
        study_dir.mkdir(parents=True, exist_ok=True)

        json_path = study_dir / "bilateral-breast-classification-likelihoods.json"
        with open(json_path, "w") as f:
            json.dump(bilateral_pred, f, indent=2)

        saved_paths.append(str(json_path))

    logger.info(f"Saved {len(saved_paths)} bilateral JSON files to {output_dir}")
    return saved_paths

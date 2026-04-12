"""
Evaluation metrics for breast MRI classification.

Computes standard classification metrics plus clinically relevant
threshold-based metrics for the malignant-vs-rest binary task.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("breast_mri")

CLASS_NAMES = ["normal", "benign", "malignant"]


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None,
    sensitivity_threshold: float = 0.9,
    specificity_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Compute all classification metrics.

    Args:
        logits: Raw model outputs, shape [N, 3].
        labels: Ground truth labels, shape [N] with values in {0, 1, 2}.
        class_names: Names for the 3 classes.
        sensitivity_threshold: Target sensitivity for specificity@sensitivity.
        specificity_threshold: Target specificity for sensitivity@specificity.

    Returns:
        Dict with keys: accuracy, confusion_matrix, per_class,
        malignant_auroc, specificity_at_sensitivity, sensitivity_at_specificity.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Convert logits to probabilities and predictions
    probs = softmax(logits, axis=1)
    preds = np.argmax(probs, axis=1)

    results = {}

    # 1. Accuracy
    results["accuracy"] = float(accuracy_score(labels, preds))

    # 2. Confusion matrix (rows = true, columns = predicted)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    results["confusion_matrix"] = cm.tolist()

    # 3. Per-class precision, recall, F1
    report = classification_report(
        labels, preds,
        labels=[0, 1, 2],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    results["per_class"] = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in class_names
    }
    results["macro_f1"] = report["macro avg"]["f1-score"]

    # 4. Malignant-vs-rest AUROC
    binary_labels = (labels == 2).astype(int)  # 1 = malignant, 0 = rest
    malignant_probs = probs[:, 2]              # P(malignant)

    if len(np.unique(binary_labels)) > 1:
        results["malignant_auroc"] = float(roc_auc_score(binary_labels, malignant_probs))

        # ROC curve for threshold-based metrics
        fpr, tpr, thresholds = roc_curve(binary_labels, malignant_probs)

        # 5. Specificity at target sensitivity
        results["specificity_at_sensitivity"] = _specificity_at_sensitivity(
            fpr, tpr, sensitivity_threshold
        )

        # 6. Sensitivity at target specificity
        results["sensitivity_at_specificity"] = _sensitivity_at_specificity(
            fpr, tpr, specificity_threshold
        )
    else:
        logger.warning("Only one class present in labels — skipping AUROC and threshold metrics")
        results["malignant_auroc"] = None
        results["specificity_at_sensitivity"] = None
        results["sensitivity_at_specificity"] = None

    return results


def _specificity_at_sensitivity(
    fpr: np.ndarray, tpr: np.ndarray, target_sensitivity: float
) -> Optional[float]:
    """
    Find the maximum specificity where sensitivity >= target.

    On the ROC curve: sensitivity = TPR, specificity = 1 - FPR.
    """
    # Filter to operating points where sensitivity >= target
    mask = tpr >= target_sensitivity
    if not mask.any():
        return None
    # Among those, find the one with minimum FPR (= maximum specificity)
    min_fpr = fpr[mask].min()
    return float(1.0 - min_fpr)


def _sensitivity_at_specificity(
    fpr: np.ndarray, tpr: np.ndarray, target_specificity: float
) -> Optional[float]:
    """
    Find the maximum sensitivity where specificity >= target.

    Specificity = 1 - FPR >= target  →  FPR <= 1 - target.
    """
    max_fpr = 1.0 - target_specificity
    mask = fpr <= max_fpr
    if not mask.any():
        return None
    # Among those, find the maximum TPR (= maximum sensitivity)
    return float(tpr[mask].max())


def compute_metrics_per_institution(
    logits: np.ndarray,
    labels: np.ndarray,
    institutions: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics broken down by institution/site.

    Args:
        logits: Shape [N, 3].
        labels: Shape [N].
        institutions: List of institution names, length N.

    Returns:
        Dict mapping institution name to metrics dict.
    """
    unique_institutions = sorted(set(institutions))
    results = {}

    for inst in unique_institutions:
        mask = np.array([inst_name == inst for inst_name in institutions])
        if mask.sum() < 2:
            continue
        inst_metrics = compute_metrics(logits[mask], labels[mask])
        inst_metrics["n_samples"] = int(mask.sum())
        results[inst] = inst_metrics

    return results


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Pretty-print the metrics to console."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")

    if metrics.get("malignant_auroc") is not None:
        print(f"Malignant-vs-rest AUROC: {metrics['malignant_auroc']:.4f}")

    if metrics.get("specificity_at_sensitivity") is not None:
        print(f"Specificity @ 90% sensitivity: {metrics['specificity_at_sensitivity']:.4f}")

    if metrics.get("sensitivity_at_specificity") is not None:
        print(f"Sensitivity @ 90% specificity: {metrics['sensitivity_at_specificity']:.4f}")

    print(f"\nMacro F1: {metrics['macro_f1']:.4f}")

    print("\nPer-class metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 52)
    for cls_name, cls_metrics in metrics["per_class"].items():
        print(
            f"  {cls_name:<12} "
            f"{cls_metrics['precision']:>10.4f} "
            f"{cls_metrics['recall']:>10.4f} "
            f"{cls_metrics['f1']:>10.4f} "
            f"{cls_metrics['support']:>10}"
        )

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    cm = metrics["confusion_matrix"]
    header = "  " + " " * 12 + "".join(f"{'pred_' + n:>12}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:<12}" + "".join(f"{v:>12}" for v in row))

    print("=" * 60 + "\n")

#!/usr/bin/env python3
"""
Ensemble multiple prediction CSVs by averaging probabilities.

Usage:
    python scripts/ensemble_submissions.py \
        --inputs predictions_v4.csv predictions_v5.csv \
        --output predictions_ensemble.csv

    With custom weights:
    python scripts/ensemble_submissions.py \
        --inputs predictions_v4.csv predictions_v5.csv \
        --weights 0.4 0.6 \
        --output predictions_ensemble.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_predictions(csv_path: str) -> dict:
    """Load predictions CSV into {ID: [normal, benign, malignant]} dict."""
    preds = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds[row["ID"]] = [
                float(row["normal"]),
                float(row["benign"]),
                float(row["malignant"]),
            ]
    return preds


def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction CSVs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Prediction CSV files")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="Weights for each model (default: equal)")
    parser.add_argument("--output", type=str, default="predictions_ensemble.csv")
    args = parser.parse_args()

    n_models = len(args.inputs)

    # Set weights
    if args.weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        assert len(args.weights) == n_models, "Number of weights must match number of inputs"
        total = sum(args.weights)
        weights = [w / total for w in args.weights]

    print(f"Ensembling {n_models} models with weights: {weights}")

    # Load all predictions
    all_preds = [load_predictions(p) for p in args.inputs]

    # Find common IDs
    common_ids = set(all_preds[0].keys())
    for preds in all_preds[1:]:
        common_ids &= set(preds.keys())
    common_ids = sorted(common_ids)
    print(f"Found {len(common_ids)} common IDs across all models")

    # Weighted average
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "normal", "benign", "malignant"])

        for uid in common_ids:
            avg_probs = np.zeros(3)
            for preds, weight in zip(all_preds, weights):
                avg_probs += weight * np.array(preds[uid])

            # Re-normalize to ensure sum = 1
            avg_probs = avg_probs / avg_probs.sum()

            writer.writerow([uid, f"{avg_probs[0]:.4f}", f"{avg_probs[1]:.4f}", f"{avg_probs[2]:.4f}"])

    print(f"Saved ensemble predictions to {output_path}")
    print(f"Preview:")
    with open(output_path) as f:
        for i, line in enumerate(f):
            print(f"  {line.strip()}")
            if i >= 5:
                break


if __name__ == "__main__":
    main()

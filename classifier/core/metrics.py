"""Per-class evaluation metrics shared across all experiment conditions."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from classifier.core.config import cfg


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: List[str] = cfg.viable_classes,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class AUC-ROC, F1, sensitivity and specificity.

    Args:
        targets:      Ground-truth binary array, shape [N, C].
        probs:        Sigmoid probabilities, shape [N, C].
        class_names:  Names for the C classes.
        threshold:    Decision threshold for F1 / sensitivity / specificity.

    Returns:
        Nested dict ``{class_name: {auroc, f1, sensitivity, specificity}}``.
    """
    preds = (probs >= threshold).astype(int)
    results: Dict[str, Dict[str, float]] = {}

    for i, name in enumerate(class_names):
        y_true = targets[:, i]
        y_prob = probs[:, i]
        y_pred = preds[:, i]

        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auroc = float("nan")

        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[name] = {
            "auroc": auroc,
            "f1": f1,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }

    return results


def mean_auroc(metrics: Dict[str, Dict[str, float]]) -> float:
    """Return the mean AUC-ROC across all classes (ignoring NaNs)."""
    scores = [v["auroc"] for v in metrics.values() if not np.isnan(v["auroc"])]
    return float(np.mean(scores)) if scores else float("nan")


def format_metrics_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """Return a readable text table of per-class metrics."""
    header = f"{'Class':<25} {'AUC-ROC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8}"
    sep = "-" * len(header)
    rows = [header, sep]
    for name, m in metrics.items():
        rows.append(
            f"{name:<25} {m['auroc']:>8.4f} {m['f1']:>8.4f}"
            f" {m['sensitivity']:>8.4f} {m['specificity']:>8.4f}"
        )
    rows.append(sep)
    rows.append(f"{'Mean AUC-ROC':<25} {mean_auroc(metrics):>8.4f}")
    return "\n".join(rows)

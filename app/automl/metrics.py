"""Evaluation metrics, computed on plain NumPy arrays.

Nothing in this module imports scikit-learn. Every number the leaderboard
shows is produced by the functions below, which is deliberate: you can read
the metric and check the arithmetic by hand.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of rows where the predicted class matches the truth."""
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true.astype(int) == y_pred.astype(int)))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    """Rows are truth, columns are prediction."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> list[dict[str, float | int]]:
    """Precision, recall and F1 for each class, plus its support."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    out: list[dict[str, float | int]] = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else 2 * tp / denom
        out.append(
            {
                "class_index": c,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(cm[c, :].sum()),
            }
        )
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1.

    Macro rather than micro on purpose: a model that ignores a rare class
    should not be able to hide behind the majority class.
    """
    scores = per_class_f1(y_true, y_pred, n_classes)
    if not scores:
        return 0.0
    return float(np.mean([float(s["f1"]) for s in scores]))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination.

    Returns 0.0 for a constant target rather than dividing by zero, and
    allows negative values — a model worse than predicting the mean should
    show that, not be clipped to zero.
    """
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def score_predictions(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """Primary metric first, then the supporting numbers."""
    if task == "classification":
        return {
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "macro_f1": round(macro_f1(y_true, y_pred, n_classes), 4),
        }
    return {
        "r2": round(r2(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
    }


def primary_metric_name(task: str) -> str:
    return "accuracy" if task == "classification" else "r2"


def higher_is_better(task: str) -> bool:
    """Both primary metrics (accuracy, R^2) are maximised."""
    return True

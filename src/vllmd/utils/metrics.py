"""
Evaluation metrics for the video action recognition (predictions) model.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _to_indices(
    y: Union[List[str], List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Convert labels to integer indices. If already ints, pass through; else map via class_names."""
    arr = np.asarray(y)
    if arr.dtype.kind in ("i", "u") or (arr.dtype == np.int64 or arr.dtype == np.int32):
        return np.asarray(arr, dtype=np.int64)
    if class_names is None:
        raise ValueError("class_names is required when labels are strings")
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    return np.array([name_to_idx[str(v)] for v in arr], dtype=np.int64)


def classification_metrics(
    y_true: Union[List[str], List[int], np.ndarray],
    y_pred: Union[List[str], List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    average: str = "weighted",
) -> dict:
    """
    Compute classification metrics for predictions vs ground truth.

    Args:
        y_true: Ground truth labels (class names or integer indices).
        y_pred: Predicted labels (class names or integer indices).
        class_names: List of class names; required if labels are strings.
        average: Averaging for precision/recall/F1: "macro", "micro", or "weighted".

    Returns:
        Dict with keys: accuracy, precision, recall, f1, and (optionally) per_class_*.
    """
    y_true_idx = _to_indices(y_true, class_names)
    y_pred_idx = _to_indices(y_pred, class_names)

    acc = accuracy_score(y_true_idx, y_pred_idx)
    precision = precision_score(
        y_true_idx, y_pred_idx, average=average, zero_division=0  # type: ignore[arg-type]
    )
    recall = recall_score(
        y_true_idx, y_pred_idx, average=average, zero_division=0  # type: ignore[arg-type]
    )
    f1 = f1_score(
        y_true_idx, y_pred_idx, average=average, zero_division=0  # type: ignore[arg-type]
    )

    result: dict[str, Union[float, dict[str, float]]] = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if class_names is not None:
        prec_per: np.ndarray = precision_score(
            y_true_idx, y_pred_idx, average=None, zero_division=0  # type: ignore[arg-type]
        )
        rec_per: np.ndarray = recall_score(
            y_true_idx, y_pred_idx, average=None, zero_division=0  # type: ignore[arg-type]
        )
        f1_per: np.ndarray = f1_score(
            y_true_idx, y_pred_idx, average=None, zero_division=0  # type: ignore[arg-type]
        )
        n = len(class_names)
        result["per_class_precision"] = {
            class_names[i]: float(prec_per[i]) for i in range(n) if i < len(prec_per)
        }
        result["per_class_recall"] = {
            class_names[i]: float(rec_per[i]) for i in range(n) if i < len(rec_per)
        }
        result["per_class_f1"] = {
            class_names[i]: float(f1_per[i]) for i in range(n) if i < len(f1_per)
        }

    return result


def top_k_accuracy(
    y_true: Union[List[int], np.ndarray],
    probs: np.ndarray,
    k: int = 5,
) -> float:
    """
    Top-k accuracy: true label is among the k highest-probability classes.

    Args:
        y_true: Ground truth class indices, shape (n_samples,).
        probs: Predicted probabilities, shape (n_samples, n_classes).

    Returns:
        Fraction of samples where true class is in top-k.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs and y_true must have same number of samples")
    top_k_pred = np.argsort(probs, axis=1)[:, -k:]
    correct = (top_k_pred == y_true.reshape(-1, 1)).any(axis=1)
    return float(np.mean(correct))


def get_confusion_matrix(
    y_true: Union[List[str], List[int], np.ndarray],
    y_pred: Union[List[str], List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Confusion matrix (rows = true, columns = predicted).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Required if labels are strings.

    Returns:
        Matrix of shape (n_classes, n_classes).
    """
    y_true_idx = _to_indices(y_true, class_names)
    y_pred_idx = _to_indices(y_pred, class_names)
    return confusion_matrix(y_true_idx, y_pred_idx)


def evaluate_predictions(
    y_true: Union[List[str], List[int], np.ndarray],
    y_pred: Union[List[str], List[int], np.ndarray],
    probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = "weighted",
    top_k: Optional[int] = None,
) -> dict:
    """
    Full evaluation: classification metrics, optional confusion matrix and top-k accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        probs: Predicted class probabilities (n_samples, n_classes); used if top_k is set.
        class_names: Class names for string labels and per-class metrics.
        average: Averaging for precision/recall/F1.
        top_k: If set, compute top-k accuracy (requires probs).

    Returns:
        Dict with classification_metrics plus optional confusion_matrix and top_k_accuracy.
    """
    out = classification_metrics(
        y_true, y_pred, class_names=class_names, average=average
    )
    out["confusion_matrix"] = get_confusion_matrix(
        y_true, y_pred, class_names=class_names
    ).tolist()

    if top_k is not None and probs is not None:
        y_true_idx = _to_indices(y_true, class_names)
        out["top_k_accuracy"] = top_k_accuracy(y_true_idx, probs, k=top_k)

    return out

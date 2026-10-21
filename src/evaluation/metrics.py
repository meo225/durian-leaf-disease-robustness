"""Chỉ số phân loại: macro f1, balanced accuracy, recall từng lớp và confusion matrix."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Tính bộ chỉ số chính trên nhãn số nguyên, giữ đủ mọi lớp kể cả lớp không được dự đoán."""
    labels = list(range(len(classes)))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_recall": {name: float(value) for name, value in zip(classes, recalls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

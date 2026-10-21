"""Metric phân loại trên một lượt đánh giá."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def classification_scores(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Tính macro F1, balanced accuracy, recall từng lớp và confusion matrix.

    Nhãn là chỉ số lớp theo đúng thứ tự ``class_names``.
    """
    labels = list(range(len(class_names)))
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true và y_pred phải cùng số phần tử.")
    if y_true_arr.size == 0:
        raise ValueError("Không có mẫu để tính metric.")

    per_class = recall_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    return {
        "macro_f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "per_class_recall": {
            str(name): float(value) for name, value in zip(class_names, per_class)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }

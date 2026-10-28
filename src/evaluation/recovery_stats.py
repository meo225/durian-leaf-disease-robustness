"""Thống kê ghép cặp cho bản đồ phục hồi (task 4.5): bootstrap theo ảnh, McNemar chính xác, hiệu chỉnh Holm.

Mọi so sánh là trên cùng một tập ảnh và nhãn: ảnh suy giảm không xử lý và ảnh suy giảm đã xử lý. Bootstrap lấy lại mẫu theo ảnh một lần
(cùng ma trận lấy lại mẫu cho mọi ô), nên khoảng tin cậy của hiệu hai macro F1 tính được từ các điểm bootstrap của từng bên.
Khoảng tin cậy chỉ bắt được dao động do tập ảnh test, không bắt được dao động do huấn luyện (một seed).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest


class PairedBootstrap:
    """Lấy lại mẫu theo ảnh `replicates` lần. Tính macro F1 của từng lần lấy lại bằng phép nhân ma trận, nên rất nhanh."""

    def __init__(self, labels: Sequence[int], num_classes: int, replicates: int = 1000, seed: int = 42) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_classes = int(num_classes)
        count = len(self.labels)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, count, size=(int(replicates), count))
        weights = np.zeros((int(replicates), count), dtype=np.float32)
        np.add.at(weights, (np.arange(int(replicates))[:, None], draws), 1.0)
        self.weights = weights
        self._truth = np.eye(self.num_classes, dtype=np.float32)[self.labels]
        self._actual = weights @ self._truth

    def _one_hot(self, predictions: np.ndarray) -> np.ndarray:
        return np.eye(self.num_classes, dtype=np.float32)[np.asarray(predictions, dtype=np.int64)]

    def macro_f1(self, predictions: np.ndarray) -> float:
        """Macro F1 trên toàn bộ ảnh. Lớp không có ảnh hoặc không được dự đoán tính 0, như sklearn với zero_division=0."""
        predicted = self._one_hot(predictions)
        true_positive = (self._truth * predicted).sum(axis=0)
        denominator = predicted.sum(axis=0) + self._truth.sum(axis=0)
        return float(np.where(denominator > 0, 2 * true_positive / np.maximum(denominator, 1e-12), 0.0).mean())

    def macro_f1_samples(self, predictions: np.ndarray) -> np.ndarray:
        """Macro F1 của từng lần lấy lại mẫu, vector độ dài `replicates`."""
        predicted = self._one_hot(predictions)
        true_positive = self.weights @ (self._truth * predicted)
        denominator = self.weights @ predicted + self._actual
        return np.where(denominator > 0, 2 * true_positive / np.maximum(denominator, 1e-12), 0.0).mean(axis=1)

    @staticmethod
    def interval(samples: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
        """Khoảng phân vị của các điểm bootstrap. Trả NaN nếu không có điểm hợp lệ."""
        values = samples[~np.isnan(samples)]
        if values.size == 0:
            return float("nan"), float("nan")
        tail = (1.0 - confidence) / 2.0 * 100.0
        return float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))


def mcnemar_exact(base_correct: np.ndarray, other_correct: np.ndarray) -> dict[str, float]:
    """McNemar chính xác (nhị thức hai phía) trên cặp ảnh. gains: sai ở bên gốc, đúng ở bên kia. harms: ngược lại."""
    gains = int((~base_correct & other_correct).sum())
    harms = int((base_correct & ~other_correct).sum())
    pvalue = float(binomtest(gains, gains + harms, 0.5).pvalue) if gains + harms > 0 else 1.0
    return {"gains": gains, "harms": harms, "p": pvalue}


def holm(pvalues: Mapping[str, float] | Sequence[float]) -> dict[str, float] | list[float]:
    """Hiệu chỉnh Holm cho một nhóm p-value. Trả cùng dạng với đầu vào (dict theo khóa hoặc list)."""
    keys = list(pvalues.keys()) if isinstance(pvalues, Mapping) else list(range(len(pvalues)))
    values = [pvalues[key] for key in keys]
    order = np.argsort(values)
    total = len(values)
    adjusted = [0.0] * total
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (total - rank)))
        adjusted[index] = running
    return dict(zip(keys, adjusted)) if isinstance(pvalues, Mapping) else adjusted


def compare(
    bootstrap: PairedBootstrap,
    base_predictions: np.ndarray,
    other_predictions: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """So hai bộ dự đoán trên cùng ảnh: macro F1 hai bên, hiệu (other trừ base), khoảng tin cậy của hiệu, McNemar."""
    labels = bootstrap.labels
    base_f1 = bootstrap.macro_f1(base_predictions)
    other_f1 = bootstrap.macro_f1(other_predictions)
    difference = bootstrap.macro_f1_samples(other_predictions) - bootstrap.macro_f1_samples(base_predictions)
    low, high = bootstrap.interval(difference, confidence)
    test = mcnemar_exact(np.asarray(base_predictions) == labels, np.asarray(other_predictions) == labels)
    return {"base_f1": base_f1, "other_f1": other_f1, "delta": other_f1 - base_f1, "delta_low": low, "delta_high": high, **test}


def normalized_recovery(
    bootstrap: PairedBootstrap,
    clean_predictions: np.ndarray,
    corrupted_predictions: np.ndarray,
    processed_predictions: np.ndarray,
    min_degradation: float = 0.02,
    interval_min_degradation: float = 0.05,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Degradation, recovery và normalized recovery kèm khoảng tin cậy.

    Normalized recovery chỉ tính khi degradation lớn hơn `min_degradation`. Khoảng tin cậy của nó chỉ báo khi degradation từ
    `interval_min_degradation` trở lên (tỉ số của số nhỏ không ổn định), và bỏ các lần lấy lại mẫu có degradation dưới 0,01.
    """
    clean = bootstrap.macro_f1(clean_predictions)
    corrupted = bootstrap.macro_f1(corrupted_predictions)
    processed = bootstrap.macro_f1(processed_predictions)
    degradation = clean - corrupted
    recovery = processed - corrupted
    clean_s = bootstrap.macro_f1_samples(clean_predictions)
    corrupted_s = bootstrap.macro_f1_samples(corrupted_predictions)
    processed_s = bootstrap.macro_f1_samples(processed_predictions)
    degradation_s, recovery_s = clean_s - corrupted_s, processed_s - corrupted_s
    out = {
        "degradation": degradation,
        "degradation_low": bootstrap.interval(degradation_s, confidence)[0],
        "degradation_high": bootstrap.interval(degradation_s, confidence)[1],
        "recovery": recovery,
        "recovery_low": bootstrap.interval(recovery_s, confidence)[0],
        "recovery_high": bootstrap.interval(recovery_s, confidence)[1],
        "normalized_recovery": recovery / degradation if degradation > min_degradation else float("nan"),
        "normalized_low": float("nan"),
        "normalized_high": float("nan"),
    }
    if degradation >= interval_min_degradation:
        ratio = np.where(degradation_s > 0.01, recovery_s / np.where(degradation_s > 0.01, degradation_s, 1.0), np.nan)
        out["normalized_low"], out["normalized_high"] = bootstrap.interval(ratio, confidence)
    return out

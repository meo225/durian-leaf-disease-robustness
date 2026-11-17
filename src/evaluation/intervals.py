"""Khoảng tin cậy cho các số đo trên ít ảnh hoặc ít họ (task 5.7, 5.8, 5.11).

- `wilson_interval`: khoảng Wilson cho một tỉ lệ, dùng cho recall từng lớp. Lớp ALLOCARIDARA_ATTACK trên validation chỉ có 46 ảnh,
  nên recall 0.25 có khoảng 95% rộng từ 0.16 đến 0.40. Khoảng Wilson không rơi ra ngoài [0, 1] và dùng được khi k = 0 hoặc k = n.
- `bootstrap_ci`: khoảng bootstrap phân vị cho một thống kê bất kỳ của một mẫu.
- `spearman_with_ci`: hệ số Spearman cùng khoảng bootstrap theo cặp. Với 14 đến 20 họ, khoảng này rất rộng, và đó là điều cần báo.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import stats

Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Khoảng Wilson cho tỉ lệ `successes / total`. Với total = 0 trả về (0, 1): không có dữ liệu thì không loại trừ giá trị nào."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Cần 0 <= successes <= total.")
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def per_class_recall_wilson(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    classes: Sequence[str],
    z: float = Z_95,
) -> dict[str, dict[str, float]]:
    """Recall từng lớp kèm số ảnh thật và khoảng Wilson. Lớp không có ảnh nào có recall NaN và khoảng (0, 1)."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape != predictions.shape:
        raise ValueError("labels và predictions phải cùng kích thước.")
    table: dict[str, dict[str, float]] = {}
    for index, name in enumerate(classes):
        support = int((labels == index).sum())
        correct = int(((labels == index) & (predictions == index)).sum())
        low, high = wilson_interval(correct, support, z)
        table[name] = {"support": support, "recall": correct / support if support else float("nan"), "low": low, "high": high}
    return table


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Khoảng bootstrap phân vị của `statistic` trên mẫu `values`."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("Mẫu rỗng.")
    rng = np.random.default_rng(seed)
    draws = np.array([statistic(data[rng.integers(0, data.size, data.size)]) for _ in range(n_resamples)])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def spearman_with_ci(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Mapping[str, float]:
    """Spearman giữa hai dãy cùng độ dài, khoảng bootstrap theo cặp. Cặp lặp làm mẫu bootstrap suy biến bị bỏ qua (hệ số NaN)."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("x và y phải là hai dãy một chiều cùng độ dài.")
    if a.size < 3:
        raise ValueError("Cần ít nhất 3 cặp.")
    rho = float(stats.spearmanr(a, b).statistic)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        index = rng.integers(0, a.size, a.size)
        if np.unique(a[index]).size < 2 or np.unique(b[index]).size < 2:
            continue
        draws.append(float(stats.spearmanr(a[index], b[index]).statistic))
    alpha = (1.0 - confidence) / 2.0
    low, high = (float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))) if draws else (float("nan"), float("nan"))
    return {"rho": rho, "low": low, "high": high, "n": float(a.size)}

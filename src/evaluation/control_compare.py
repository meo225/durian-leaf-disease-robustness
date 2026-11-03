"""So sánh các cách huấn luyện trên cùng ảnh validation và cùng điều kiện suy giảm (task 5.6).

Mỗi cách huấn luyện là một ma trận dự đoán (1 + C) × N: hàng đầu là ảnh sạch, C hàng sau là các điều kiện suy giảm theo thứ tự cố định.
Mọi so sánh ghép cặp theo ảnh: cùng chỉ số bootstrap cho mọi cách, McNemar trên cùng ảnh. Chỉ đọc kết quả đã lưu, không mở tập test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.recovery_stats import holm, mcnemar_exact
from src.selector.calibrate import macro_f1_rows


def split_name(condition: str) -> tuple[str, int]:
    family, severity = condition.rsplit("_s", 1)
    return family, int(severity)


def reference_matrix(path: str | Path, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Nhãn và ma trận dự đoán của reference model trên validation, từ xác suất đã lưu ở task 4.4."""
    data = np.load(path, allow_pickle=False)
    index = {str(name): position for position, name in enumerate(data["conditions"])}
    rows = ["clean", *names]
    missing = [name for name in rows if name not in index]
    if missing:
        raise KeyError(f"Thiếu điều kiện trong {path}: {missing[:3]}")
    probabilities = data["probs_val"]
    return data["labels_val"].astype(np.int64), np.stack([probabilities[index[name]].argmax(axis=1) for name in rows]).astype(np.int64)


def retrained_matrix(path: str | Path, config_id: str, names: Sequence[str], technique: str = "retrained") -> tuple[np.ndarray, np.ndarray]:
    """Nhãn và ma trận dự đoán của một model train lại, từ file .npz của evaluate_with_predictions."""
    data = np.load(path, allow_pickle=False)
    rows = ["clean", *names]
    keys = [f"{technique}__{config_id}__{name}" for name in rows]
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Thiếu khóa trong {path}: {missing[:3]}")
    return data["labels"].astype(np.int64), np.stack([data[key] for key in keys]).astype(np.int64)


def check_same_images(labels: Mapping[str, np.ndarray]) -> np.ndarray:
    """Các cách phải được đo trên cùng ảnh theo cùng thứ tự. Trả về nhãn chung."""
    first_name, first = next(iter(labels.items()))
    for name, other in labels.items():
        if not np.array_equal(first, other):
            raise ValueError(f"Nhãn của {name} khác {first_name}: hai bên không đo trên cùng ảnh.")
    return first


def bootstrap_indices(images: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, images, size=(resamples, images))


def objective_samples(labels: np.ndarray, matrix: np.ndarray, classes: int, indices: np.ndarray) -> np.ndarray:
    """Macro F1 trung bình các điều kiện suy giảm (bỏ hàng ảnh sạch) trên từng mẫu bootstrap."""
    values = np.empty(len(indices))
    for row, chosen in enumerate(indices):
        values[row] = macro_f1_rows(labels[chosen], matrix[1:, chosen], classes).mean()
    return values


def _interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(low), float(high)


def summarize_methods(
    labels: np.ndarray,
    matrices: Mapping[str, np.ndarray],
    names: Sequence[str],
    classes: int,
    base: str,
    resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Một dòng mỗi cách: macro F1 ảnh sạch, trung bình các điều kiện, trung bình theo mức, hiệu so với `base` kèm khoảng tin cậy ghép cặp 95%."""
    indices = bootstrap_indices(len(labels), resamples, seed)
    samples = {name: objective_samples(labels, matrix, classes, indices) for name, matrix in matrices.items()}
    levels = np.array([split_name(name)[1] for name in names])
    rows = []
    for method, matrix in matrices.items():
        f1 = macro_f1_rows(labels, matrix, classes)
        delta = samples[method] - samples[base]
        low, high = _interval(delta)
        own_low, own_high = _interval(samples[method])
        row = {
            "method": method,
            "clean_f1": float(f1[0]),
            "objective": float(f1[1:].mean()),
            "objective_low": own_low,
            "objective_high": own_high,
            "delta_vs_base": float(f1[1:].mean() - macro_f1_rows(labels, matrices[base], classes)[1:].mean()),
            "delta_low": low,
            "delta_high": high,
        }
        for level in sorted(set(levels.tolist())):
            row[f"level_{level}"] = float(f1[1:][levels == level].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def family_scores(labels: np.ndarray, matrices: Mapping[str, np.ndarray], names: Sequence[str], classes: int, level: int | None = None) -> pd.DataFrame:
    """Macro F1 theo họ (hàng) và cách (cột), trung bình ba mức hoặc chỉ một mức nếu cho `level`."""
    families = list(dict.fromkeys(split_name(name)[0] for name in names))
    table = {}
    for method, matrix in matrices.items():
        f1 = macro_f1_rows(labels, matrix, classes)[1:]
        table[method] = {
            family: float(np.mean([value for name, value in zip(names, f1) if split_name(name)[0] == family and (level is None or split_name(name)[1] == level)]))
            for family in families
        }
    return pd.DataFrame(table)


def condition_comparison(labels: np.ndarray, base: np.ndarray, other: np.ndarray, names: Sequence[str], classes: int) -> pd.DataFrame:
    """Từng điều kiện: macro F1 hai bên, hiệu (other trừ base), số ảnh đổi từ sai sang đúng và ngược lại, McNemar chính xác và p hiệu chỉnh Holm trên mọi điều kiện."""
    base_f1 = macro_f1_rows(labels, base[1:], classes)
    other_f1 = macro_f1_rows(labels, other[1:], classes)
    rows = []
    for position, name in enumerate(names, start=1):
        test = mcnemar_exact(base[position] == labels, other[position] == labels)
        family, severity = split_name(name)
        rows.append(
            {
                "condition": name, "family": family, "severity": severity,
                "base_f1": float(base_f1[position - 1]), "other_f1": float(other_f1[position - 1]),
                "delta": float(other_f1[position - 1] - base_f1[position - 1]),
                "gains": test["gains"], "harms": test["harms"], "p": test["p"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame["p"].tolist())
    return frame


def verdict(delta: float, low: float, high: float, margin: float = 0.0) -> str:
    """Kết luận theo khoảng tin cậy của hiệu: tốt hơn, kém hơn hoặc chưa phân biệt được. `margin` là ngưỡng bỏ qua nếu cần."""
    if low > margin:
        return "tốt hơn"
    if high < -margin:
        return "kém hơn"
    return "chưa phân biệt được"

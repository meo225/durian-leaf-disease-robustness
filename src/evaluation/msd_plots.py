"""Biểu đồ cho MSD (task 5.7): thanh MSD từng họ cùng ngưỡng gần/xa, và ma trận khoảng cách giữa các họ."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"near": "#2a9d8f", "far": "#e76f51", "ambiguous": "#999999"}


def plot_msd_bars(rows: Sequence[Mapping[str, Any]], threshold: float, path: str | Path, title: str) -> Path:
    """Thanh ngang cho mỗi họ: MSD tới augmentation suy giảm, vạch lỗi là khoảng bootstrap 95%, màu theo nhóm, đường đứt là ngưỡng leave-one-out.

    `rows` gồm family, msd, ci_low, ci_high, group.
    """
    ordered = sorted(rows, key=lambda row: row["msd"])
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(8, 0.32 * len(ordered) + 1.5))
    values = np.array([row["msd"] for row in ordered])
    low = np.array([row["ci_low"] for row in ordered])
    high = np.array([row["ci_high"] for row in ordered])
    axis.barh(positions, values, color=[COLORS.get(row["group"], "#555555") for row in ordered])
    axis.errorbar(values, positions, xerr=[np.clip(values - low, 0, None), np.clip(high - values, 0, None)], fmt="none", ecolor="black", capsize=2, linewidth=0.8)
    axis.axvline(threshold, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(positions, [row["family"] for row in ordered])
    axis.set_xlabel("MSD tới augmentation suy giảm (đường đứt là ngưỡng gần/xa)")
    axis.set_title(title)
    figure.tight_layout()
    target = Path(path)
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def plot_distance_matrix(matrix: Mapping[str, Mapping[str, float]], path: str | Path, title: str) -> Path:
    """Ma trận khoảng cách giữa đặc trưng của từng cặp họ, các họ xếp theo thứ tự khai báo của `matrix`."""
    names = list(matrix)
    data = np.array([[matrix[a][b] for b in names] for a in names])
    figure, axis = plt.subplots(figsize=(0.45 * len(names) + 3, 0.45 * len(names) + 2.5))
    image = axis.imshow(data, cmap="viridis")
    axis.set_xticks(range(len(names)), names, rotation=90)
    axis.set_yticks(range(len(names)), names)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    target = Path(path)
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target

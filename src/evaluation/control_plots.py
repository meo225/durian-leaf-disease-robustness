"""Hình cho đối chứng augmentation tổng quát (task 5.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#9aa5b1", "#d98c1f", "#1f77b4", "#2a9d8f", "#8e6bbf"]


def _colors(methods: Sequence[str]) -> dict[str, str]:
    return {method: PALETTE[index % len(PALETTE)] for index, method in enumerate(methods)}


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)


def plot_objective(summary: pd.DataFrame, path: str | Path, title: str) -> None:
    """Macro F1 trung bình các điều kiện của từng cách, kèm khoảng tin cậy bootstrap 95%, cùng điểm ảnh sạch."""
    colors = _colors(list(summary["method"]))
    figure, axis = plt.subplots(figsize=(8, 0.9 * len(summary) + 1.8))
    positions = np.arange(len(summary))[::-1]
    for position, (_, row) in zip(positions, summary.iterrows()):
        axis.barh(position, row["objective"], color=colors[row["method"]], height=0.55)
        axis.errorbar(row["objective"], position, xerr=[[row["objective"] - row["objective_low"]], [row["objective_high"] - row["objective"]]],
                      fmt="none", ecolor="#222222", elinewidth=1, capsize=3)
        axis.plot(row["clean_f1"], position, marker="D", color="#222222", markersize=5, linestyle="none")
        axis.text(row["objective_high"] + 0.003, position, f"{row['objective']:.4f}", va="center", fontsize=9)
    axis.set_yticks(positions)
    axis.set_yticklabels(summary["method"])
    axis.set_xlim(max(0.0, summary["objective_low"].min() - 0.05), 1.02)
    axis.set_xlabel("Macro F1 trung bình các điều kiện (cột, kèm CI 95%) và ảnh sạch (hình thoi)")
    axis.set_title(title, fontsize=11)
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_levels(summary: pd.DataFrame, path: str | Path, title: str) -> None:
    """Macro F1 trung bình theo mức suy giảm, ảnh sạch ở mức 0."""
    colors = _colors(list(summary["method"]))
    level_columns = [column for column in summary.columns if column.startswith("level_")]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for _, row in summary.iterrows():
        values = [row["clean_f1"], *[row[column] for column in level_columns]]
        axis.plot(range(len(values)), values, marker="o", color=colors[row["method"]], label=row["method"])
    axis.set_xticks(range(len(level_columns) + 1))
    axis.set_xticklabels(["Sạch", *[f"Mức {column.split('_')[1]}" for column in level_columns]])
    axis.set_ylabel("Macro F1")
    axis.set_title(title, fontsize=11)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_family_bars(scores: pd.DataFrame, path: str | Path, title: str, ylabel: str) -> None:
    """Macro F1 theo họ cho từng cách, nhóm cột theo họ."""
    colors = _colors(list(scores.columns))
    width = 0.8 / scores.shape[1]
    figure, axis = plt.subplots(figsize=(max(8, 0.9 * len(scores) + 3), 4.8))
    for offset, method in enumerate(scores.columns):
        axis.bar(np.arange(len(scores)) + offset * width, scores[method], width=width, color=colors[method], label=method)
    axis.set_xticks(np.arange(len(scores)) + width * (scores.shape[1] - 1) / 2)
    axis.set_xticklabels(scores.index, rotation=40, ha="right")
    axis.set_ylim(max(0.0, float(scores.to_numpy().min()) - 0.08), 1.0)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.legend(frameon=False, fontsize=8, ncol=scores.shape[1])
    axis.grid(axis="y", alpha=0.25)
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_delta_heatmap(comparison: pd.DataFrame, path: str | Path, title: str) -> None:
    """Hiệu macro F1 (cách so sánh trừ cách gốc) theo họ và mức. Ô có dấu sao có p Holm dưới 0,05."""
    grid = comparison.pivot(index="family", columns="severity", values="delta").loc[list(dict.fromkeys(comparison["family"]))]
    significant = comparison.assign(flag=comparison["p_holm"] < 0.05).pivot(index="family", columns="severity", values="flag").loc[grid.index]
    limit = max(0.02, float(np.abs(grid.to_numpy()).max()))
    figure, axis = plt.subplots(figsize=(0.9 * grid.shape[1] + 4.5, 0.4 * len(grid) + 2.0))
    image = axis.imshow(grid.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(grid.shape[1]))
    axis.set_xticklabels([f"Mức {level}" for level in grid.columns])
    axis.set_yticks(range(len(grid)))
    axis.set_yticklabels(grid.index)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid.iat[row, column]
            mark = "*" if bool(significant.iat[row, column]) else ""
            axis.text(column, row, f"{value:+.3f}{mark}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.6 * limit else "#222222")
    figure.colorbar(image, ax=axis, label="Hiệu macro F1", fraction=0.04, pad=0.02)
    axis.set_title(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_scatter_methods(comparison: pd.DataFrame, path: str | Path, xlabel: str, ylabel: str, title: str) -> None:
    """Mỗi điểm là một điều kiện. Nằm trên đường chéo nghĩa là cách ở trục dọc tốt hơn."""
    figure, axis = plt.subplots(figsize=(5.8, 5.4))
    low = float(min(comparison["base_f1"].min(), comparison["other_f1"].min())) - 0.02
    axis.plot([low, 1.0], [low, 1.0], color="#222222", linewidth=1, linestyle="--")
    cmap = plt.get_cmap("viridis")
    for severity, group in comparison.groupby("severity"):
        axis.scatter(group["base_f1"], group["other_f1"], s=36, color=cmap((severity - 1) / 2), label=f"Mức {severity}", zorder=3)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

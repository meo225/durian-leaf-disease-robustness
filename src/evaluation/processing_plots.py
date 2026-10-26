"""Biểu đồ cho benchmark kỹ thuật xử lý ảnh. Mỗi hàm ghi một file PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

SEVERITY_COLORS = {1: "#4c78a8", 2: "#f58518", 3: "#b3262d"}


def _condition_order(frame: pd.DataFrame, families: Sequence[str]) -> list[str]:
    names = frame.drop_duplicates("condition").set_index("condition")
    return [f"{family}_s{severity}" for family in families for severity in (1, 2, 3) if f"{family}_s{severity}" in names.index]


def plot_matrix_heatmap(matrix: pd.DataFrame, value: str, families: Sequence[str], path: str | Path, title: str, limit: float | None = None) -> None:
    """Kỹ thuật × điều kiện. Ô có khung đen là cặp phù hợp, ô không khung là đối chứng không phù hợp."""
    conditions = _condition_order(matrix, families)
    techniques = list(dict.fromkeys(matrix["technique"]))
    grid = np.full((len(techniques), len(conditions)), np.nan)
    suitable = np.zeros_like(grid, dtype=bool)
    for row, technique in enumerate(techniques):
        for col, condition in enumerate(conditions):
            hit = matrix[(matrix["technique"] == technique) & (matrix["condition"] == condition)]
            if not hit.empty:
                grid[row, col] = hit.iloc[0][value]
                suitable[row, col] = bool(hit.iloc[0]["suitable"])
    bound = limit if limit is not None else max(float(np.nanmax(np.abs(grid))), 0.05)
    fig = Figure(figsize=(0.55 * len(conditions) + 3, 0.7 * len(techniques) + 2.6))
    ax = fig.subplots()
    image = ax.imshow(grid, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound), aspect="auto")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=90, fontsize=7)
    ax.set_yticks(range(len(techniques)))
    ax.set_yticklabels(techniques, fontsize=9)
    for row in range(len(techniques)):
        for col in range(len(conditions)):
            if not np.isnan(grid[row, col]):
                ax.text(col, row, f"{grid[row, col]:+.2f}", ha="center", va="center", fontsize=6, color="black")
                if suitable[row, col]:
                    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2))
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_grid_search(frame: pd.DataFrame, technique: str, families: Sequence[str], grid: Sequence[str], path: str | Path) -> None:
    """Macro F1 của từng cấu hình trên các họ phù hợp, ba cột là ba mức. Đường đứt là điểm khi không xử lý."""
    tuned = frame[(frame["role"] == "tune") & (frame["technique"] == technique)]
    raw = frame[frame["role"] == "raw"].set_index("condition")["macro_f1"]
    clean_f1 = float(raw["clean"])
    columns = min(2, len(families))
    rows = int(np.ceil(len(families) / columns))
    fig = Figure(figsize=(max(6.0, 0.55 * len(grid)) * columns + 1, 3.4 * rows))
    axes = fig.subplots(rows, columns, squeeze=False)
    positions = np.arange(len(grid))
    for index, family in enumerate(families):
        ax = axes[index // columns][index % columns]
        for offset, severity in zip((-0.27, 0.0, 0.27), (1, 2, 3)):
            values = [
                float(tuned[(tuned["family"] == family) & (tuned["config_id"] == config) & (tuned["severity"] == severity)]["macro_f1"].mean())
                for config in grid
            ]
            ax.bar(positions + offset, values, width=0.26, color=SEVERITY_COLORS[severity], label=f"mức {severity}")
            ax.axhline(float(raw[f"{family}_s{severity}"]), color=SEVERITY_COLORS[severity], linestyle="--", linewidth=1)
        ax.axhline(clean_f1, color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(grid, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{technique} trên {family}", fontsize=9)
        ax.set_ylabel("macro F1")
        if index == 0:
            ax.legend(fontsize=7, loc="lower left", ncol=3)
    for index in range(len(families), rows * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle("Cột là kỹ thuật, đường đứt là không xử lý cùng màu mức, đường chấm xám là ảnh sạch", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_best_per_condition(best: pd.DataFrame, clean_f1: float, path: str | Path) -> None:
    """Mỗi điều kiện hai cột: không xử lý và kỹ thuật tốt nhất trên validation. Nhãn là tên kỹ thuật."""
    names = list(best.index)
    positions = np.arange(len(names))
    fig = Figure(figsize=(0.7 * len(names) + 2, 5.2))
    ax = fig.subplots()
    ax.bar(positions - 0.19, best["raw_macro_f1"], width=0.38, color="#9e9e9e", label="không xử lý")
    ax.bar(positions + 0.19, best["macro_f1"], width=0.38, color="#2a7ab0", label="kỹ thuật tốt nhất")
    for position, (_, row) in zip(positions, best.iterrows()):
        ax.text(position + 0.19, row["macro_f1"] + 0.01, row["technique"], rotation=90, ha="center", va="bottom", fontsize=6)
    ax.axhline(clean_f1, color="black", linestyle=":", linewidth=1, label="ảnh sạch")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("macro F1 trên validation")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_clean_cost(cost: pd.DataFrame, classes: Sequence[str], path: str | Path) -> None:
    """Chi phí trên ảnh sạch: thay đổi macro F1 và thay đổi recall từng lớp so với không xử lý."""
    fig = Figure(figsize=(12, 0.6 * len(cost) + 3))
    ax_bar, ax_map = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.5]})
    labels = [f"{row.technique}\n{row.config_id}" for row in cost.itertuples()]
    colors = ["#2a7ab0" if value >= 0 else "#b3262d" for value in cost["change_macro_f1"]]
    ax_bar.barh(range(len(cost)), cost["change_macro_f1"], color=colors)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_yticks(range(len(cost)))
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("thay đổi macro F1 trên ảnh sạch")
    grid = np.array([[row[f"d_recall_{name}"] for name in classes] for _, row in cost.iterrows()])
    bound = max(float(np.abs(grid).max()), 0.02)
    image = ax_map.imshow(grid, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound), aspect="auto")
    ax_map.set_xticks(range(len(classes)))
    ax_map.set_xticklabels([name.replace("_", "\n") for name in classes], fontsize=7)
    ax_map.set_yticks(range(len(cost)))
    ax_map.set_yticklabels([])
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            ax_map.text(col, row, f"{grid[row, col]:+.2f}", ha="center", va="center", fontsize=7)
    ax_map.set_title("thay đổi recall từng lớp trên ảnh sạch", fontsize=9)
    fig.colorbar(image, ax=ax_map, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_confusion_cases(cases: Sequence[tuple[str, Sequence[Sequence[int]]]], classes: Sequence[str], path: str | Path) -> None:
    """Lưới confusion matrix của các ca đáng chú ý, mỗi ca là (tiêu đề, ma trận)."""
    columns = 2
    rows = int(np.ceil(len(cases) / columns))
    fig = Figure(figsize=(5.6 * columns, 4.8 * rows))
    axes = fig.subplots(rows, columns, squeeze=False)
    short = [name.replace("_", "\n") for name in classes]
    for index, (title, matrix) in enumerate(cases):
        ax = axes[index // columns][index % columns]
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(short, fontsize=6)
        ax.set_yticklabels(short, fontsize=6)
        threshold = max(max(row) for row in matrix) / 2
        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                ax.text(c, r, str(value), ha="center", va="center", fontsize=8, color="white" if value > threshold else "black")
        ax.set_title(title, fontsize=8)
    for index in range(len(cases), rows * columns):
        axes[index // columns][index % columns].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_luminance_histograms(items: Sequence[tuple[str, Mapping[str, Sequence[float]]]], path: str | Path) -> None:
    """Phân bố độ sáng trung bình mỗi ảnh ở ba trạng thái: sạch, suy giảm, đã xử lý."""
    fig = Figure(figsize=(4.4 * len(items), 3.6))
    axes = fig.subplots(1, len(items), squeeze=False)[0]
    colors = {"sạch": "#4c78a8", "suy giảm": "#b3262d", "đã xử lý": "#54a24b"}
    for ax, (title, series) in zip(axes, items):
        for label, values in series.items():
            ax.hist(values, bins=np.linspace(0, 1, 41), alpha=0.55, color=colors[label], label=label)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("độ sáng trung bình mỗi ảnh")
    axes[0].set_ylabel("số ảnh")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_rank_table(frame: pd.DataFrame, value: str, path: str | Path, title: str) -> None:
    """Bảng họ × kỹ thuật với giá trị trung bình trên ba mức, ô tốt nhất mỗi họ được tô đậm."""
    table = frame.groupby(["family", "technique"])[value].mean().unstack("technique")
    table = table.reindex(list(dict.fromkeys(frame["family"])))
    fig = Figure(figsize=(1.3 * len(table.columns) + 3, 0.5 * len(table) + 1.8))
    ax = fig.subplots()
    ax.axis("off")
    cells = [[f"{table.iloc[r, c]:+.3f}" if not np.isnan(table.iloc[r, c]) else "" for c in range(table.shape[1])] for r in range(table.shape[0])]
    shown = ax.table(cellText=cells, rowLabels=list(table.index), colLabels=list(table.columns), loc="center", cellLoc="center")
    shown.auto_set_font_size(False)
    shown.set_fontsize(8)
    shown.scale(1, 1.4)
    for r in range(table.shape[0]):
        best = int(np.nanargmax(table.iloc[r].values))
        shown[r + 1, best].set_facecolor("#cfe8cf")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)

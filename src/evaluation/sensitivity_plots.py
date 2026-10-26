"""Biểu đồ cho phân tích độ nhạy (task 4.2). Mỗi hàm ghi một file PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

SEVERITY_COLORS = {1: "#4c78a8", 2: "#f58518", 3: "#b3262d"}
CATEGORY_COLORS = {"severe": "#b3262d", "moderate": "#f58518", "mild": "#e6c229", "negligible": "#9aa0a6"}
CLASS_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b3262d", "#8f6cb8"]


def _short(name: str) -> str:
    return name.replace("ALLOCARIDARA_ATTACK", "ALLOCARIDARA").replace("PHOMOPSIS_LEAF_SPOT", "PHOMOPSIS").replace("ALGAL_LEAF_SPOT", "ALGAL").replace("HEALTHY_LEAF", "HEALTHY").replace("LEAF_BLIGHT", "BLIGHT")


def _save(fig: Figure, path: str | Path, dpi: int = 130) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)


def plot_degradation_heatmap(table: pd.DataFrame, path: str | Path, title: str) -> None:
    """Họ × mức, giá trị là degradation. Dấu sao là khác biệt có ý nghĩa sau hiệu chỉnh Holm."""
    order = table.groupby("family", sort=False)["degradation"].mean().sort_values(ascending=False).index.tolist()
    grid = np.array([[table[(table["family"] == f) & (table["severity"] == s)]["degradation"].iloc[0] for s in (1, 2, 3)] for f in order])
    star = np.array([[bool(table[(table["family"] == f) & (table["severity"] == s)]["significant"].iloc[0]) for s in (1, 2, 3)] for f in order])
    fig = Figure(figsize=(5.6, 0.42 * len(order) + 1.8))
    ax = fig.subplots()
    image = ax.imshow(grid, cmap="Reds", vmin=0, vmax=max(0.6, float(grid.max())), aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["mức 1", "mức 2", "mức 3"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    for row in range(len(order)):
        for col in range(3):
            ax.text(col, row, f"{grid[row, col]:.3f}{'*' if star[row, col] else ''}", ha="center", va="center", fontsize=8, color="white" if grid[row, col] > 0.35 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="F1 sạch trừ F1 suy giảm")
    _save(fig, path)


def plot_severity_curves(table: pd.DataFrame, path: str | Path, title: str) -> None:
    """Macro F1 theo mức của từng họ, kèm khoảng tin cậy bootstrap. Đường ngang là F1 ảnh sạch và khoảng tin cậy của nó."""
    clean = table.attrs["clean"]
    families = list(dict.fromkeys(table["family"]))
    columns = 4
    rows = int(np.ceil(len(families) / columns))
    fig = Figure(figsize=(3.4 * columns, 2.7 * rows))
    axes = np.atleast_2d(fig.subplots(rows, columns, sharey=True))
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, family in zip(axes.ravel(), families):
        part = table[table["family"] == family].sort_values("severity")
        ax.set_visible(True)
        ax.axhspan(clean["clean_f1_low"], clean["clean_f1_high"], color="#54a24b", alpha=0.18)
        ax.axhline(clean["clean_macro_f1"], color="#54a24b", linewidth=1)
        ax.errorbar(part["severity"], part["macro_f1"], yerr=[part["macro_f1"] - part["f1_low"], part["f1_high"] - part["macro_f1"]], marker="o", color="#b3262d", capsize=3)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([str(v) for v in part["value"]], fontsize=8)
        ax.set_title(f"{family}{' (mở rộng)' if part['group'].iloc[0] == 'extension' else ''}", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def plot_flip_counts(table: pd.DataFrame, path: str | Path, title: str) -> None:
    """Mỗi điều kiện: số mẫu đổi từ đúng sang sai (đỏ, sang trái) và từ sai sang đúng (xanh, sang phải)."""
    ordered = table.sort_values("right_to_wrong", ascending=True)
    fig = Figure(figsize=(8, 0.2 * len(ordered) + 1.6))
    ax = fig.subplots()
    positions = np.arange(len(ordered))
    ax.barh(positions, -ordered["right_to_wrong"], color="#b3262d", label="đúng sang sai")
    ax.barh(positions, ordered["wrong_to_right"], color="#4c78a8", label="sai sang đúng")
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["condition"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("số mẫu")
    ax.legend(loc="lower right")
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_class_recall_change(table: pd.DataFrame, classes: Sequence[str], path: str | Path, title: str) -> None:
    """Điều kiện × lớp, giá trị là recall suy giảm trừ recall sạch."""
    clean = table.attrs["clean_recall"]
    ordered = table.sort_values(["degradation"], ascending=False)
    grid = np.array([[row[f"recall_{label}"] - clean[label] for label in classes] for _, row in ordered.iterrows()])
    fig = Figure(figsize=(6.4, 0.22 * len(ordered) + 2))
    ax = fig.subplots()
    image = ax.imshow(grid, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-1, vmax=1), aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([_short(c) for c in classes], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["condition"], fontsize=7)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            ax.text(col, row, f"{grid[row, col]:+.2f}", ha="center", va="center", fontsize=6, color="white" if abs(grid[row, col]) > 0.6 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="thay đổi recall")
    _save(fig, path)


def plot_flip_destinations(flips: pd.DataFrame, classes: Sequence[str], path: str | Path, title: str) -> None:
    """Mẫu đúng ở ảnh sạch nhưng sai khi suy giảm rơi vào lớp nào. Ba cột là ba mức. Cột lệch về một lớp là model sụp về lớp đó."""
    fig = Figure(figsize=(15, 0.36 * flips["condition"].map(lambda c: c.rsplit("_s", 1)[0]).nunique() + 2.4))
    axes = fig.subplots(1, 3, sharey=True)
    families = list(dict.fromkeys(flips["condition"].map(lambda c: c.rsplit("_s", 1)[0])))
    for ax, severity in zip(axes, (1, 2, 3)):
        part = flips.set_index("condition")
        left = np.zeros(len(families))
        for index, label in enumerate(classes):
            share = np.array([part.loc[f"{f}_s{severity}", f"to_{label}"] if f"{f}_s{severity}" in part.index else 0.0 for f in families])
            ax.barh(range(len(families)), share, left=left, color=CLASS_COLORS[index % len(CLASS_COLORS)], label=_short(label))
            left += share
        lost = [int(part.loc[f"{f}_s{severity}", "lost"]) if f"{f}_s{severity}" in part.index else 0 for f in families]
        for row, count in enumerate(lost):
            ax.text(1.01, row, f"{count}", va="center", fontsize=7)
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families, fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.set_title(f"mức {severity}", fontsize=10)
        ax.invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=7, bbox_to_anchor=(0, -0.2), ncol=5)
    fig.suptitle(title + " (số bên phải là số mẫu bị mất)", fontsize=10)
    _save(fig, path)


def plot_confusion_grid(matrices: Mapping[str, np.ndarray], classes: Sequence[str], path: str | Path, title: str) -> None:
    """Confusion matrix chuẩn hóa theo hàng của ảnh sạch và các điều kiện được chọn."""
    names = list(matrices)
    columns = min(4, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig = Figure(figsize=(3.6 * columns, 3.5 * rows))
    axes = np.atleast_2d(fig.subplots(rows, columns))
    for ax in axes.ravel():
        ax.set_visible(False)
    labels = [_short(c)[:6] for c in classes]
    for ax, name in zip(axes.ravel(), names):
        matrix = matrices[name].astype(float)
        norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.set_visible(True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_title(name, fontsize=9)
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def plot_fragile_samples(fragile: pd.DataFrame, images: np.ndarray, conditions_total: int, top: int, path: str | Path, title: str) -> None:
    """Phân bố số điều kiện làm mỗi ảnh sai, và các ảnh mong manh nhất. images là mảng N×H×W×3 trong 0 đến 1 theo chỉ số ảnh."""
    fig = Figure(figsize=(2.2 * top / 2 + 4.6, 6.6))
    grid = fig.add_gridspec(3, max(top // 2, 1), height_ratios=[1.3, 1, 1])
    hist = fig.add_subplot(grid[0, :])
    hist.hist(fragile["flips"], bins=np.arange(0, conditions_total + 2) - 0.5, color="#4c78a8")
    hist.set_xlabel(f"số điều kiện làm ảnh sai, trên {conditions_total}")
    hist.set_ylabel("số ảnh")
    hist.set_title(title, fontsize=10)
    for position, row in enumerate(fragile.head(top).itertuples()):
        ax = fig.add_subplot(grid[1 + position // (top // 2), position % (top // 2)])
        ax.imshow(images[row.index])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{_short(row.label)}\nsai {row.flips}/{conditions_total}\nthường thành {_short(row.common_wrong)}", fontsize=6.5)
    _save(fig, path)


def plot_val_vs_test(val: pd.DataFrame, test: pd.DataFrame, summary: Mapping[str, float], path: str | Path) -> None:
    """Degradation trên validation (đã dùng hiệu chỉnh mức ở 4.1) và trên test. Điểm nằm sát đường chéo là hiệu chỉnh vẫn đúng ở test."""
    merged = val.merge(test, on="condition", suffixes=("_val", "_test"))
    fig = Figure(figsize=(6.2, 6))
    ax = fig.subplots()
    colors = merged["severity_val"].map(SEVERITY_COLORS)
    ax.scatter(merged["degradation_val"], merged["degradation_test"], c=colors, s=36)
    top = max(float(merged[["degradation_val", "degradation_test"]].max().max()), 0.1) * 1.05
    ax.plot([0, top], [0, top], color="gray", linestyle="--", linewidth=1)
    gap = (merged["degradation_test"] - merged["degradation_val"]).abs().sort_values(ascending=False).head(4).index
    for index in gap:
        ax.annotate(merged.loc[index, "condition"], (merged.loc[index, "degradation_val"], merged.loc[index, "degradation_test"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("degradation trên validation")
    ax.set_ylabel("degradation trên test")
    ax.set_title(f"Spearman {summary['spearman']:.3f}, lệch F1 trung bình {summary['mean_abs_f1_gap']:.3f}", fontsize=10)
    for severity, color in SEVERITY_COLORS.items():
        ax.scatter([], [], color=color, label=f"mức {severity}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_priority(summary: pd.DataFrame, path: str | Path, title: str) -> None:
    """Thứ tự ưu tiên cho 4.5: thanh là mức giảm trung bình, vạch là khoảng từ mức nhẹ nhất đến mạnh nhất, nhãn là task xử lý."""
    ordered = summary.sort_values("mean_degradation", ascending=True)
    fig = Figure(figsize=(8.4, 0.38 * len(ordered) + 1.8))
    ax = fig.subplots()
    positions = np.arange(len(ordered))
    ax.barh(positions, ordered["mean_degradation"], color=[CATEGORY_COLORS[c] for c in ordered["category"]])
    ax.hlines(positions, ordered["degradation_s1"], ordered["max_degradation"], color="black", linewidth=1)
    for position, row in zip(positions, ordered.itertuples()):
        label = "+".join(row.handlers) if row.handlers else "không có"
        ax.text(max(row.max_degradation, row.mean_degradation) + 0.01, position, f"task {label}" if row.handlers else "không có kỹ thuật", va="center", fontsize=7)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["family"], fontsize=8)
    ax.set_xlabel("degradation trung bình qua ba mức (vạch: từ mức 1 đến mức lớn nhất)")
    for name, color in CATEGORY_COLORS.items():
        ax.barh([], [], color=color, label=name)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, float(ordered["max_degradation"].max()) + 0.14)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_quality_curves(summary: pd.DataFrame, conditions: Mapping[str, Mapping], clean: Mapping[str, float], external: Mapping[str, tuple[float, float, float]] | None, path: str | Path) -> None:
    """Thống kê chất lượng ảnh theo mức của các họ liên quan. Dải xám là khoảng 10 đến 90 phần trăm của ảnh sạch, dải xanh là của bộ ngoài."""
    panels = [
        ("brightness", "độ sáng", ["brightness_dark", "brightness_bright", "lowlight"]),
        ("contrast", "độ tương phản", ["contrast_low", "contrast_high"]),
        ("sharpness", "độ nét (phương sai Laplacian)", ["blur", "motion_blur", "lowres", "jpeg"]),
        ("noise", "nhiễu (sigma Immerkaer)", ["noise", "sensor_noise", "lowlight"]),
    ]
    fig = Figure(figsize=(12, 8))
    axes = fig.subplots(2, 2).ravel()
    indexed = summary.set_index("condition")
    for ax, (stat, label, families) in zip(axes, panels):
        low, high = clean[f"{stat}_p10"], clean[f"{stat}_p90"]
        ax.axhspan(low, high, color="#9aa0a6", alpha=0.25, label="ảnh sạch")
        if external and stat in external:
            ax.axhspan(external[stat][0], external[stat][2], color="#54a24b", alpha=0.25, label="bộ ngoài")
            ax.axhline(external[stat][1], color="#54a24b", linewidth=1)
        for index, family in enumerate(families):
            names = [f"{family}_s{s}" for s in (0, 1, 2, 3) if s and f"{family}_s{s}" in indexed.index]
            values = [clean[f"{stat}_mean"], *[indexed.loc[n, f"{stat}_mean"] for n in names]]
            ax.plot(range(len(values)), values, marker="o", label=family)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["sạch", "mức 1", "mức 2", "mức 3"])
        ax.set_title(label, fontsize=10)
        if stat in ("sharpness", "noise"):
            ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    _save(fig, path)

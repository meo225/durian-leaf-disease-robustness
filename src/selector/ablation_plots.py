"""Hình cho ablation của bộ chọn (task 5.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KIND_COLORS = {
    "reference": "#9aa5b1", "full": "#1f77b4", "feature_group": "#d98c1f", "action": "#2a9d8f",
    "technique": "#8e6bbf", "rule": "#c4527a", "guard": "#6b8e23", "oracle": "#444444",
}
KIND_NAMES = {
    "reference": "Tham chiếu", "full": "Bộ chọn đầy đủ", "feature_group": "Bỏ nhóm đặc trưng", "action": "Bỏ hành động",
    "technique": "Bỏ kỹ thuật", "rule": "Chỉ một luật", "guard": "Bỏ mệnh đề chặn", "oracle": "Cận trên",
}


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)


def plot_variants(summary: pd.DataFrame, path: str | Path) -> None:
    """Mục tiêu (macro F1 trung bình 42 điều kiện) của mọi biến thể, kèm khoảng tin cậy bootstrap 95% của chính mục tiêu."""
    frame = summary.sort_values("objective").reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(10, 0.36 * len(frame) + 1.6))
    positions = np.arange(len(frame))
    colors = [KIND_COLORS[kind] for kind in frame["kind"]]
    axis.barh(positions, frame["objective"], color=colors, height=0.7)
    axis.errorbar(frame["objective"], positions, xerr=[frame["objective"] - frame["objective_ci_low"], frame["objective_ci_high"] - frame["objective"]],
                  fmt="none", ecolor="#222222", elinewidth=1, capsize=2)
    for line, name, style in ((summary.loc[summary["variant"] == "none", "objective"].iloc[0], "không xử lý", ":"),
                              (summary.loc[summary["variant"] == "full", "objective"].iloc[0], "đầy đủ", "--")):
        axis.axvline(line, color="#222222", linestyle=style, linewidth=1)
        axis.text(line, len(frame) - 0.35, f" {name}", fontsize=8, va="bottom")
    for position, value in zip(positions, frame["objective"]):
        axis.text(value + 0.0035, position, f"{value:.4f}", va="center", fontsize=8)
    axis.set_yticks(positions)
    axis.set_yticklabels(frame["label"])
    axis.set_xlim(max(0.0, frame["objective_ci_low"].min() - 0.02), min(1.0, frame["objective_ci_high"].max() + 0.04))
    axis.set_xlabel("Macro F1 trung bình 42 điều kiện (validation)")
    seen = [kind for kind in KIND_COLORS if kind in set(frame["kind"])]
    axis.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=KIND_COLORS[k]) for k in seen], labels=[KIND_NAMES[k] for k in seen],
                frameon=False, loc="lower right", fontsize=8)
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_contribution(summary: pd.DataFrame, kinds: Sequence[str], title: str, path: str | Path) -> None:
    """Mức mất mục tiêu so với bộ chọn đầy đủ khi bỏ từng thành phần, kèm khoảng tin cậy bootstrap ghép cặp 95%.

    Cột dương nghĩa là thành phần đó có ích. Khoảng tin cậy chứa 0 nghĩa là chưa đủ bằng chứng.
    """
    frame = summary[summary["kind"].isin(kinds)].copy()
    frame["loss"] = -frame["delta_vs_full"]
    frame["loss_low"] = -frame["delta_ci_high"]
    frame["loss_high"] = -frame["delta_ci_low"]
    frame = frame.sort_values("loss").reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(9, 0.42 * len(frame) + 1.5))
    positions = np.arange(len(frame))
    axis.barh(positions, frame["loss"], color=[KIND_COLORS[k] for k in frame["kind"]], height=0.65)
    axis.errorbar(frame["loss"], positions, xerr=[frame["loss"] - frame["loss_low"], frame["loss_high"] - frame["loss"]],
                  fmt="none", ecolor="#222222", elinewidth=1, capsize=2)
    axis.axvline(0, color="#222222", linewidth=1)
    for position, value in zip(positions, frame["loss"]):
        axis.text(value + (0.0008 if value >= 0 else -0.0008), position, f"{value:+.4f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    axis.set_yticks(positions)
    axis.set_yticklabels(frame["label"])
    axis.set_xlabel("Mục tiêu của bộ chọn đầy đủ trừ mục tiêu của biến thể (dương: thành phần có ích)")
    axis.set_title(title, fontsize=11)
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_family_heatmap(gains: pd.DataFrame, summary: pd.DataFrame, path: str | Path, variants: Sequence[str] | None = None) -> None:
    """Thay đổi macro F1 của từng họ suy giảm so với không xử lý (trung bình ba mức), cho từng biến thể."""
    names = [name for name in (variants if variants is not None else gains.index) if name != "none"]
    data = gains.loc[names]
    labels = summary.set_index("variant").loc[names, "label"]
    limit = max(0.02, float(np.abs(data.to_numpy()).max()))
    figure, axis = plt.subplots(figsize=(0.75 * data.shape[1] + 3.5, 0.34 * len(names) + 2.0))
    image = axis.imshow(data.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(data.shape[1]))
    axis.set_xticklabels(data.columns, rotation=40, ha="right")
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(labels)
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            value = data.iat[row, column]
            axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.6 * limit else "#222222")
    figure.colorbar(image, ax=axis, label="Thay đổi macro F1 so với không xử lý", fraction=0.03, pad=0.02)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_tradeoff(summary: pd.DataFrame, path: str | Path) -> None:
    """Mục tiêu so với tỉ lệ ảnh sạch bị đổi dự đoán. Biến thể tốt nằm cao và sát trục dọc."""
    frame = summary[summary["kind"] != "oracle"]
    figure, axis = plt.subplots(figsize=(7.5, 5))
    for kind, group in frame.groupby("kind"):
        axis.scatter(group["changed_share_clean"] * 100, group["objective"], s=55, color=KIND_COLORS[kind], label=KIND_NAMES[kind], zorder=3)
    for _, row in frame.iterrows():
        if row["kind"] in ("full", "feature_group", "technique") or row["variant"] == "none":
            axis.annotate(row["label"], (row["changed_share_clean"] * 100, row["objective"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    axis.set_xlabel("Ảnh sạch bị đổi dự đoán so với không xử lý (%)")
    axis.set_ylabel("Macro F1 trung bình 42 điều kiện (validation)")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

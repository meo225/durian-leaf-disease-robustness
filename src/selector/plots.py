"""Hình cho bộ chọn đã hiệu chỉnh (task 5.3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_by_family(frame: pd.DataFrame, path: str | Path) -> None:
    """Macro F1 trung bình ba mức của từng họ khi không xử lý và khi dùng bộ chọn, kèm tỉ lệ ảnh bị kích hoạt xử lý."""
    degraded = frame[frame["condition"] != "clean"]
    grouped = degraded.groupby("family", sort=False)[["f1_none", "f1_selector", "trigger_rate"]].mean()
    positions = range(len(grouped))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[3, 1.4])
    top.bar([p - 0.2 for p in positions], grouped["f1_none"], 0.4, label="Không xử lý", color="#9aa5b1")
    top.bar([p + 0.2 for p in positions], grouped["f1_selector"], 0.4, label="Bộ chọn", color="#1f77b4")
    top.set_ylabel("Macro F1 (validation)")
    top.set_ylim(max(0.0, grouped[["f1_none", "f1_selector"]].min().min() - 0.05), 1.0)
    top.legend(frameon=False, loc="lower left")
    bottom.bar(list(positions), grouped["trigger_rate"], 0.5, color="#d98c1f")
    bottom.set_ylabel("Tỉ lệ ảnh được xử lý")
    bottom.set_ylim(0, 1)
    bottom.set_xticks(list(positions))
    bottom.set_xticklabels(grouped.index, rotation=40, ha="right")
    for axis in (top, bottom):
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

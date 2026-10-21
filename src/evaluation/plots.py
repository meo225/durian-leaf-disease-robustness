"""Biểu đồ đường huấn luyện và confusion matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_history(history: pd.DataFrame, path: str | Path | None = None) -> Figure:
    """Vẽ loss và macro f1 theo epoch, đánh dấu epoch có macro f1 val cao nhất."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(history["epoch"], history["train_loss"], label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()

    best = history.loc[history["val_macro_f1"].idxmax()]
    ax_f1.plot(history["epoch"], history["val_macro_f1"], label="val macro f1")
    ax_f1.axvline(best["epoch"], color="gray", linestyle="--", label=f"tốt nhất: epoch {int(best['epoch'])}")
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("macro f1")
    ax_f1.legend()

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]] | np.ndarray,
    classes: Sequence[str],
    path: str | Path | None = None,
) -> Figure:
    """Vẽ confusion matrix với số lượng ảnh trong từng ô."""
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(values, cmap="Blues")

    ticks = np.arange(len(classes))
    ax.set_xticks(ticks, labels=classes, rotation=35, ha="right")
    ax.set_yticks(ticks, labels=classes)
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("nhãn thật")

    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                int(values[row, col]),
                ha="center",
                va="center",
                color="white" if values[row, col] > threshold else "black",
            )

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig

"""Biểu đồ đường huấn luyện và confusion matrix, ghi ra file PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from matplotlib.figure import Figure


def plot_history(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
    best_epoch: int | None = None,
) -> None:
    """Vẽ loss và macro F1 validation theo epoch, đánh dấu epoch được chọn nếu có."""
    epochs = [row["epoch"] for row in history]
    fig = Figure(figsize=(10, 4))
    ax_loss, ax_f1 = fig.subplots(1, 2)

    ax_loss.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax_loss.plot(epochs, [row["val_loss"] for row in history], label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_f1.plot(epochs, [row["val_macro_f1"] for row in history], label="val macro F1")
    if best_epoch is not None:
        ax_f1.axvline(best_epoch, color="gray", linestyle="--", label=f"checkpoint: epoch {best_epoch}")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Macro F1")
    ax_f1.set_ylim(0, 1)
    ax_f1.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: str | Path,
) -> None:
    """Vẽ confusion matrix với số ảnh trong từng ô."""
    fig = Figure(figsize=(6.5, 5.5))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    positions = list(range(len(class_names)))
    short_names = [name.replace("_", "\n") for name in class_names]
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    threshold = max(max(row) for row in matrix) / 2
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_corruption_examples(rows: Sequence[tuple[str, Sequence[tuple[str, Any]]]], path: str | Path) -> None:
    """Vẽ lưới ảnh mẫu: mỗi dòng là một họ suy giảm, cột đầu là ảnh sạch, các cột sau là từng mức."""
    columns = max(len(cells) for _, cells in rows)
    fig = Figure(figsize=(2.2 * columns, 2.3 * len(rows)))
    axes = fig.subplots(len(rows), columns, squeeze=False)
    for row_index, (family, cells) in enumerate(rows):
        for col_index in range(columns):
            ax = axes[row_index][col_index]
            ax.set_xticks([])
            ax.set_yticks([])
            if col_index >= len(cells):
                ax.axis("off")
                continue
            title, image = cells[col_index]
            ax.imshow(image)
            ax.set_title(title, fontsize=8)
            if col_index == 0:
                ax.set_ylabel(family, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)

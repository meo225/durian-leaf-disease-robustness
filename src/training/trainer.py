"""Vòng huấn luyện fine-tune, đánh giá trên validation và lưu checkpoint tốt nhất."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Chạy một epoch huấn luyện, trả về loss và accuracy trung bình."""
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += targets.size(0)
    return {"loss": total_loss / count, "accuracy": correct / count}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: Sequence[str],
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Đánh giá model trên một loader, trả về loss (nếu có criterion) và bộ chỉ số phân loại."""
    model.eval()
    total_loss, count = 0.0, 0
    y_true, y_pred = [], []
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, targets).item() * targets.size(0)
        count += targets.size(0)
        y_true.append(targets.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    metrics = compute_metrics(np.concatenate(y_true), np.concatenate(y_pred), classes)
    metrics["loss"] = total_loss / count if criterion is not None else None
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    val_macro_f1: float,
    classes: Sequence[str],
) -> None:
    """Lưu trọng số cùng epoch và macro f1 validation tương ứng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_macro_f1": val_macro_f1,
            "classes": list(classes),
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    """Nạp trọng số từ checkpoint vào model, trả về epoch và macro f1 đã lưu."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    return {"epoch": state["epoch"], "val_macro_f1": state["val_macro_f1"]}


def fit(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    cfg: Mapping[str, Any],
    device: torch.device,
    classes: Sequence[str],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Fine-tune bằng AdamW và cosine annealing, chọn checkpoint theo macro f1 trên validation.

    Dừng sớm khi macro f1 validation không tăng sau `patience` epoch liên tiếp.
    Chỉ dùng loader train và val, không đụng tới test.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    model.to(device)
    history: list[dict[str, float]] = []
    best_f1, best_epoch, stale = -1.0, 0, 0
    start = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_stats = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_stats = evaluate(model, loaders["val"], device, classes, criterion)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_macro_f1": val_stats["macro_f1"],
                "val_balanced_accuracy": val_stats["balanced_accuracy"],
                "seconds": time.perf_counter() - epoch_start,
            }
        )

        if val_stats["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val_stats["macro_f1"], epoch, 0
            save_checkpoint(checkpoint_path, model, epoch, best_f1, classes)
        else:
            stale += 1

        print(
            f"epoch {epoch:02d} | train loss {train_stats['loss']:.4f} acc {train_stats['accuracy']:.4f}"
            f" | val loss {val_stats['loss']:.4f} macro f1 {val_stats['macro_f1']:.4f}"
            f" | {history[-1]['seconds']:.1f}s"
        )

        if stale >= cfg["patience"]:
            break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "epochs_run": len(history),
        "train_seconds": time.perf_counter() - start,
        "optimizer": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__,
    }

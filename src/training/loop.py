"""Vòng fine-tune. Checkpoint chọn bằng macro F1 trên validation."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import classification_scores


def set_seed(seed: int) -> None:
    """Gán seed cho Python, NumPy và PyTorch."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    """Dùng GPU khi có, không thì dùng CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def trainable_parameter_count(model: nn.Module) -> int:
    """Đếm số tham số được cập nhật."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _batches(loader: DataLoader):
    for batch in loader:
        if len(batch) < 2:
            raise ValueError("Mỗi batch cần có tensor ảnh và nhãn.")
        images, labels = batch[0], batch[1]
        yield images, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Một epoch train. Trả về loss trung bình theo mẫu."""
    model.train()
    total_loss = 0.0
    total_count = 0
    for images, labels in _batches(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    if total_count == 0:
        raise ValueError("Train loader không có mẫu.")
    return total_loss / total_count


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Đánh giá một loader. Không cập nhật trọng số."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in _batches(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        predictions = torch.argmax(logits, dim=1)
        y_true.extend(int(value) for value in labels.detach().cpu().tolist())
        y_pred.extend(int(value) for value in predictions.detach().cpu().tolist())
    if total_count == 0:
        raise ValueError("Loader đánh giá không có mẫu.")
    scores = classification_scores(y_true, y_pred, class_names)
    scores["loss"] = total_loss / total_count
    return scores


def build_optimizer(model: nn.Module, train_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Tạo optimizer từ cấu hình. Hiện chỉ dùng AdamW."""
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Optimizer chưa hỗ trợ: {name}.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: Sequence[str],
    train_cfg: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Fine-tune và giữ trọng số có macro F1 validation cao nhất.

    Tập test không được truyền vào hàm này.
    """
    if str(train_cfg.get("selection_metric", "macro_f1")) != "macro_f1":
        raise ValueError("Checkpoint chỉ chọn bằng macro_f1 trên validation.")

    model.to(device)
    optimizer = build_optimizer(model, train_cfg)
    criterion = nn.CrossEntropyLoss()
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    min_delta = float(train_cfg.get("min_delta", 0.0))

    best_score = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_val: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_scores = evaluate_loader(model, val_loader, criterion, device, class_names)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_scores["loss"],
            "val_macro_f1": val_scores["macro_f1"],
            "val_balanced_accuracy": val_scores["balanced_accuracy"],
        }
        history.append(row)
        improved = val_scores["macro_f1"] > best_score + min_delta
        if improved:
            best_score = val_scores["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val = val_scores
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    if best_val is None:
        raise RuntimeError("Chưa có checkpoint validation.")
    return {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "history": history,
        "state_dict": best_state,
    }

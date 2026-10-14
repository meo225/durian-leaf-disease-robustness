"""Fine-tune ResNet-50 trên split train và chọn checkpoint bằng validation.

Tập test không được mở. Kết quả ghi vào outputs/.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.dataloaders import build_dataloaders
from src.models.build import build_model
from src.training.loop import (
    fit_model,
    resolve_device,
    set_seed,
    trainable_parameter_count,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_PATH = ROOT / "configs" / "train_resnet50.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Cấu hình không phải mapping: {path}")
    return loaded


def _save_confusion_figure(matrix: list, class_names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    positions = list(range(len(class_names)))
    short_names = [name.replace("_", "\n") for name in class_names]
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_history_figure(history: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="val macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _resource_name(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    return "cpu"


def save_run_artifacts(
    result: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: torch.device,
    elapsed_seconds: float,
    output_dir: Path,
) -> dict[str, str]:
    """Ghi checkpoint và bảng metric của lần chạy."""
    model_name = str(train_cfg["model_name"])
    seed = int(pipeline_cfg["seed"])
    stem = f"{model_name}_seed{seed}"
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{stem}.pt"
    metrics_path = result_dir / f"{stem}.json"
    best_val = result["best_val"]
    record = {
        "model_name": model_name,
        "timm_name": train_cfg["timm_name"],
        "train_mode": "finetune",
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "selection_metric": "macro_f1",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "resource": _resource_name(device),
        "trainable_parameters": result["trainable_parameters"],
        "hyperparameters": {
            "epochs": train_cfg["epochs"],
            "patience": train_cfg["patience"],
            "min_delta": train_cfg["min_delta"],
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
            "optimizer": train_cfg["optimizer"],
            "batch_size": pipeline_cfg["batch_size"],
            "image_size": pipeline_cfg["image_size"],
            "pretrained": train_cfg["pretrained"],
        },
        "classes": list(pipeline_cfg["classes"]),
        "val_macro_f1": best_val["macro_f1"],
        "val_balanced_accuracy": best_val["balanced_accuracy"],
        "val_loss": best_val["loss"],
        "per_class_recall": best_val["per_class_recall"],
        "confusion_matrix": best_val["confusion_matrix"],
        "history": result["history"],
    }
    torch.save(
        {
            "model_name": model_name,
            "timm_name": train_cfg["timm_name"],
            "classes": list(pipeline_cfg["classes"]),
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "state_dict": result["state_dict"],
            "val_macro_f1": best_val["macro_f1"],
        },
        checkpoint_path,
    )
    metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    figure_path = result_dir / f"{stem}_confusion_matrix.png"
    history_path = result_dir / f"{stem}_history.png"
    _save_confusion_figure(best_val["confusion_matrix"], record["classes"], figure_path)
    _save_history_figure(result["history"], history_path)
    return {
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "confusion_matrix": str(figure_path),
        "history": str(history_path),
    }


def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Đọc cấu hình train và pipeline. Không mở tập test."""
    train_cfg = _load_yaml(TRAIN_CONFIG_PATH)
    if train_cfg.get("model_name") != "resnet50":
        raise ValueError("File này chỉ chạy resnet50.")
    pipeline_cfg = _load_yaml(ROOT / str(train_cfg["pipeline_config"]))
    return train_cfg, pipeline_cfg


def prepare_loaders(pipeline_cfg: dict[str, Any]) -> tuple[torch.device, dict]:
    """Tạo loader train và val. Thiếu GPU thì dừng trước khi train."""
    set_seed(int(pipeline_cfg["seed"]))
    device = resolve_device()
    if device.type != "cuda":
        raise RuntimeError(
            "Notebook train cần GPU. Trong Session options chọn Accelerator GPU, rồi chạy lại từ cell đầu."
        )
    loaders = build_dataloaders(pipeline_cfg, include_test=False)
    if "test" in loaders:
        raise RuntimeError("Task 3.2 không được nạp tập test.")
    return device, loaders


def print_run_report(
    result: dict[str, Any],
    paths: dict[str, str],
    device: torch.device,
    elapsed_seconds: float,
) -> None:
    """In bảng metric ra log. Notebook dùng cùng các số này để hiển thị."""
    best = result["best_val"]
    print(f"Thiết bị: {_resource_name(device)}", flush=True)
    print(f"Thời gian train (giây): {elapsed_seconds:.1f}", flush=True)
    print(f"Số tham số: {result['trainable_parameters']}", flush=True)
    print(f"Epoch được chọn: {result['best_epoch']}", flush=True)
    print(f"Val macro F1: {best['macro_f1']:.4f}", flush=True)
    print(f"Val balanced accuracy: {best['balanced_accuracy']:.4f}", flush=True)
    print(f"Val loss: {best['loss']:.4f}", flush=True)
    print("Recall từng lớp:", flush=True)
    for name, value in best["per_class_recall"].items():
        print(f"  {name}: {value:.4f}", flush=True)
    print(f"Checkpoint: {paths['checkpoint']}", flush=True)
    print(f"Metric: {paths['metrics']}", flush=True)
    print(f"Confusion matrix: {paths['confusion_matrix']}", flush=True)
    print(f"History: {paths['history']}", flush=True)


def main() -> dict[str, str]:
    """Chạy fine-tune ResNet-50. Chỉ tạo loader train và val."""
    train_cfg, pipeline_cfg = load_configs()
    device, loaders = prepare_loaders(pipeline_cfg)
    model = build_model(
        timm_name=str(train_cfg["timm_name"]),
        num_classes=int(pipeline_cfg["num_classes"]),
        pretrained=bool(train_cfg["pretrained"]),
    )
    started = time.perf_counter()
    result = fit_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        class_names=list(pipeline_cfg["classes"]),
        train_cfg=train_cfg,
        device=device,
    )
    elapsed = time.perf_counter() - started
    result["trainable_parameters"] = trainable_parameter_count(model)
    paths = save_run_artifacts(
        result=result,
        pipeline_cfg=pipeline_cfg,
        train_cfg=train_cfg,
        device=device,
        elapsed_seconds=elapsed,
        output_dir=ROOT / "outputs",
    )
    print_run_report(result, paths, device, elapsed)
    return paths


if __name__ == "__main__":
    main()

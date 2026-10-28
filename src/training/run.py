"""Fine-tune một baseline theo file cấu hình train và chọn checkpoint bằng validation.

Dùng chung cho mọi baseline. Tập test không được mở. Kết quả ghi vào outputs/.
Chạy từ dòng lệnh: python -m src.training.run configs/train_resnet50.yaml
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.dataloaders import build_dataloaders
from src.evaluation import plot_confusion_matrix, plot_history
from src.models.build import build_model
from src.training.loop import fit_model, resolve_device, set_seed

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Cấu hình không phải mapping: {path}")
    return loaded


def load_configs(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Đọc cấu hình train và pipeline. Không mở tập test."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    train_cfg = _load_yaml(path)
    pipeline_cfg = _load_yaml(ROOT / str(train_cfg["pipeline_config"]))
    return train_cfg, pipeline_cfg


def load_finetune_configs(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Đọc cấu hình train và pipeline. Nếu config train có `seed` thì dùng seed đó thay cho seed trong configs/pipeline.yaml.

    Cho phép chạy lại cùng cấu hình với seed khác mà không sửa file pipeline dùng chung. Split và seed của suy giảm
    (configs/corruption.yaml) không đổi, nên mọi lần chạy được đánh giá trên đúng cùng ảnh.
    """
    train_cfg, pipeline_cfg = load_configs(config_path)
    if "seed" in train_cfg:
        pipeline_cfg = {**pipeline_cfg, "seed": int(train_cfg["seed"])}
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


def resource_name(device: torch.device) -> str:
    return str(torch.cuda.get_device_name(device)) if device.type == "cuda" else "cpu"


def build_record(
    result: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    resource: str,
) -> dict[str, Any]:
    """Gom kết quả một lần chạy thành một record JSON. Mọi baseline dùng chung định dạng này."""
    best_val = result["best_val"]
    return {
        "model_name": train_cfg["model_name"],
        "library": train_cfg["library"],
        "backbone": train_cfg["backbone"],
        "train_mode": "finetune",
        "seed": int(pipeline_cfg["seed"]),
        "best_epoch": result["best_epoch"],
        "epochs_run": len(result["history"]),
        "selection_metric": train_cfg["selection_metric"],
        "elapsed_seconds": round(result["elapsed_seconds"], 3),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "resource": resource,
        "trainable_parameters": result["trainable_parameters"],
        "hyperparameters": {
            "epochs": train_cfg["epochs"],
            "patience": train_cfg["patience"],
            "min_delta": train_cfg["min_delta"],
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
            "optimizer": train_cfg["optimizer"],
            "scheduler": train_cfg.get("scheduler", "none"),
            "dropout": train_cfg.get("dropout"),
            "batch_size": pipeline_cfg["batch_size"],
            "image_size": pipeline_cfg["image_size"],
            "pretrained": train_cfg["pretrained"],
        },
        "classes": list(pipeline_cfg["classes"]),
        "val_macro_f1": best_val["macro_f1"],
        "val_balanced_accuracy": best_val["balanced_accuracy"],
        "val_accuracy": best_val["accuracy"],
        "val_loss": best_val["loss"],
        "per_class_recall": best_val["per_class_recall"],
        "confusion_matrix": best_val["confusion_matrix"],
        "history": result["history"],
    }


def save_run_artifacts(
    result: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, str]:
    """Ghi checkpoint, JSON và hình của lần chạy."""
    record = build_record(result, pipeline_cfg, train_cfg, resource_name(device))
    stem = f"{record['model_name']}_seed{record['seed']}"
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "checkpoint": checkpoint_dir / f"{stem}.pt",
        "metrics": result_dir / f"{stem}.json",
        "confusion_matrix": result_dir / f"{stem}_confusion_matrix.png",
        "history": result_dir / f"{stem}_history.png",
    }
    torch.save(
        {
            "model_name": record["model_name"],
            "library": record["library"],
            "backbone": record["backbone"],
            "classes": record["classes"],
            "seed": record["seed"],
            "best_epoch": record["best_epoch"],
            "state_dict": result["state_dict"],
            "val_macro_f1": record["val_macro_f1"],
        },
        paths["checkpoint"],
    )
    paths["metrics"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    plot_confusion_matrix(record["confusion_matrix"], record["classes"], paths["confusion_matrix"])
    plot_history(record["history"], paths["history"], best_epoch=record["best_epoch"])
    return {key: str(value) for key, value in paths.items()}


def print_run_report(result: dict[str, Any], paths: dict[str, str], resource: str) -> None:
    """In bảng metric ra log."""
    best = result["best_val"]
    print(f"Thiết bị: {resource}", flush=True)
    print(f"Thời gian train (giây): {result['elapsed_seconds']:.1f}", flush=True)
    print(f"Số tham số: {result['trainable_parameters']}", flush=True)
    print(f"Epoch được chọn: {result['best_epoch']}", flush=True)
    print(f"Val macro F1: {best['macro_f1']:.4f}", flush=True)
    print(f"Val balanced accuracy: {best['balanced_accuracy']:.4f}", flush=True)
    print(f"Val loss: {best['loss']:.4f}", flush=True)
    print("Recall từng lớp:", flush=True)
    for name, value in best["per_class_recall"].items():
        print(f"  {name}: {value:.4f}", flush=True)
    for label, key in (
        ("Checkpoint", "checkpoint"),
        ("Metric", "metrics"),
        ("Confusion matrix", "confusion_matrix"),
        ("History", "history"),
    ):
        print(f"{label}: {paths[key]}", flush=True)


def main(config_path: str | Path) -> dict[str, str]:
    """Chạy fine-tune theo cấu hình. Chỉ tạo loader train và val."""
    train_cfg, pipeline_cfg = load_finetune_configs(config_path)
    device, loaders = prepare_loaders(pipeline_cfg)
    model = build_model(train_cfg, num_classes=int(pipeline_cfg["num_classes"]))
    result = fit_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        class_names=list(pipeline_cfg["classes"]),
        train_cfg=train_cfg,
        device=device,
    )
    paths = save_run_artifacts(result, pipeline_cfg, train_cfg, device, ROOT / "outputs")
    print_run_report(result, paths, resource_name(device))
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Dùng: python -m src.training.run configs/<file>.yaml")
    main(sys.argv[1])

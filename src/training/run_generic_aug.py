"""Đối chứng augmentation tổng quát: train Swin-Tiny với TrivialAugment, không có phép làm hỏng nhắm vào benchmark (task 5.6).

Cùng giao thức với task 4.5 (src/training/run_aug.py): cùng model, optimizer, scheduler, seed, fit_model, cách lưu và
cùng validation hỗn hợp để chọn checkpoint. Chỉ khác loại augmentation lúc train. Mục đích là tách phần cải thiện do
augmentation nói chung khỏi phần do augmentation suy giảm nhắm đúng các họ cần kiểm.

Chạy từ dòng lệnh: python -m src.training.run_generic_aug configs/train_swin_tiny_trivialaugment.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.corruption import build_loader
from src.data.dataloaders import seed_worker
from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import build_train_transform
from src.models.build import build_model
from src.training.loop import evaluate_loader, fit_model, resolve_device, set_seed
from src.training.run import ROOT, load_configs, print_run_report, resource_name, save_run_artifacts
from src.training.run_aug import CachedImages, _image_root, build_mixed_validation

SUPPORTED_METHODS = ("trivialaugment_wide",)


def build_generic_augmentation(train_cfg: Mapping[str, Any]) -> transforms.TrivialAugmentWide:
    """TrivialAugment từ config. Chỉ nhận phương pháp đã khai báo để config sai không âm thầm chạy khác ý."""
    spec = train_cfg["augmentation"]
    method = spec["method"]
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Phương pháp augmentation không hỗ trợ: {method}. Hợp lệ: {SUPPORTED_METHODS}")
    return transforms.TrivialAugmentWide(num_magnitude_bins=int(spec.get("num_magnitude_bins", 31)))


def build_train_steps(pipeline_cfg: Mapping[str, Any], augmentation: transforms.TrivialAugmentWide) -> list[Any]:
    """Pipeline train của baseline với TrivialAugment chèn ngay trước ToTensor (ảnh còn là PIL, sau lật và xoay)."""
    steps = list(build_train_transform(pipeline_cfg).transforms)
    position = next(index for index, step in enumerate(steps) if isinstance(step, transforms.ToTensor))
    steps.insert(position, augmentation)
    return steps


def build_train_loader(train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], augmentation: transforms.TrivialAugmentWide) -> DataLoader:
    """Loader train: cùng tham số loader với task 4.5, chỉ khác bước augmentation."""
    dataset = DurianDataset(
        split_df=get_split("train"),
        image_root=_image_root(pipeline_cfg),
        transform=transforms.Compose(build_train_steps(pipeline_cfg, augmentation)),
        classes=pipeline_cfg["classes"],
    )
    generator = torch.Generator()
    generator.manual_seed(int(pipeline_cfg["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(pipeline_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", pipeline_cfg.get("num_workers", 0))),
        pin_memory=bool(pipeline_cfg.get("pin_memory", True)),
        generator=generator,
        worker_init_fn=seed_worker,
    )


def selection_config(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Khóa cấu hình theo dạng của run_aug để dùng lại cách dựng validation hỗn hợp của task 4.5."""
    spec = train_cfg["validation"]
    return {"augmentation": {"corruption_config": spec["corruption_config"]}, "validation": {"severity": int(spec["severity"])}}


def prepare_loaders(
    train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], require_gpu: bool = True
) -> tuple[torch.device, dict[str, DataLoader], transforms.TrivialAugmentWide]:
    """Seed, thiết bị, loader train có TrivialAugment và loader validation hỗn hợp. Thiếu GPU thì dừng trước khi train."""
    set_seed(int(pipeline_cfg["seed"]))
    device = resolve_device()
    if require_gpu and device.type != "cuda":
        raise RuntimeError("Notebook train cần GPU. Trong Session options chọn Accelerator GPU, rồi chạy lại từ cell đầu.")
    augmentation = build_generic_augmentation(train_cfg)
    val_dataset: CachedImages = build_mixed_validation(selection_config(train_cfg), pipeline_cfg)
    loaders = {
        "train": build_train_loader(train_cfg, pipeline_cfg, augmentation),
        "val": DataLoader(val_dataset, batch_size=int(pipeline_cfg["batch_size"]) * 2, shuffle=False, num_workers=0),
    }
    return device, loaders, augmentation


def amend_record(
    paths: Mapping[str, str],
    train_cfg: Mapping[str, Any],
    augmentation: transforms.TrivialAugmentWide,
    clean_val: Mapping[str, Any],
    validation_images: int,
) -> dict[str, Any]:
    """Thêm vào file kết quả những gì khác baseline: augmentation, quy tắc chọn checkpoint và điểm trên validation sạch."""
    record = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
    record["train_mode"] = "finetune_with_generic_augmentation"
    record["val_set"] = "mixed"
    record["val_selection"] = {
        "description": "Ảnh sạch cộng mỗi họ suy giảm ở một mức cố định, giống task 4.5. Các chỉ số val_* ở trên tính trên tập hỗn hợp này, không phải validation sạch.",
        "severity": int(train_cfg["validation"]["severity"]),
        "images": int(validation_images),
    }
    record["augmentation"] = {
        "method": train_cfg["augmentation"]["method"],
        "num_magnitude_bins": int(train_cfg["augmentation"].get("num_magnitude_bins", 31)),
        "repr": repr(augmentation),
        "degradation_families_in_training": [],
    }
    record["clean_val"] = {
        "macro_f1": clean_val["macro_f1"],
        "balanced_accuracy": clean_val["balanced_accuracy"],
        "accuracy": clean_val["accuracy"],
        "per_class_recall": clean_val["per_class_recall"],
        "confusion_matrix": clean_val["confusion_matrix"],
    }
    Path(paths["metrics"]).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main(config_path: str | Path, require_gpu: bool = True) -> dict[str, str]:
    """Train với TrivialAugment, lưu checkpoint và kết quả. Chỉ dùng train và validation."""
    train_cfg, pipeline_cfg = load_configs(config_path)
    device, loaders, augmentation = prepare_loaders(train_cfg, pipeline_cfg, require_gpu)
    print(f"Augmentation: {augmentation}", flush=True)
    print(f"Ảnh train: {len(loaders['train'].dataset)}, validation hỗn hợp: {len(loaders['val'].dataset)}", flush=True)
    print("Tập test: không mở", flush=True)

    model = build_model(train_cfg, num_classes=int(pipeline_cfg["num_classes"]))
    classes = list(pipeline_cfg["classes"])
    result = fit_model(model=model, train_loader=loaders["train"], val_loader=loaders["val"], class_names=classes, train_cfg=train_cfg, device=device)
    paths = save_run_artifacts(result, pipeline_cfg, train_cfg, device, ROOT / "outputs")

    clean_val = evaluate_loader(model, build_loader(pipeline_cfg, "val"), torch.nn.CrossEntropyLoss(), device, classes)
    amend_record(paths, train_cfg, augmentation, clean_val, len(loaders["val"].dataset))
    print_run_report(result, paths, resource_name(device))
    print(f"Validation sạch (không dùng để chọn checkpoint): macro F1 {clean_val['macro_f1']:.4f}", flush=True)
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Dùng: python -m src.training.run_generic_aug configs/<file>.yaml")
    main(sys.argv[1])

"""Đối chứng huấn luyện lại reference model bằng augmentation suy giảm (task 4.5).

Cùng giao thức với baseline (src/training/run.py): cùng model, optimizer, scheduler, seed, fit_model và cách lưu. Chỉ khác hai điểm:
- Train thêm augmentation suy giảm ngẫu nhiên lấy từ các họ của benchmark 4.1 (src/corruption/augment.py).
- Checkpoint chọn bằng macro F1 trên validation hỗn hợp: ảnh sạch cộng mỗi họ suy giảm ở một mức cố định, tạo sẵn một lần.
  Quy tắc này khóa trước khi chạy, và validation là nơi duy nhất dùng để chọn. Tập test không được mở ở bước này.

Chạy từ dòng lệnh: python -m src.training.run_aug configs/train_convnext_tiny_aug.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.corruption import build_loader, corrupt_tensor, load_conditions, load_seed
from src.corruption.augment import RandomCorruption
from src.data.dataloaders import seed_worker
from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import ConditionalResize, build_train_transform
from src.models.build import build_model
from src.training.loop import evaluate_loader, fit_model, resolve_device, set_seed
from src.training.run import ROOT, load_configs, print_run_report, resource_name, save_run_artifacts


class CachedImages(Dataset):
    """Ảnh uint8 N×3×H×W đã tạo sẵn, chuẩn hóa khi lấy. Dùng cho validation hỗn hợp để không tạo lại suy giảm mỗi epoch."""

    def __init__(self, images: torch.Tensor, labels: list[int], mean: list[float], std: list[float]) -> None:
        self.images = images
        self.labels = labels
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = (self.images[index].float() / 255.0 - self.mean) / self.std
        return image, self.labels[index]


def _image_root(pipeline_cfg: Mapping[str, Any]) -> Path:
    root = Path(pipeline_cfg["image_root"])
    return root if root.is_absolute() else ROOT / root


def build_augmentation(train_cfg: Mapping[str, Any]) -> RandomCorruption:
    """Augmentation suy giảm từ config. Lấy điều kiện của benchmark 4.1, không dùng tập test."""
    spec = train_cfg["augmentation"]
    conditions = load_conditions(spec["corruption_config"], "train")
    return RandomCorruption(
        conditions,
        probability=float(spec["probability"]),
        jitter=float(spec["jitter"]),
        exclude_families=list(spec.get("exclude_families") or []),
    )


def build_train_loader(train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], augmentation: RandomCorruption) -> DataLoader:
    """Loader train: pipeline chuẩn của baseline, cộng suy giảm ngẫu nhiên sau ToTensor và trước chuẩn hóa."""
    steps = list(build_train_transform(pipeline_cfg).transforms)
    position = next(index for index, step in enumerate(steps) if isinstance(step, transforms.ToTensor))
    steps.insert(position + 1, augmentation)
    dataset = DurianDataset(
        split_df=get_split("train"),
        image_root=_image_root(pipeline_cfg),
        transform=transforms.Compose(steps),
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


def validation_condition_names(train_cfg: Mapping[str, Any]) -> list[str]:
    """Tên điều kiện của validation hỗn hợp: mỗi họ ở mức cố định trong config. Ảnh sạch thêm riêng."""
    conditions = load_conditions(train_cfg["augmentation"]["corruption_config"], "val")
    severity = int(train_cfg["validation"]["severity"])
    families = list(dict.fromkeys(spec["family"] for spec in conditions.values()))
    names = [f"{family}_s{severity}" for family in families]
    missing = [name for name in names if name not in conditions]
    if missing:
        raise ValueError(f"Mức {severity} không có ở các họ: {missing}")
    return names


def build_mixed_validation(train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any]) -> CachedImages:
    """Validation hỗn hợp: 444 ảnh sạch cộng mỗi họ ở một mức, suy giảm cố định theo benchmark 4.1."""
    spec_path = train_cfg["augmentation"]["corruption_config"]
    conditions = load_conditions(spec_path, "val")
    seed = load_seed(spec_path)
    dataset = DurianDataset(
        split_df=get_split("val"),
        image_root=_image_root(pipeline_cfg),
        transform=transforms.Compose([ConditionalResize(int(pipeline_cfg["image_size"])), transforms.ToTensor()]),
        classes=pipeline_cfg["classes"],
    )
    clean = [dataset[index] for index in range(len(dataset))]
    labels = [item[1] for item in clean]
    chunks = [torch.stack([torch.round(item[0] * 255).to(torch.uint8) for item in clean])]
    for name in validation_condition_names(train_cfg):
        chunks.append(torch.stack([torch.round(corrupt_tensor(item[0], name, conditions[name], seed) * 255).to(torch.uint8) for item in clean]))
    return CachedImages(torch.cat(chunks), labels * len(chunks), list(pipeline_cfg["mean"]), list(pipeline_cfg["std"]))


def prepare_loaders(
    train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], require_gpu: bool = True
) -> tuple[torch.device, dict[str, DataLoader], RandomCorruption]:
    """Seed, thiết bị, loader train có augmentation và loader validation hỗn hợp. Thiếu GPU thì dừng trước khi train."""
    set_seed(int(pipeline_cfg["seed"]))
    device = resolve_device()
    if require_gpu and device.type != "cuda":
        raise RuntimeError("Notebook train cần GPU. Trong Session options chọn Accelerator GPU, rồi chạy lại từ cell đầu.")
    augmentation = build_augmentation(train_cfg)
    val_dataset = build_mixed_validation(train_cfg, pipeline_cfg)
    loaders = {
        "train": build_train_loader(train_cfg, pipeline_cfg, augmentation),
        "val": DataLoader(val_dataset, batch_size=int(pipeline_cfg["batch_size"]) * 2, shuffle=False, num_workers=0),
    }
    return device, loaders, augmentation


def amend_record(
    paths: Mapping[str, str],
    train_cfg: Mapping[str, Any],
    augmentation: RandomCorruption,
    clean_val: Mapping[str, Any],
    validation_images: int,
) -> dict[str, Any]:
    """Thêm vào file kết quả những gì khác baseline: augmentation, quy tắc chọn checkpoint và điểm trên validation sạch."""
    record = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
    record["train_mode"] = "finetune_with_corruption_augmentation"
    record["val_set"] = "mixed"
    record["val_selection"] = {
        "description": "Ảnh sạch cộng mỗi họ suy giảm ở một mức cố định. Các chỉ số val_* ở trên tính trên tập hỗn hợp này, không phải validation sạch.",
        "severity": int(train_cfg["validation"]["severity"]),
        "images": int(validation_images),
    }
    record["augmentation"] = {
        "corruption_config": train_cfg["augmentation"]["corruption_config"],
        "probability": augmentation.probability,
        "jitter": augmentation.jitter,
        "families": augmentation.families(),
        "excluded_families": list(train_cfg["augmentation"].get("exclude_families") or []),
        "conditions": len(augmentation.specs),
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
    """Train với augmentation suy giảm, lưu checkpoint và kết quả. Chỉ dùng train và validation."""
    train_cfg, pipeline_cfg = load_configs(config_path)
    device, loaders, augmentation = prepare_loaders(train_cfg, pipeline_cfg, require_gpu)
    print(f"Augmentation: {augmentation}", flush=True)
    print(f"Ảnh train: {len(loaders['train'].dataset)}, validation hỗn hợp: {len(loaders['val'].dataset)}", flush=True)
    print("Tập test: không mở", flush=True)

    model = build_model(train_cfg, num_classes=int(pipeline_cfg["num_classes"]))
    classes = list(pipeline_cfg["classes"])
    result = fit_model(model=model, train_loader=loaders["train"], val_loader=loaders["val"], class_names=classes, train_cfg=train_cfg, device=device)
    paths = save_run_artifacts(result, pipeline_cfg, train_cfg, device, ROOT / "outputs")

    clean_loader = build_loader(pipeline_cfg, "val")
    clean_val = evaluate_loader(model, clean_loader, torch.nn.CrossEntropyLoss(), device, classes)
    amend_record(paths, train_cfg, augmentation, clean_val, len(loaders["val"].dataset))
    print_run_report(result, paths, resource_name(device))
    print(f"Validation sạch (không dùng để chọn checkpoint): macro F1 {clean_val['macro_f1']:.4f}", flush=True)
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Dùng: python -m src.training.run_aug configs/<file>.yaml")
    main(sys.argv[1])

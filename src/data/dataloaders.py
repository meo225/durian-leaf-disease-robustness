"""Xây dựng PyTorch DataLoader cho DurianLDD."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import build_eval_transform, build_train_transform

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def seed_worker(worker_id: int) -> None:
    """Khởi tạo seed ngẫu nhiên cho từng tiến trình worker của DataLoader."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _get_param(cfg: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_dataloaders(
    cfg: Mapping[str, Any] | Any,
    include_test: bool = False,
) -> dict[str, DataLoader]:
    """Tạo DataLoader cho các tập dữ liệu theo cấu hình.

    Args:
        cfg: Từ điển hoặc đối tượng cấu hình từ pipeline.yaml.
        include_test: Cờ kích hoạt nạp tập test. Nếu True, hàm get_split('test')
            sẽ kiểm tra biến môi trường FINAL_EVAL=1 trước khi mở.

    Returns:
        Từ điển chứa các DataLoader: 'train', 'val', và tùy chọn 'test'.
    """
    raw_root = _get_param(cfg, "image_root", None) or _get_param(cfg, "dataset_root", "datasets/durian-ldd")
    if isinstance(raw_root, Path):
        image_root = raw_root
    else:
        root_path = Path(raw_root)
        image_root = root_path if root_path.is_absolute() else _ROOT_DIR / root_path

    batch_size = int(_get_param(cfg, "batch_size", 32))
    num_workers = int(_get_param(cfg, "num_workers", 0))
    pin_memory = bool(_get_param(cfg, "pin_memory", True))
    seed = int(_get_param(cfg, "seed", 42))
    classes = _get_param(cfg, "classes", None)

    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_df = get_split("train")
    train_dataset = DurianDataset(
        split_df=train_df,
        image_root=image_root,
        transform=train_transform,
        classes=classes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    val_df = get_split("val")
    val_dataset = DurianDataset(
        split_df=val_df,
        image_root=image_root,
        transform=eval_transform,
        classes=classes,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    loaders: dict[str, DataLoader] = {
        "train": train_loader,
        "val": val_loader,
    }

    if include_test:
        test_df = get_split("test")
        test_dataset = DurianDataset(
            split_df=test_df,
            image_root=image_root,
            transform=eval_transform,
            classes=classes,
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders

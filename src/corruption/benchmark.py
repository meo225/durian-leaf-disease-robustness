"""Benchmark suy giảm ảnh: điều kiện cố định, nhiễu lặp lại được, loader trên đúng mẫu của một split."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.corruption.basic import apply_corruption
from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import ConditionalResize

ROOT = Path(__file__).resolve().parents[2]


def load_conditions(config_path: str | Path, split: str) -> dict[str, dict[str, Any]]:
    """Đọc các điều kiện suy giảm từ config. Tập test chỉ dùng được khi config đã khóa.

    Tên điều kiện có dạng `<họ>_s<mức>`, ví dụ `blur_s2`. Mỗi điều kiện có type, value, family và severity.
    """
    path = Path(config_path)
    cfg = yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8"))
    if split == "test" and not cfg["locked"]:
        raise PermissionError("Config suy giảm chưa khóa (locked: false). Không mở tập test.")
    return {
        f"{family}_s{severity}": {"type": spec["type"], "value": value, "family": family, "severity": severity}
        for family, spec in cfg["families"].items()
        for severity, value in enumerate(spec["levels"], start=1)
    }


def load_seed(config_path: str | Path) -> int:
    path = Path(config_path)
    return int(yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8"))["seed"])


def corrupt_tensor(image: torch.Tensor, name: str, spec: Mapping[str, Any], seed: int) -> torch.Tensor:
    """Áp một điều kiện lên ảnh 0 đến 1 rồi làm tròn về 8 bit, như khi lưu ảnh.

    Nhiễu lấy seed từ nội dung ảnh, tên điều kiện và seed chung. Một ảnh luôn nhận cùng một nhiễu,
    không phụ thuộc thứ tự batch hay số worker.
    """
    generator = None
    if spec["type"] == "noise":
        digest = hashlib.sha256(image.numpy().tobytes() + f"{seed}:{name}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF)
    return torch.round(apply_corruption(image, spec, generator) * 255) / 255


def build_loader(
    pipeline_cfg: Mapping[str, Any],
    split: str,
    name: str | None = None,
    spec: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> DataLoader:
    """Loader trên một split. Không truyền spec thì là ảnh sạch. Suy giảm áp trước bước chuẩn hóa.

    Thứ tự, mẫu và nhãn giống hệt giữa mọi điều kiện, nên so được từng ảnh sạch với ảnh suy giảm.
    """
    steps: list[Any] = [ConditionalResize(int(pipeline_cfg["image_size"])), transforms.ToTensor()]
    if spec is not None:
        steps.append(transforms.Lambda(lambda image: corrupt_tensor(image, str(name), spec, seed)))
    steps.append(transforms.Normalize(pipeline_cfg["mean"], pipeline_cfg["std"]))
    root = Path(pipeline_cfg["image_root"])
    dataset = DurianDataset(
        split_df=get_split(split),
        image_root=root if root.is_absolute() else ROOT / root,
        transform=transforms.Compose(steps),
        classes=pipeline_cfg["classes"],
    )
    return DataLoader(dataset, batch_size=int(pipeline_cfg["batch_size"]), shuffle=False, num_workers=0)


def example_rows(
    pipeline_cfg: Mapping[str, Any],
    split: str,
    conditions: Mapping[str, Mapping[str, Any]],
    seed: int,
    index: int,
) -> list[tuple[str, list[tuple[str, Any]]]]:
    """Một ảnh mẫu qua mọi mức, mỗi họ một dòng: ảnh sạch rồi các mức. Ảnh là mảng H×W×3 trong khoảng 0 đến 1."""
    root = Path(pipeline_cfg["image_root"])
    dataset = DurianDataset(
        split_df=get_split(split),
        image_root=root if root.is_absolute() else ROOT / root,
        transform=transforms.Compose([ConditionalResize(int(pipeline_cfg["image_size"])), transforms.ToTensor()]),
        classes=pipeline_cfg["classes"],
    )
    image = dataset[index][0]
    rows = []
    for family in dict.fromkeys(spec["family"] for spec in conditions.values()):
        cells = [("sạch", image.permute(1, 2, 0).numpy())]
        for name, spec in conditions.items():
            if spec["family"] == family:
                cells.append((str(spec["value"]), corrupt_tensor(image, name, spec, seed).permute(1, 2, 0).numpy()))
        rows.append((family, cells))
    return rows

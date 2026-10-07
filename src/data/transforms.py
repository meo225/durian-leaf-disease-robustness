"""Xây dựng transform tiền xử lý và tăng cường ảnh chuẩn cho DurianLDD."""

from __future__ import annotations

from typing import Any, Mapping

from PIL import Image
import torch
from torchvision import transforms


class ConditionalResize:
    """Resize ảnh về kích thước chỉ định chỉ khi kích thước hiện tại khác kích thước mục tiêu."""

    def __init__(self, size: int | tuple[int, int]) -> None:
        if isinstance(size, int):
            self.target_size = (size, size)
        else:
            self.target_size = (size[0], size[1])

    def __call__(self, img: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:
        target_h, target_w = self.target_size
        if isinstance(img, Image.Image):
            current_w, current_h = img.size
        elif isinstance(img, torch.Tensor):
            current_h, current_w = img.shape[-2], img.shape[-1]
        else:
            return transforms.functional.resize(img, [target_h, target_w])

        if (current_h, current_w) != (target_h, target_w):
            return transforms.functional.resize(img, [target_h, target_w])
        return img

    def __repr__(self) -> str:
        return f"ConditionalResize(size={self.target_size})"


def _get_param(cfg: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_train_transform(cfg: Mapping[str, Any] | Any) -> transforms.Compose:
    """Tạo chuỗi transform huấn luyện gồm tăng cường hình học, ToTensor và chuẩn hóa."""
    image_size = _get_param(cfg, "image_size", 224)
    mean = _get_param(cfg, "mean", [0.485, 0.456, 0.406])
    std = _get_param(cfg, "std", [0.229, 0.224, 0.225])
    hflip_prob = _get_param(cfg, "hflip_prob", 0.5)
    vflip_prob = _get_param(cfg, "vflip_prob", 0.5)
    rotation_degrees = _get_param(cfg, "rotation_degrees", 15)

    return transforms.Compose([
        ConditionalResize(size=image_size),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.RandomVerticalFlip(p=vflip_prob),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_eval_transform(cfg: Mapping[str, Any] | Any) -> transforms.Compose:
    """Tạo chuỗi transform đánh giá (dùng chung cho val và test) không chứa augmentation."""
    image_size = _get_param(cfg, "image_size", 224)
    mean = _get_param(cfg, "mean", [0.485, 0.456, 0.406])
    std = _get_param(cfg, "std", [0.229, 0.224, 0.225])

    return transforms.Compose([
        ConditionalResize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

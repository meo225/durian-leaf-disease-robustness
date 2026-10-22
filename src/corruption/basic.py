"""Bốn kiểu suy giảm cơ bản trên tensor ảnh RGB trong khoảng 0 đến 1, dạng C×H×W."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch
from torchvision.transforms import functional as TF


def brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Nhân độ sáng với factor. Dưới 1 là tối đi, trên 1 là sáng lên."""
    return (image * factor).clamp(0.0, 1.0)


def contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Kéo giá trị về hoặc ra xa trung bình xám của ảnh."""
    mean = image.mean()
    return ((image - mean) * factor + mean).clamp(0.0, 1.0)


def blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Làm mờ Gauss với độ lệch chuẩn sigma tính theo pixel."""
    kernel = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(image, [kernel, kernel], [sigma, sigma])


def noise(image: torch.Tensor, std: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Cộng nhiễu Gauss với độ lệch chuẩn std. Có generator thì nhiễu lặp lại được."""
    sample = torch.randn(image.shape, generator=generator) if generator is not None else torch.randn_like(image)
    return (image + sample * std).clamp(0.0, 1.0)


CORRUPTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "brightness": brightness,
    "contrast": contrast,
    "blur": blur,
    "noise": noise,
}


def apply_corruption(
    image: torch.Tensor, spec: Mapping[str, Any], generator: torch.Generator | None = None
) -> torch.Tensor:
    """Áp một suy giảm theo spec, ví dụ {"type": "brightness", "value": 0.5}. Generator chỉ dùng cho noise."""
    kind = str(spec["type"])
    if kind not in CORRUPTIONS:
        raise ValueError(f"Kiểu suy giảm chưa hỗ trợ: {kind}.")
    if kind == "noise":
        return noise(image, float(spec["value"]), generator)
    return CORRUPTIONS[kind](image, float(spec["value"]))

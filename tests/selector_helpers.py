"""Hàm dùng chung cho tests/test_selector.py và tests/test_selector_calibration.py."""

import torch
import torch.nn.functional as F
from torch import nn


def scene(seed: int) -> torch.Tensor:
    """Ảnh tổng hợp có kết cấu thô, kết cấu mịn và một mảng cạnh, đủ để đặc trưng độ nét và nhiễu có nghĩa."""
    generator = torch.Generator().manual_seed(seed)
    coarse = F.interpolate(torch.rand(1, 3, 14, 14, generator=generator), size=224, mode="bicubic", align_corners=False)[0]
    fine = F.interpolate(torch.rand(1, 3, 56, 56, generator=generator), size=224, mode="bilinear", align_corners=False)[0]
    image = 0.6 * coarse + 0.4 * fine
    image[:, 60:160, 80:140] = image[:, 60:160, 80:140] * 0.4 + 0.5
    return ((image - image.min()) / (image.max() - image.min())).clamp(0.0, 1.0)


def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 5, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))


def uint8_images(count: int) -> torch.Tensor:
    return torch.stack([torch.round(scene(seed) * 255).to(torch.uint8) for seed in range(count)])

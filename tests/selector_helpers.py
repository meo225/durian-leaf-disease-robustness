"""Hàm dùng chung cho tests/test_selector.py và tests/test_selector_calibration.py."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.selector.calibrate import CLEAN, CalibrationTable
from src.selector.features import FEATURES

ACTIONS = ["keep", "unsharp_strong", "unsharp_mild", "bilateral_wide", "bilateral_narrow"]


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


def synthetic_table(images: int = 100, seed: int = 0, helpful: bool = True, separable: bool = True) -> CalibrationTable:
    """Bảng ba điều kiện (sạch, mờ, nhiễu). Khi `helpful`, unsharp chữa mờ và bilateral chữa nhiễu, còn trên ảnh sạch cả hai đều làm hỏng."""
    rng = np.random.default_rng(seed)
    labels = np.arange(images) % 5
    wrong = (labels + 1) % 5
    count = 3
    features = rng.uniform(0.2, 0.8, size=(count, images, len(FEATURES))).astype(np.float32)
    index = {name: FEATURES.index(name) for name in FEATURES}
    if separable:
        features[0, :, index["sharpness"]] = rng.uniform(0.5, 1.0, images)
        features[1, :, index["sharpness"]] = rng.uniform(0.0, 0.1, images)
        features[2, :, index["sharpness"]] = rng.uniform(0.5, 1.0, images)
        features[[0, 1], :, index["noise_sigma"]] = rng.uniform(0.0, 0.01, (2, images))
        features[2, :, index["noise_sigma"]] = rng.uniform(0.05, 0.1, images)
        features[:, :, index["noise_ratio"]] = features[:, :, index["noise_sigma"]]
    predictions = np.zeros((len(ACTIONS), count, images), dtype=np.int8)
    predictions[0, 0] = labels
    predictions[0, 1:] = np.where(np.arange(images) % 10 < 7, wrong, labels)
    for action in range(1, len(ACTIONS)):
        predictions[action, 0] = wrong if helpful else labels
        predictions[action, 1:] = predictions[0, 1:]
    if helpful:
        predictions[ACTIONS.index("unsharp_strong"), 1] = labels
        predictions[ACTIONS.index("bilateral_wide"), 2] = labels
    return CalibrationTable(
        [CLEAN, "blur_s1", "noise_s1"], [CLEAN, "blur", "noise"], list(ACTIONS), features, predictions, None, labels,
        [f"ảnh_{i}.jpg" for i in range(images)], [f"lớp_{i}" for i in range(5)],
    )

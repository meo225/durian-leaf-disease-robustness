"""Kỹ thuật cải thiện ánh sáng, tương phản và màu (task 4.3).

Tất cả nhận ảnh RGB C×H×W trong khoảng 0 đến 1 và trả ảnh cùng dạng, đã làm tròn về 8 bit như khi lưu ảnh.
Giao diện giống src/corruption nên dùng chung được với các nhóm kỹ thuật khác.

Gamma, gamma tự động, CLAHE, Retinex và cân bằng trắng gray-world. Tham số cụ thể nằm ở configs/illumination.yaml.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch

LUMA = (0.299, 0.587, 0.114)


def _quantize(image: torch.Tensor) -> torch.Tensor:
    return torch.round(image.clamp(0.0, 1.0) * 255) / 255


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray((image.clamp(0.0, 1.0) * 255).round().byte().permute(1, 2, 0).numpy())


def _from_uint8(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0


def mean_luminance(image: torch.Tensor) -> float:
    """Độ sáng trung bình của ảnh, trong khoảng 0 đến 1."""
    weights = torch.tensor(LUMA).view(3, 1, 1)
    return float((image * weights).sum(dim=0).mean())


def gamma(image: torch.Tensor, gamma: float) -> torch.Tensor:
    """Hiệu chỉnh gamma với một giá trị cố định. Dưới 1 làm sáng vùng tối, trên 1 làm tối vùng sáng."""
    return _quantize(image.clamp(0.0, 1.0) ** float(gamma))


def gamma_auto(image: torch.Tensor, target: float, low: float = 0.3, high: float = 3.0) -> torch.Tensor:
    """Gamma tự chọn theo từng ảnh để độ sáng trung bình về gần target. Không cần biết ảnh bị tối hay sáng."""
    mean = min(max(mean_luminance(image), 1e-3), 1.0 - 1e-3)
    value = math.log(target) / math.log(mean)
    return gamma(image, min(max(value, low), high))


def clahe(image: torch.Tensor, clip: float, tiles: int) -> torch.Tensor:
    """Cân bằng histogram thích ứng có giới hạn tương phản, chỉ trên kênh sáng L của không gian LAB để giữ màu."""
    lab = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2LAB)
    operator = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    lab[..., 0] = operator.apply(lab[..., 0])
    return _from_uint8(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def retinex(image: torch.Tensor, sigmas: Sequence[float], alpha: float = 1.0) -> torch.Tensor:
    """Multi-scale Retinex trên kênh giá trị V của HSV, giữ nguyên sắc độ và độ bão hòa.

    Lấy log ảnh trừ log ảnh làm mờ ở từng sigma để bỏ phần chiếu sáng chậm thay đổi, rồi co giãn về 0 đến 255.
    alpha trộn kết quả với ảnh gốc, alpha bằng 1 là dùng hoàn toàn Retinex.
    """
    hsv = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2HSV)
    value = hsv[..., 2].astype(np.float32) + 1.0
    log_value = np.log(value)
    response = np.zeros_like(value)
    for sigma in sigmas:
        response += log_value - np.log(cv2.GaussianBlur(value, (0, 0), float(sigma)))
    response /= len(sigmas)
    low, high = np.percentile(response, (1, 99))
    stretched = np.clip((response - low) / (high - low + 1e-6), 0.0, 1.0) * 255.0
    mixed = (1.0 - alpha) * (value - 1.0) + alpha * stretched
    hsv[..., 2] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return _from_uint8(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))


def gray_world(image: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Cân bằng trắng gray-world: giả định màu trung bình của cảnh là xám, nên kéo ba kênh về cùng một mức.

    strength bằng 1 là cân bằng hoàn toàn, nhỏ hơn là kéo một phần.
    """
    means = image.mean(dim=(1, 2)).clamp_min(1e-3)
    gains = (means.mean() / means).clamp(0.5, 2.0)
    gains = 1.0 + float(strength) * (gains - 1.0)
    return _quantize(image * gains.view(3, 1, 1))


TECHNIQUES: dict[str, Callable[..., torch.Tensor]] = {
    "gamma": gamma,
    "gamma_auto": gamma_auto,
    "clahe": clahe,
    "retinex": retinex,
    "gray_world": gray_world,
}


def apply_processing(image: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    """Áp một kỹ thuật theo spec, ví dụ {"technique": "clahe", "params": {"clip": 2.0, "tiles": 8}}."""
    name = str(spec["technique"])
    if name not in TECHNIQUES:
        raise ValueError(f"Kỹ thuật chưa hỗ trợ: {name}.")
    return TECHNIQUES[name](image, **dict(spec.get("params", {})))


def load_selected(path: str | Path) -> dict[str, Any]:
    """Đọc kết quả chọn tham số đã lưu ở reports/processing/illumination_selected.json."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

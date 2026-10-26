"""Kỹ thuật khử nhiễu và làm nét (task 4.4).

Tất cả nhận ảnh RGB C×H×W trong khoảng 0 đến 1 và trả ảnh cùng dạng, đã làm tròn về 8 bit như khi lưu ảnh.
Giao diện giống illumination.py và src/corruption nên dùng chung được với các nhóm kỹ thuật khác.

Khử nhiễu: Gaussian, median, bilateral, non-local means. Làm nét: unsharp mask. Tham số cụ thể nằm ở configs/restoration.yaml.
Các hàm đều là bộ lọc OpenCV nên giải phóng GIL, chạy song song bằng luồng được.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import cv2
import numpy as np
import torch


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray((image.clamp(0.0, 1.0) * 255).round().byte().permute(1, 2, 0).numpy())


def _from_uint8(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0


def gaussian(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Làm mờ Gauss để giảm nhiễu. sigma tính theo pixel. Giảm nhiễu mạnh nhưng làm mờ cả cạnh và vết bệnh."""
    return _from_uint8(cv2.GaussianBlur(_to_uint8(image), (0, 0), float(sigma)))


def median(image: torch.Tensor, ksize: int) -> torch.Tensor:
    """Lọc trung vị với cửa sổ ksize×ksize (ksize lẻ). Giữ cạnh tốt hơn Gaussian, hợp với nhiễu xung."""
    return _from_uint8(cv2.medianBlur(_to_uint8(image), int(ksize)))


def bilateral(image: torch.Tensor, d: int, sigma_color: float, sigma_space: float) -> torch.Tensor:
    """Lọc song phương: làm mịn các pixel gần nhau về vị trí và về màu, giữ cạnh. Thang màu tính theo 0 đến 255."""
    return _from_uint8(cv2.bilateralFilter(_to_uint8(image), int(d), float(sigma_color), float(sigma_space)))


def nlm(image: torch.Tensor, h: float, template: int = 7, search: int = 21) -> torch.Tensor:
    """Non-local means: trung bình các mảnh ảnh giống nhau trong cửa sổ tìm kiếm. h càng lớn càng mịn, thang 0 đến 255."""
    return _from_uint8(cv2.fastNlMeansDenoisingColored(_to_uint8(image), None, float(h), float(h), int(template), int(search)))


def unsharp(image: torch.Tensor, sigma: float, amount: float) -> torch.Tensor:
    """Unsharp mask: cộng thêm amount lần phần chênh giữa ảnh và bản làm mờ Gauss. Làm cạnh rõ hơn, cũng khuếch đại nhiễu."""
    array = _to_uint8(image).astype(np.float32)
    blurred = cv2.GaussianBlur(array, (0, 0), float(sigma))
    sharpened = np.clip(np.round(array + float(amount) * (array - blurred)), 0, 255).astype(np.uint8)
    return _from_uint8(sharpened)


TECHNIQUES: dict[str, Callable[..., torch.Tensor]] = {
    "gaussian": gaussian,
    "median": median,
    "bilateral": bilateral,
    "nlm": nlm,
    "unsharp": unsharp,
}


def apply_processing(image: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    """Áp một kỹ thuật theo spec, ví dụ {"technique": "nlm", "params": {"h": 12}}."""
    name = str(spec["technique"])
    if name not in TECHNIQUES:
        raise ValueError(f"Kỹ thuật chưa hỗ trợ: {name}.")
    return TECHNIQUES[name](image, **dict(spec.get("params", {})))

"""Các kiểu suy giảm ảnh trên tensor RGB trong khoảng 0 đến 1, dạng C×H×W.

Bốn họ cơ bản của plan: độ sáng, tương phản, mờ, nhiễu. Các kiểu còn lại mô phỏng điều kiện chụp ngoài vườn:
chói nắng, bóng đổ, thiếu sáng, rung tay, camera kém. Kiểu có yếu tố ngẫu nhiên nhận generator để lặp lại được.
"""

from __future__ import annotations

import io
import math
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


def brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Nhân độ sáng với factor. Dưới 1 là tối đi, trên 1 là sáng lên và cháy sáng."""
    return (image * factor).clamp(0.0, 1.0)


def contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Kéo giá trị về (factor dưới 1) hoặc ra xa (factor trên 1) trung bình xám của ảnh."""
    mean = image.mean()
    return ((image - mean) * factor + mean).clamp(0.0, 1.0)


def blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Làm mờ Gauss, giống mất nét. sigma tính theo pixel."""
    kernel = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(image, [kernel, kernel], [sigma, sigma])


def motion_blur(image: torch.Tensor, length: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Mờ theo một hướng, giống rung tay hoặc lá đung đưa. length là độ dài vệt mờ theo pixel, hướng lấy ngẫu nhiên."""
    size = int(length) | 1
    angle = float(torch.rand(1, generator=generator)) * math.pi
    kernel = torch.zeros(size, size)
    center = (size - 1) / 2
    for t in torch.linspace(-center, center, 4 * size):
        row = int(round(center + float(t) * math.sin(angle)))
        col = int(round(center + float(t) * math.cos(angle)))
        kernel[row, col] = 1.0
    kernel = (kernel / kernel.sum()).expand(image.shape[0], 1, size, size)
    padded = F.pad(image[None], [size // 2] * 4, mode="reflect")
    return F.conv2d(padded, kernel, groups=image.shape[0])[0].clamp(0.0, 1.0)


def noise(image: torch.Tensor, std: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Cộng nhiễu Gauss với độ lệch chuẩn std."""
    return (image + _randn(image.shape, generator) * std).clamp(0.0, 1.0)


def sensor_noise(image: torch.Tensor, peak: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Nhiễu cảm biến camera. Nhiễu hạt phụ thuộc cường độ sáng, cộng thêm nhiễu đọc nhỏ. peak càng thấp càng nhiễu."""
    shot = torch.poisson(image * peak, generator=generator) / peak
    return (shot + _randn(image.shape, generator) * 0.01).clamp(0.0, 1.0)


def lowlight(image: torch.Tensor, factor: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Chụp thiếu sáng: ảnh tối đi theo factor và nhiễu hạt tăng theo, vì ít photon hơn."""
    return sensor_noise(image * factor, 200.0, generator)


def glare(image: torch.Tensor, strength: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Vệt chói nắng: một vùng sáng mềm ở vị trí ngẫu nhiên, pha về màu trắng nên mất màu và chi tiết tại đó."""
    _, height, width = image.shape
    cy, cx = (0.15 + 0.7 * torch.rand(2, generator=generator)).tolist()
    ys = torch.linspace(0, 1, height).view(-1, 1)
    xs = torch.linspace(0, 1, width).view(1, -1)
    blob = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 0.18**2))
    weight = (strength * blob).clamp(0.0, 1.0)
    return image * (1 - weight) + weight


def shadow(image: torch.Tensor, strength: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Bóng tán lá đổ lên ảnh: các mảng tối loang lổ, mép mềm. strength là độ tối của mảng bóng."""
    _, height, width = image.shape
    field = _randn((1, 1, 6, 6), generator)
    field = F.interpolate(field, size=(height, width), mode="bilinear", align_corners=False)[0]
    mask = torch.sigmoid(4.0 * (field - field.mean()) / (field.std() + 1e-6))
    return image * (1 - strength * mask)


def jpeg(image: torch.Tensor, quality: float) -> torch.Tensor:
    """Nén JPEG chất lượng thấp, như ảnh gửi qua ứng dụng nhắn tin hoặc camera rẻ."""
    buffer = io.BytesIO()
    TF.to_pil_image(image).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return TF.to_tensor(Image.open(buffer).convert("RGB"))


def lowres(image: torch.Tensor, scale: float) -> torch.Tensor:
    """Giảm độ phân giải xuống scale lần rồi phóng lại về kích thước cũ, như camera độ phân giải thấp hoặc zoom số."""
    _, height, width = image.shape
    small = F.interpolate(image[None], size=(max(1, round(height * scale)), max(1, round(width * scale))), mode="area")
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)[0].clamp(0.0, 1.0)


def color_cast(image: torch.Tensor, strength: float) -> torch.Tensor:
    """Lệch cân bằng trắng sang ám vàng đỏ, như nắng chiều hoặc cân bằng trắng tự động sai."""
    gains = torch.tensor([1 + strength, 1.0, 1 - strength]).view(3, 1, 1)
    return (image * gains).clamp(0.0, 1.0)


def _randn(shape: Any, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(shape, generator=generator)


CORRUPTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "brightness": brightness,
    "contrast": contrast,
    "blur": blur,
    "motion_blur": motion_blur,
    "noise": noise,
    "sensor_noise": sensor_noise,
    "lowlight": lowlight,
    "glare": glare,
    "shadow": shadow,
    "jpeg": jpeg,
    "lowres": lowres,
    "color_cast": color_cast,
}

# Kiểu cần generator để lặp lại được.
RANDOM_TYPES = frozenset({"motion_blur", "noise", "sensor_noise", "lowlight", "glare", "shadow"})


def apply_corruption(
    image: torch.Tensor, spec: Mapping[str, Any], generator: torch.Generator | None = None
) -> torch.Tensor:
    """Áp một suy giảm theo spec, ví dụ {"type": "brightness", "value": 0.5}. Generator chỉ dùng cho kiểu ngẫu nhiên."""
    kind = str(spec["type"])
    if kind not in CORRUPTIONS:
        raise ValueError(f"Kiểu suy giảm chưa hỗ trợ: {kind}.")
    if kind in RANDOM_TYPES:
        return CORRUPTIONS[kind](image, float(spec["value"]), generator)
    return CORRUPTIONS[kind](image, float(spec["value"]))

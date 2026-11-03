"""Sáu họ suy giảm chưa thấy (task 5.5), chỉ dùng để đánh giá, không bao giờ đưa vào augmentation.

Hai nhóm theo khoảng cách tới 14 họ của benchmark 4.1:
- gần (near): cùng cơ chế với một họ đã thấy nhưng khác dạng toán, nên bộ chọn chưa từng được hiệu chỉnh trên chúng.
  `defocus_blur` là mờ theo đĩa (họ đã thấy: mờ Gauss), `chroma_noise` là nhiễu màu loang, chỉ nằm ở kênh màu (họ đã thấy: nhiễu cộng lên cả ba kênh).
- xa (far): cơ chế mới hoàn toàn. `fog` theo mô hình tán xạ khí quyển, `spatter` là giọt nước bám lên ống kính, `rain_streak` là vệt mưa rơi chắn ngang ảnh, `saturate` là màu bị đẩy quá bão hòa.

Các hàm nhận tensor RGB trong khoảng 0 đến 1, dạng C×H×W, giống `src.corruption.basic`. Họ ngẫu nhiên nhận generator để lặp lại được.
Registry nằm tách khỏi `basic.CORRUPTIONS`, nên `RandomCorruption` (augmentation) không gọi được các họ này: nó báo lỗi kiểu chưa hỗ trợ.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

LUMA_WEIGHTS = (0.299, 0.587, 0.114)
# Ánh sáng khí quyển của sương mù: xám sáng, hơi dưới trắng để không cháy sáng cả ảnh.
FOG_AIRLIGHT = 0.85
# Cạnh lưới thô của trường độ sâu giả. Lưới nhỏ cho vùng sương liền khối, không thành nhiễu hạt.
DEPTH_GRID = 4
# Vệt mưa: độ dài (pixel), độ nghiêng tối đa so với phương thẳng đứng (độ), độ đục và màu của vệt.
RAIN_LENGTH = 15
RAIN_MAX_TILT_DEGREES = 20.0
RAIN_OPACITY = 0.6
RAIN_COLOR = 0.9
# Độ mượt (sigma, pixel) của nhiễu màu, để thành đốm nhỏ.
CHROMA_BLUR_SIGMA = 1.0
# Độ mờ (sigma, pixel) của lớp nền dưới giọt nước và độ trong của giọt.
SPATTER_MIX_BLUR = 3.0
SPATTER_GAIN = 1.08


def defocus_blur(image: torch.Tensor, radius: float) -> torch.Tensor:
    """Mất nét theo khẩu độ tròn: tích chập với một đĩa phẳng bán kính `radius` pixel (khác mờ Gauss, mép đĩa sắc)."""
    half = math.ceil(radius)
    coords = torch.arange(-half, half + 1, dtype=image.dtype)
    distance = torch.sqrt(coords.view(-1, 1) ** 2 + coords.view(1, -1) ** 2)
    kernel = (radius + 0.5 - distance).clamp(0.0, 1.0)  # mép đĩa chuyển mượt trong một pixel
    kernel = (kernel / kernel.sum()).expand(image.shape[0], 1, 2 * half + 1, 2 * half + 1)
    padded = F.pad(image[None], [half] * 4, mode="reflect")
    return F.conv2d(padded, kernel, groups=image.shape[0])[0].clamp(0.0, 1.0)


def chroma_noise(image: torch.Tensor, std: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Nhiễu màu loang như camera điện thoại thiếu sáng sau khi khử nhiễu: chỉ nhiễu ở hai kênh màu Cb, Cr, độ sáng giữ nguyên.

    Nhiễu được làm mượt nhẹ nên thành các đốm màu nhỏ thay vì hạt từng pixel. Khác `noise` và `sensor_noise`, đặc trưng nhiễu đo trên độ sáng gần như không thấy nó.
    """
    weights = torch.tensor(LUMA_WEIGHTS, dtype=image.dtype).view(3, 1, 1)
    luma = (image * weights).sum(dim=0, keepdim=True)
    field = TF.gaussian_blur(torch.randn((2, *image.shape[1:]), generator=generator), [5, 5], [CHROMA_BLUR_SIGMA] * 2)
    field = field / field.flatten(1).std(dim=1).clamp_min(1e-6).view(2, 1, 1) * std
    cb, cr = (image[2:3] - luma) / 1.772 + field[0:1], (image[0:1] - luma) / 1.402 + field[1:2]
    red = luma + 1.402 * cr
    blue = luma + 1.772 * cb
    green = (luma - LUMA_WEIGHTS[0] * red - LUMA_WEIGHTS[2] * blue) / LUMA_WEIGHTS[1]
    return torch.cat([red, green, blue]).clamp(0.0, 1.0)


def fog(image: torch.Tensor, density: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sương mù theo mô hình tán xạ khí quyển: I = J·t + A·(1 − t), với t = exp(−density·độ sâu).

    Độ sâu là trường ngẫu nhiên mượt, chuẩn hóa vào [0.5, 1.5], nên sương dày mỏng không đều trên ảnh. Ảnh bị mất tương phản và nhạt màu
    theo vùng, khác giảm tương phản đồng đều của họ `contrast_low`.
    """
    _, height, width = image.shape
    field = F.interpolate(torch.randn((1, 1, DEPTH_GRID, DEPTH_GRID), generator=generator), size=(height, width), mode="bicubic", align_corners=False)[0]
    span = (field.max() - field.min()).clamp_min(1e-6)
    depth = 0.5 + (field - field.min()) / span
    transmission = torch.exp(-density * depth)
    return (image * transmission + FOG_AIRLIGHT * (1.0 - transmission)).clamp(0.0, 1.0)


def spatter(image: torch.Tensor, coverage: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Giọt nước bám ống kính: các đốm tròn che khoảng `coverage` diện tích ảnh, bên trong đốm ảnh bị nhòe và sáng nhẹ."""
    _, height, width = image.shape
    seeds = torch.randn((1, height, width), generator=generator)
    smooth = TF.gaussian_blur(seeds, [9, 9], [2.5, 2.5])[0]
    threshold = torch.quantile(smooth.flatten(), 1.0 - coverage)
    scale = smooth.std().clamp_min(1e-6)
    mask = torch.sigmoid((smooth - threshold) / (0.15 * scale))  # mép mềm
    droplet = (TF.gaussian_blur(image, [19, 19], [SPATTER_MIX_BLUR] * 2) * SPATTER_GAIN).clamp(0.0, 1.0)
    return image * (1.0 - mask) + droplet * mask


def rain_streak(image: torch.Tensor, density: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Mưa rơi: các vệt sáng mảnh gần thẳng đứng, nghiêng nhẹ theo gió, che một phần ảnh.

    `density` là xác suất một pixel là điểm đầu của một vệt dài `RAIN_LENGTH` pixel, nên tỉ lệ phủ xấp xỉ density · RAIN_LENGTH.
    Khác `spatter` (đốm tròn nhòe bám ống kính), đây là cấu trúc đường mảnh có hướng, ảnh bên dưới vẫn nét giữa các vệt.
    """
    _, height, width = image.shape
    tilt = math.radians((2.0 * float(torch.rand(1, generator=generator)) - 1.0) * RAIN_MAX_TILT_DEGREES)
    kernel = torch.zeros(RAIN_LENGTH, RAIN_LENGTH)
    center = (RAIN_LENGTH - 1) / 2
    for t in torch.linspace(-center, center, 4 * RAIN_LENGTH):
        kernel[int(round(center + float(t) * math.cos(tilt))), int(round(center + float(t) * math.sin(tilt)))] = 1.0
    drops = (torch.rand((1, 1, height, width), generator=generator) < density).float()
    mask = F.conv2d(F.pad(drops, [RAIN_LENGTH // 2] * 4), kernel.view(1, 1, RAIN_LENGTH, RAIN_LENGTH))[0].clamp(0.0, 1.0)
    alpha = RAIN_OPACITY * TF.gaussian_blur(mask, [3, 3], [0.7, 0.7])
    return image * (1.0 - alpha) + RAIN_COLOR * alpha


def saturate(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Đẩy màu ra xa mức xám của pixel: gray + factor·(ảnh − gray). factor trên 1 là bão hòa quá mức, màu lá và vết bệnh bị bệt."""
    weights = torch.tensor(LUMA_WEIGHTS, dtype=image.dtype).view(3, 1, 1)
    gray = (image * weights).sum(dim=0, keepdim=True)
    return (gray + factor * (image - gray)).clamp(0.0, 1.0)


UNSEEN_CORRUPTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "defocus_blur": defocus_blur,
    "chroma_noise": chroma_noise,
    "fog": fog,
    "spatter": spatter,
    "rain_streak": rain_streak,
    "saturate": saturate,
}

# Họ cần generator để lặp lại được.
UNSEEN_RANDOM_TYPES = frozenset({"chroma_noise", "fog", "spatter", "rain_streak"})


def apply_unseen(image: torch.Tensor, spec: Mapping[str, object], generator: torch.Generator | None = None) -> torch.Tensor:
    """Áp một họ chưa thấy theo spec, ví dụ {"type": "fog", "value": 1.6}. Generator chỉ dùng cho họ ngẫu nhiên."""
    kind = str(spec["type"])
    if kind not in UNSEEN_CORRUPTIONS:
        raise ValueError(f"Họ suy giảm chưa thấy không hỗ trợ: {kind}.")
    if kind in UNSEEN_RANDOM_TYPES:
        return UNSEEN_CORRUPTIONS[kind](image, float(spec["value"]), generator)
    return UNSEEN_CORRUPTIONS[kind](image, float(spec["value"]))

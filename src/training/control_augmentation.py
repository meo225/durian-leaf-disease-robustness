"""Augmentation cho đối chứng 5.10: TrivialAugment chỉ giữ phép hình học.

TrivialAugment Wide có cả phép màu, độ sáng, tương phản và độ nét (R31). Các phép đó trùng vùng giá trị với vài họ của benchmark
(brightness, contrast, color_cast), nên TrivialAugment nguyên bản không phải đối chứng "tổng quát" sạch (xem docs/task-5.10-ta-geometric-and-leave-group.md).
Bản này chỉ giữ phép hình học và Identity: mỗi ảnh vẫn nhận đúng một phép với cường độ ngẫu nhiên, chỉ bỏ nhóm phép quang học.
"""

from __future__ import annotations

from torch import Tensor
from torchvision import transforms

# Tên phép theo torchvision. Identity giữ lại để mỗi ảnh vẫn có xác suất không bị đổi, như TrivialAugment gốc.
GEOMETRIC_OPS = ("Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate")


class GeometricTrivialAugment(transforms.TrivialAugmentWide):
    """TrivialAugment Wide giới hạn trong `GEOMETRIC_OPS`. Cách chọn phép, cường độ và dấu giữ nguyên như bản gốc."""

    def _augmentation_space(self, num_bins: int) -> dict[str, tuple[Tensor, bool]]:
        full = super()._augmentation_space(num_bins)
        missing = [name for name in GEOMETRIC_OPS if name not in full]
        if missing:
            raise RuntimeError(f"Phiên bản torchvision này không có phép: {missing}")
        return {name: full[name] for name in GEOMETRIC_OPS}


def operation_names(augmentation: transforms.TrivialAugmentWide) -> list[str]:
    """Danh sách phép mà augmentation có thể chọn, theo thứ tự torchvision."""
    return list(augmentation._augmentation_space(augmentation.num_magnitude_bins).keys())


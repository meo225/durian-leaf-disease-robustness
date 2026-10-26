"""Augmentation bằng suy giảm ngẫu nhiên cho đối chứng huấn luyện lại (task 4.5).

Dùng đúng các hàm suy giảm của benchmark 4.1 (apply_corruption), nhưng mỗi lần gọi chọn ngẫu nhiên một điều kiện và một phần ngẫu nhiên mới.
Khác corrupt_tensor ở chỗ không lấy seed từ nội dung ảnh, nên cùng một ảnh nhận suy giảm khác nhau giữa các epoch.
Suy giảm áp lên ảnh 0 đến 1, trước bước chuẩn hóa, rồi làm tròn 8 bit như benchmark.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from src.corruption.basic import apply_corruption

# Cận của giá trị sau khi dao động, để giữ tham số hợp lệ (chất lượng JPEG, độ dài vệt mờ, tỉ lệ độ phân giải, peak nhiễu).
VALUE_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "jpeg": (1.0, 95.0),
    "motion_blur": (2.0, None),
    "lowres": (0.2, 1.0),
    "sensor_noise": (10.0, None),
}


class RandomCorruption:
    """Với xác suất `probability` áp một điều kiện suy giảm chọn đều trong `conditions`, còn lại giữ ảnh nguyên.

    Giá trị của điều kiện được nhân với một hệ số ngẫu nhiên trong khoảng 1 ± jitter, nên mô hình không chỉ thấy đúng ba mức cố định.
    Phần ngẫu nhiên lấy từ bộ sinh số toàn cục của torch. Trong DataLoader, mỗi worker có seed riêng.
    """

    def __init__(
        self,
        conditions: Mapping[str, Mapping[str, Any]],
        probability: float = 0.5,
        jitter: float = 0.1,
        exclude_families: Sequence[str] = (),
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("probability phải trong khoảng 0 đến 1.")
        if not 0.0 <= float(jitter) < 1.0:
            raise ValueError("jitter phải trong khoảng 0 đến dưới 1.")
        excluded = set(exclude_families)
        self.specs = [dict(spec) for spec in conditions.values() if spec["family"] not in excluded]
        if not self.specs:
            raise ValueError("Không còn điều kiện suy giảm nào để áp.")
        self.probability = float(probability)
        self.jitter = float(jitter)

    def families(self) -> list[str]:
        return list(dict.fromkeys(spec["family"] for spec in self.specs))

    def _value(self, spec: Mapping[str, Any]) -> float:
        value = float(spec["value"]) * (1.0 + self.jitter * (2.0 * float(torch.rand(1)) - 1.0))
        low, high = VALUE_BOUNDS.get(str(spec["type"]), (None, None))
        if low is not None:
            value = max(value, low)
        if high is not None:
            value = min(value, high)
        return value

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(1)) >= self.probability:
            return image
        spec = self.specs[int(torch.randint(len(self.specs), (1,)))]
        corrupted = apply_corruption(image, {"type": spec["type"], "value": self._value(spec)})
        return torch.round(corrupted.clamp(0.0, 1.0) * 255) / 255

    def __repr__(self) -> str:
        return f"RandomCorruption(p={self.probability}, jitter={self.jitter}, điều kiện={len(self.specs)}, họ={len(self.families())})"

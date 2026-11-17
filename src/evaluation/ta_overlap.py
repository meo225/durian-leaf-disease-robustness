"""Bảng chồng lấn giữa TrivialAugment (Wide) và các họ suy giảm có hệ số cùng quy ước (task 5.7).

Đối chứng 5.6 gọi TrivialAugment là augmentation "tổng quát", nhưng bốn phép quang học của nó (Brightness, Color, Contrast, Sharpness) lấy hệ số 1 ± m với m tới 0.99, tức hệ số từ 0.01 đến 1.99.
Ba họ dưới đây dùng đúng quy ước hệ số đó (nhân độ sáng, kéo về trung bình xám, trộn với xám theo luma), nên chồng lấn đếm được bằng cách liệt kê:
với mỗi mức của họ, xác suất một lần gọi TA cho ra hệ số trong khoảng ± `tolerance` quanh mức đó.

Chưa tính các phép gần đúng khác quy ước (Sharpness với mờ, Solarize hay Posterize với jpeg), nên bảng này là cận dưới của chồng lấn, không phải toàn bộ.
Bảng thay cho câu "TA có thể trùng họ" trong phân tích bằng một con số kiểm tra được.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.evaluation.msd import trivial_augment_ops

# loại điều kiện của repo -> phép TA có cùng quy ước hệ số
TYPE_TO_TA_OP = {"brightness": "Brightness", "contrast": "Contrast", "saturate": "Color"}


def ta_factors(op_name: str, num_magnitude_bins: int = 31) -> np.ndarray:
    """Mọi hệ số TA có thể cho ra với phép `op_name`, mỗi phần tử có xác suất bằng nhau (dấu và mức chọn đều, kể cả cường độ 0 xuất hiện hai lần theo cách code TA bốc dấu)."""
    magnitudes, signed = trivial_augment_ops(num_magnitude_bins)[op_name]
    if not signed or magnitudes.ndim == 0:
        raise ValueError(f"Phép {op_name} không phải phép hệ số có dấu.")
    m = magnitudes.numpy().astype(float)
    return np.concatenate([1.0 + m, 1.0 - m])


def overlap_row(spec: Mapping[str, Any], tolerance: float = 0.1, num_magnitude_bins: int = 31) -> dict[str, Any]:
    """Một dòng của bảng cho một điều kiện: có nằm trong khoảng hệ số của TA không, và xác suất một lần gọi TA cho hệ số trong ± tolerance quanh mức."""
    op = TYPE_TO_TA_OP[str(spec["type"])]
    factors = ta_factors(op, num_magnitude_bins)
    value = float(spec["value"])
    n_ops = len(trivial_augment_ops(num_magnitude_bins))
    within = np.abs(factors - value) <= tolerance * value
    return {
        "family": spec["family"],
        "severity": spec["severity"],
        "value": value,
        "ta_op": op,
        "ta_factor_range": [float(factors.min()), float(factors.max())],
        "inside_ta_range": bool(factors.min() <= value <= factors.max()),
        "p_ta_within_tolerance": float(within.mean()) / n_ops,
    }


def overlap_table(conditions: Mapping[str, Mapping[str, Any]], tolerance: float = 0.1) -> list[dict[str, Any]]:
    """Bảng chồng lấn cho mọi điều kiện có phép TA cùng quy ước, theo thứ tự điều kiện. Điều kiện khác bị bỏ qua."""
    return [overlap_row(spec, tolerance) for spec in conditions.values() if str(spec["type"]) in TYPE_TO_TA_OP]

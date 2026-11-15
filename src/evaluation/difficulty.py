"""Khớp độ khó của họ chưa thấy theo ΔF1 trên mô hình thay thế (task 5.8).

Vấn đề của bộ v1: mức do nhóm chọn tay và hiệu chỉnh bằng chính Swin-Tiny, nên so họ này với họ kia là so hai độ khó khác nhau. Ở đây, theo Mintun et al. (R33),
mức của họ mới được khớp với độ khó của benchmark đã thấy trên một mô hình thay thế (ConvNeXt-Tiny) trước khi ai nhìn kết quả của các hệ thống cần so:

1. Mục tiêu: các phân vị (mặc định 0.2, 0.5, 0.8) của ΔF1 của mô hình thay thế trên 42 điều kiện của benchmark 4.1, đo trên validation.
2. Với mỗi họ ứng viên và mỗi mục tiêu chọn mức có ΔF1 gần mục tiêu nhất. Sai lệch phải không quá `tolerance` (tuyệt đối, theo ΔF1).
3. Ba mức phải khác nhau, tăng dần theo cường độ, và ΔF1 phải tăng dần. Không thỏa thì bỏ cả họ, ghi lý do. Không nới dung sai và không chọn lại mức sau khi thấy kết quả.

ΔF1 = macro F1 sạch − macro F1 suy giảm. Độ khó thực của Swin-Tiny trên các mức đã chọn được báo riêng (`realised_difficulty`), không dùng để chọn.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def target_levels(seen_degradation: Mapping[str, float], quantiles: Sequence[float]) -> list[float]:
    """Mục tiêu ΔF1 là các phân vị của ΔF1 mô hình thay thế trên các điều kiện đã thấy."""
    values = np.asarray(list(seen_degradation.values()), dtype=float)
    if values.size == 0:
        raise ValueError("Không có điều kiện đã thấy để lấy phân vị.")
    if list(quantiles) != sorted(quantiles) or not all(0.0 < q < 1.0 for q in quantiles):
        raise ValueError("Phân vị phải tăng dần và nằm trong (0, 1).")
    return [float(np.quantile(values, q)) for q in quantiles]


def match_levels(
    candidates: Mapping[float, float],
    targets: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    """Chọn một mức ứng viên cho mỗi mục tiêu. `candidates` là {mức nghiêm trọng: ΔF1 mô hình thay thế}.

    Trả về {"ok": True, "values", "delta_f1", "errors"} hoặc {"ok": False, "reason"}. Chọn lần lượt từng mục tiêu theo thứ tự tăng dần, không dùng lại mức đã chọn.
    """
    if not candidates:
        return {"ok": False, "reason": "không có ứng viên"}
    chosen: list[float] = []
    for target in targets:
        pool = {value: delta for value, delta in candidates.items() if value not in chosen}
        if not pool:
            return {"ok": False, "reason": "hết ứng viên khác nhau"}
        value = min(pool, key=lambda v: (abs(pool[v] - target), v))
        if abs(pool[value] - target) > tolerance:
            best = min(abs(delta - target) for delta in candidates.values())
            return {"ok": False, "reason": f"không mức nào trong ±{tolerance:g} của mục tiêu {target:.3f} (gần nhất lệch {best:.3f})"}
        chosen.append(value)
    deltas = [candidates[v] for v in chosen]
    if chosen != sorted(chosen) or deltas != sorted(deltas):
        return {"ok": False, "reason": "mức đã chọn không tăng đồng thời theo cường độ và theo ΔF1"}
    return {"ok": True, "values": [float(v) for v in chosen], "delta_f1": [float(d) for d in deltas], "errors": [float(d - t) for d, t in zip(deltas, targets)]}


def match_families(
    delta_table: Mapping[str, Mapping[float, float]],
    targets: Sequence[float],
    tolerance: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Khớp mức cho từng họ. Trả về (họ giữ lại với mức đã chọn, họ bị bỏ với lý do), cả hai theo thứ tự tên họ."""
    kept: dict[str, dict[str, Any]] = {}
    dropped: dict[str, str] = {}
    for family in sorted(delta_table):
        result = match_levels(dict(delta_table[family]), targets, tolerance)
        if result["ok"]:
            kept[family] = result
        else:
            dropped[family] = str(result["reason"])
    return kept, dropped

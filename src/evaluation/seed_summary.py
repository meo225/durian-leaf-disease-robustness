"""Tổng hợp nhiều seed cho các đối chứng huấn luyện lại (task 5.9, 5.10).

Quy tắc kết luận khai báo trước: với 3 seed, chỉ kết luận khi chênh lệch trung bình lớn hơn `K` lần độ lệch chuẩn seed.
Độ lệch chuẩn là SD mẫu (ddof = 1) của từng nhánh, hai nhánh gộp theo căn bậc hai của trung bình hai phương sai. Với 3 seed
SD tự nó rất nhiễu, nên ngưỡng này là mức tối thiểu để dám nói, không phải kiểm định có ý nghĩa thống kê.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd

K = 2.5

# So sánh khai báo trước của 5.9 (a trừ b), cùng thước đo: macro F1 trung bình 42 điều kiện đã thấy trên validation.
PAIRS_5_9 = (("trivialaugment", "baseline"), ("degradation", "baseline"), ("degradation", "trivialaugment"))


def condition_objective(frame: pd.DataFrame, families: Sequence[str] | None = None) -> float:
    """Macro F1 trung bình các điều kiện suy giảm (bỏ dòng ảnh sạch). `families` giới hạn vào một số họ."""
    rows = frame[frame["condition"] != "clean"]
    if families is not None:
        rows = rows[rows["family"].isin(list(families))]
    if rows.empty:
        raise ValueError("Không có điều kiện nào để lấy trung bình.")
    return float(rows["macro_f1"].mean())


def clean_f1(frame: pd.DataFrame) -> float:
    rows = frame.loc[frame["condition"] == "clean", "macro_f1"]
    if len(rows) != 1:
        raise ValueError("Cần đúng một dòng ảnh sạch.")
    return float(rows.iloc[0])


def seed_stats(values: Sequence[float]) -> dict[str, float]:
    """Số seed, trung bình, SD mẫu (NaN nếu dưới 2 seed), nhỏ nhất, lớn nhất."""
    data = [float(value) for value in values]
    if not data:
        raise ValueError("Không có giá trị nào.")
    mean = sum(data) / len(data)
    sd = math.sqrt(sum((value - mean) ** 2 for value in data) / (len(data) - 1)) if len(data) > 1 else float("nan")
    return {"n": len(data), "mean": mean, "sd": sd, "min": min(data), "max": max(data)}


def pooled_sd(*sds: float) -> float:
    """Căn bậc hai của trung bình các phương sai. NaN nếu có SD thiếu."""
    if any(math.isnan(sd) for sd in sds):
        return float("nan")
    return math.sqrt(sum(sd * sd for sd in sds) / len(sds))


def decide(delta: float, sd: float, k: float = K) -> str:
    """`vượt` nếu |chênh lệch| lớn hơn k·SD, `chưa kết luận` nếu không, `chưa đủ seed` nếu thiếu SD."""
    if math.isnan(sd):
        return "chưa đủ seed"
    return "vượt" if abs(delta) > k * sd else "chưa kết luận"


def compare_arms(values: Mapping[str, Sequence[float]], pairs: Sequence[tuple[str, str]], k: float = K) -> pd.DataFrame:
    """So từng cặp (a, b) giữa các nhánh nhiều seed: chênh lệch trung bình a − b, SD gộp, tỉ lệ chênh/SD và kết luận."""
    stats = {arm: seed_stats(items) for arm, items in values.items()}
    rows = []
    for a, b in pairs:
        delta = stats[a]["mean"] - stats[b]["mean"]
        sd = pooled_sd(stats[a]["sd"], stats[b]["sd"])
        rows.append({
            "a": a, "b": b, "delta": delta, "pooled_sd": sd,
            "ratio": abs(delta) / sd if sd and not math.isnan(sd) else float("nan"),
            "kết luận": decide(delta, sd, k),
        })
    return pd.DataFrame(rows)


def compare_single_to_arm(value: float, reference_values: Sequence[float], k: float = K) -> dict[str, Any]:
    """So một seed duy nhất (nhánh 5.10) với nhánh tham chiếu nhiều seed. SD lấy từ nhánh tham chiếu, vì một seed không có SD."""
    reference = seed_stats(reference_values)
    delta = float(value) - reference["mean"]
    return {
        "delta": delta, "reference_mean": reference["mean"], "reference_sd": reference["sd"],
        "ratio": abs(delta) / reference["sd"] if reference["sd"] and not math.isnan(reference["sd"]) else float("nan"),
        "kết luận": decide(delta, reference["sd"], k),
    }

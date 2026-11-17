"""Chia họ suy giảm thành train, val, test như chia dữ liệu (task 5.8, theo Mintun et al. R33).

Vai trò:
- train: họ được phép dùng để thiết kế hay huấn luyện (ví dụ làm họ bổ sung cho augmentation hay bộ chọn ở các task sau).
- val: họ để chọn thiết kế, phân tích thăm dò và kiểm tra độ nhạy.
- test: họ chỉ mở một lần ở đánh giá cuối (5.11, 6.1), cần FINAL_FAMILY_EVAL=1 (`family_lock`).

Quy tắc chia, cố định trước khi xem kết quả:
1. Phân tầng theo nhóm MSD (near, far, ambiguous) để mỗi vai trò có cả họ gần lẫn họ xa.
2. Trong mỗi tầng xếp họ theo sha256(`seed:tên họ`), nên thứ tự không phụ thuộc thứ tự khai báo.
3. Số họ mỗi vai trò theo phần trăm đăng ký, làm tròn bằng phương pháp phần dư lớn nhất. Tầng có ít nhất một họ đủ điều kiện mà không có họ test thì chuyển một họ từ vai trò lớn nhất sang test.
4. Họ vai trò test chỉ lấy từ nguồn ngoài (`imagenet_c`, `imagenet_c_bar`), không bao giờ từ họ nhóm tự cài (`source: team`), để họ dùng làm test không phụ thuộc vào cách nhóm hiểu phép suy giảm.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

from src.corruption.external import SOURCES
from src.corruption.family_lock import ROLES

TEAM_SOURCE = "team"


def _order_key(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()


def _counts(n: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Phương pháp phần dư lớn nhất. Hòa thì ưu tiên theo thứ tự test, val, train để kết quả không phụ thuộc thứ tự dict."""
    exact = {role: fractions[role] * n for role in ("test", "val", "train")}
    counts = {role: int(value) for role, value in exact.items()}
    for role in sorted(exact, key=lambda r: (-(exact[r] - counts[r]), ("test", "val", "train").index(r)))[: n - sum(counts.values())]:
        counts[role] += 1
    return counts


def assign_roles(
    families: Mapping[str, Mapping[str, str]],
    seed: int,
    fractions: Mapping[str, float],
) -> dict[str, str]:
    """Vai trò cho từng họ. `families[tên]` cần `group` (near, far, ambiguous) và `source`. Trả về {tên họ: train | val | test}."""
    if set(fractions) != set(ROLES) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions cần đủ train, val, test và có tổng bằng 1.")
    unknown = {spec["source"] for spec in families.values()} - set(SOURCES) - {TEAM_SOURCE}
    if unknown:
        raise ValueError(f"Nguồn họ không nhận ra: {sorted(unknown)}.")
    roles: dict[str, str] = {}
    for group in sorted({spec["group"] for spec in families.values()}):
        members = sorted((n for n, s in families.items() if s["group"] == group), key=lambda n: _order_key(seed, n))
        counts = _counts(len(members), fractions)
        eligible = [n for n in members if families[n]["source"] in SOURCES]
        if eligible and counts["test"] == 0:
            donor = max(("train", "val"), key=lambda r: (counts[r], r == "train"))
            counts[donor] -= 1
            counts["test"] += 1
        test = [n for n in members if n in set(eligible)][: counts["test"]]
        rest = [n for n in members if n not in set(test)]
        for name in test:
            roles[name] = "test"
        n_val = min(counts["val"], len(rest))
        for index, name in enumerate(rest):
            roles[name] = "val" if index < n_val else "train"
    return dict(sorted(roles.items()))

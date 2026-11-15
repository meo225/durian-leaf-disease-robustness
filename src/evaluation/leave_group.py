"""Chọn nhóm họ bỏ cho đối chứng leave-group-out (task 5.10).

Geirhos (R32) bỏ một loại suy giảm khỏi train và thấy loại đó rơi gần mức đoán bừa. Ở đây bỏ cả một nhóm họ cùng cơ chế để họ còn lại
không che phủ nhóm bị bỏ. Nhóm lấy bằng số từ ma trận khoảng cách đặc trưng giữa 14 họ đã thấy (`seen_pairwise_distance` do task 5.7 ghi ra),
không chọn theo tên: họ mỏ neo cộng các họ gần nó nhất.

Quy tắc khai báo trước, trong configs/controls/degradation_leave_group.yaml. Không đổi sau khi thấy kết quả.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_pairwise_matrix(path: str | Path, key: str = "seen_pairwise_distance") -> dict[str, dict[str, float]]:
    """Đọc ma trận khoảng cách từ reports/unseen_v2/msd.json (task 5.7)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có {path}. Task 5.10 chọn nhóm họ bỏ từ kết quả MSD của task 5.7: "
            "merge nhánh unseen-families-v2 (hoặc chép reports/unseen_v2/msd.json vào) rồi chạy lại."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if key not in record:
        raise KeyError(f"{path} không có khóa {key}.")
    return {a: {b: float(value) for b, value in row.items()} for a, row in record[key].items()}


def _check_matrix(matrix: Mapping[str, Mapping[str, float]]) -> list[str]:
    names = sorted(matrix)
    for a in names:
        if sorted(matrix[a]) != names:
            raise ValueError(f"Hàng {a} của ma trận khoảng cách không có đủ cột.")
    for i, a in enumerate(names):
        if abs(matrix[a][a]) > 1e-9:
            raise ValueError(f"Đường chéo của {a} phải bằng 0.")
        for b in names[i + 1 :]:
            if abs(matrix[a][b] - matrix[b][a]) > 1e-6 * max(1.0, abs(matrix[a][b])):
                raise ValueError(f"Ma trận khoảng cách không đối xứng tại ({a}, {b}).")
    return names


def select_leave_group(matrix: Mapping[str, Mapping[str, float]], anchor: str, size: int) -> dict[str, Any]:
    """Họ mỏ neo cộng `size - 1` họ gần nó nhất theo ma trận. Hòa thì xếp theo tên để kết quả xác định.

    Trả về nhóm (mỏ neo đứng đầu), khoảng cách của từng họ tới mỏ neo và hai số mô tả: khoảng cách trung bình trong nhóm so với
    khoảng cách trung bình từ nhóm ra các họ còn lại. Nhóm tách rõ khi số thứ nhất nhỏ hơn số thứ hai.
    """
    names = _check_matrix(matrix)
    if anchor not in matrix:
        raise ValueError(f"Họ mỏ neo {anchor} không có trong ma trận. Hợp lệ: {names}")
    if not 1 <= int(size) < len(names):
        raise ValueError(f"Kích thước nhóm phải từ 1 đến {len(names) - 1}, nhận {size}.")
    others = sorted((family for family in names if family != anchor), key=lambda family: (matrix[anchor][family], family))
    members = [anchor, *others[: int(size) - 1]]
    rest = [family for family in names if family not in members]
    inside = [matrix[a][b] for index, a in enumerate(members) for b in members[index + 1 :]]
    outside = [matrix[a][b] for a in members for b in rest]
    return {
        "anchor": anchor,
        "size": int(size),
        "families": members,
        "distance_to_anchor": {family: matrix[anchor][family] for family in members},
        "mean_distance_inside": sum(inside) / len(inside) if inside else float("nan"),
        "mean_distance_to_rest": sum(outside) / len(outside) if outside else float("nan"),
        "remaining_families": rest,
    }

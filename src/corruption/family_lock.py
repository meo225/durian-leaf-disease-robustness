"""Khóa bộ họ chưa thấy v2 (task 5.7, 5.8): mã băm sha256 của quy tắc đăng ký trước và danh sách họ, cùng cổng mở tập test.

Bộ v2 chỉ có hiệu lực khi `locked: true` và mã băm trong `lock.sha256` khớp nội dung hiện tại. Sửa một mức, một vai trò hay một quy tắc sau khi khóa đổi mã băm,
nên `verify_lock` báo lỗi thay vì để kết quả test dùng cấu hình khác với cấu hình đã khóa. Mở tập test còn cần biến môi trường FINAL_FAMILY_EVAL=1, giống
cách FINAL_EVAL=1 bảo vệ `get_split("test")`. Mỗi lần mở ghi vào manifests/SPLIT_PROTOCOL.md §3.4.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

PROTOCOL = "unseen_families_v2"
FINAL_EVAL_ENV = "FINAL_FAMILY_EVAL"
ROLES = ("train", "val", "test")


def lock_payload(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Phần của config được khóa: giao thức, seed, quy tắc đăng ký trước và danh sách họ cùng mức, vai trò, nhóm."""
    return {
        "protocol": cfg.get("protocol"),
        "version": cfg.get("version"),
        "seed": cfg.get("seed"),
        "registration": cfg.get("registration"),
        "families": cfg.get("families"),
    }


def compute_lock(cfg: Mapping[str, Any]) -> str:
    """sha256 của JSON chuẩn tắc (khóa sắp xếp, không khoảng trắng) của `lock_payload`."""
    blob = json.dumps(lock_payload(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=float)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_v2(cfg: Mapping[str, Any]) -> bool:
    return cfg.get("protocol") == PROTOCOL


def verify_lock(cfg: Mapping[str, Any]) -> None:
    """Config v2 đã khóa phải có mã băm khớp. Config khác giao thức này không bị kiểm tra."""
    if not is_v2(cfg) or not cfg.get("locked"):
        return
    recorded = (cfg.get("lock") or {}).get("sha256")
    if not recorded:
        raise PermissionError("Config v2 đặt locked: true nhưng thiếu lock.sha256.")
    if recorded != compute_lock(cfg):
        raise PermissionError("Config v2 đã bị sửa sau khi khóa: mã sha256 không khớp lock.sha256.")


def guard_split(cfg: Mapping[str, Any], split: str) -> None:
    """Cổng mở tập test của bộ v2: phải khóa, mã băm khớp và có FINAL_FAMILY_EVAL=1. Tập val và train không bị chặn."""
    if not is_v2(cfg) or split != "test":
        return
    if not cfg.get("locked"):
        raise PermissionError("Config v2 chưa khóa (locked: false). Không mở tập test.")
    verify_lock(cfg)
    if os.environ.get(FINAL_EVAL_ENV) != "1":
        raise PermissionError(f"Mở tập test của bộ họ v2 cần {FINAL_EVAL_ENV}=1 và một dòng ghi trong SPLIT_PROTOCOL.md §3.4.")

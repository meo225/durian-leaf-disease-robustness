"""Quản lý và tải các tập dữ liệu phân chia (train, val, test) của DurianLDD."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = _ROOT_DIR / "manifests" / "split_manifest.csv"
META_PATH = _ROOT_DIR / "manifests" / "split_meta.json"
VALID_SPLITS = {"train", "val", "test"}


def _compute_sha256(path: Path) -> str:
    """Tính mã băm SHA256 của tệp."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_split(name: str) -> pd.DataFrame:
    """Đọc dữ liệu của một tập phân chia từ manifest đã khóa.

    Args:
        name: Tên tập phân chia ('train', 'val', hoặc 'test').

    Returns:
        DataFrame chứa các ảnh thuộc tập tương ứng.

    Raises:
        ValueError: Nếu tên tập không hợp lệ hoặc mã SHA256 không khớp.
        PermissionError: Nếu truy cập tập 'test' khi chưa đặt FINAL_EVAL=1.
        FileNotFoundError: Nếu tệp manifest hoặc meta không tồn tại.
    """
    normalized_name = name.strip().lower()
    if normalized_name not in VALID_SPLITS:
        raise ValueError(
            f"Tên tập không hợp lệ: '{name}'. Phải là một trong {sorted(VALID_SPLITS)}."
        )

    if normalized_name == "test" and os.environ.get("FINAL_EVAL") != "1":
        raise PermissionError(
            "Tập test đã bị khóa! Không được phép truy cập tập test trừ khi biến môi trường FINAL_EVAL=1."
        )

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp manifest tại: {MANIFEST_PATH}")
    if not META_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp siêu dữ liệu tại: {META_PATH}")

    with META_PATH.open("r", encoding="utf-8") as handle:
        meta_info = json.load(handle)

    expected_sha256 = meta_info.get("manifest_sha256")
    actual_sha256 = _compute_sha256(MANIFEST_PATH)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Tính toàn vẹn của manifest bị vi phạm! Mã băm SHA256 thực tế ({actual_sha256}) "
            f"không khớp với mã băm lưu trong meta ({expected_sha256})."
        )

    df_manifest = pd.read_csv(MANIFEST_PATH)
    subset = df_manifest[df_manifest["split"] == normalized_name].copy().reset_index(drop=True)
    return subset

"""Đặc trưng chất lượng ảnh không tham chiếu cho bộ chọn xử lý ảnh thích ứng (task 5.1, 5.2).

Mọi đặc trưng tính trên ảnh RGB C×H×W trong khoảng 0 đến 1, đúng ảnh mà reference model sắp nhận, không cần ảnh sạch tương ứng.
Đặc trưng chia thành sáu nhóm để ablation (task 5.4) bỏ từng nhóm:

- exposure: độ sáng và tỉ lệ vùng tối hoặc cháy sáng.
- contrast: độ lệch chuẩn của độ sáng.
- sharpness: độ nét, chuẩn hóa theo độ tương phản nên không phụ thuộc độ sáng hay độ tương phản chung.
- noise: ước lượng nhiễu của Immerkaer và nhiễu so với tín hiệu.
- color: độ lệch đỏ so với xanh dương.
- naturalness: phương sai của hệ số MSCN, đặc trưng nền của BRISQUE (Mittal và cộng sự, 2012). Đây là thống kê tự nhiên của ảnh, không có
  bộ hồi quy đã huấn luyện của BRISQUE, nên không phải điểm BRISQUE đầy đủ.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
EPS = 1e-6
# Ngưỡng cố định của hai tỉ lệ vùng tối và vùng sáng, tính trên độ sáng 0 đến 1.
DARK_LEVEL = 0.1
BRIGHT_LEVEL = 0.9

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "exposure": ("luma_mean", "dark_fraction", "bright_fraction"),
    "contrast": ("luma_std",),
    "sharpness": ("sharpness",),
    "noise": ("noise_sigma", "noise_ratio"),
    "color": ("cast",),
    "naturalness": ("mscn_var",),
}
FEATURES: tuple[str, ...] = tuple(name for names in FEATURE_GROUPS.values() for name in names)

# Nhân của Immerkaer (1996) cho ước lượng độ lệch chuẩn nhiễu Gauss trắng từ một ảnh.
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def group_of(feature: str) -> str:
    """Nhóm chứa một đặc trưng."""
    for group, names in FEATURE_GROUPS.items():
        if feature in names:
            return group
    raise KeyError(f"Đặc trưng chưa hỗ trợ: {feature}.")


def _luma(image: torch.Tensor) -> np.ndarray:
    array = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy().astype(np.float32)
    return array @ LUMA


def _noise_sigma(luma: np.ndarray) -> float:
    """Độ lệch chuẩn nhiễu của độ sáng theo Immerkaer, thang 0 đến 1. Kết cấu mịn của lá cũng đóng góp, nên đây là cận trên của nhiễu."""
    height, width = luma.shape
    response = cv2.filter2D(luma, cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(response).sum() / (6.0 * (height - 2) * (width - 2)))


def _mscn(luma: np.ndarray) -> np.ndarray:
    """Hệ số MSCN (mean subtracted contrast normalized) của BRISQUE: cửa sổ Gauss 7×7, sigma 7/6, hằng số 1, thang 0 đến 255."""
    scaled = luma.astype(np.float64) * 255.0
    mean = cv2.GaussianBlur(scaled, (7, 7), 7.0 / 6.0)
    variance = cv2.GaussianBlur(scaled * scaled, (7, 7), 7.0 / 6.0) - mean * mean
    return (scaled - mean) / (np.sqrt(np.abs(variance)) + 1.0)


def extract_features(image: torch.Tensor) -> dict[str, float]:
    """Đặc trưng chất lượng của một ảnh RGB C×H×W trong khoảng 0 đến 1. Trả về từ điển theo thứ tự `FEATURES`."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Cần ảnh 3×H×W, nhận {tuple(image.shape)}.")
    if min(image.shape[1:]) < 8:
        raise ValueError("Ảnh quá nhỏ để ước lượng đặc trưng (cần tối thiểu 8×8).")
    luma = _luma(image)
    luma_var = float(luma.var())
    luma_std = float(np.sqrt(luma_var))
    laplacian = cv2.Laplacian(luma, cv2.CV_32F, ksize=1)
    noise = _noise_sigma(luma)
    mscn = _mscn(luma)
    channel_mean = image.clamp(0.0, 1.0).mean(dim=(1, 2)).numpy().astype(np.float64)
    values = {
        "luma_mean": float(luma.mean()),
        "dark_fraction": float((luma < DARK_LEVEL).mean()),
        "bright_fraction": float((luma > BRIGHT_LEVEL).mean()),
        "luma_std": luma_std,
        "sharpness": float(laplacian.var()) / (luma_var + EPS),
        "noise_sigma": noise,
        "noise_ratio": noise / (luma_std + EPS),
        "cast": float(np.log((channel_mean[0] + EPS) / (channel_mean[2] + EPS))),
        "mscn_var": float(mscn.var()),
    }
    return {name: values[name] for name in FEATURES}


def extract_batch(images: torch.Tensor, workers: int = 1) -> np.ndarray:
    """Đặc trưng của một lô ảnh uint8 N×3×H×W, trả về mảng float32 N×len(FEATURES) theo thứ tự `FEATURES`.

    Các hàm OpenCV giải phóng GIL nên workers lớn hơn 1 chạy song song bằng luồng, kết quả không đổi.
    """
    if images.dtype != torch.uint8:
        raise TypeError("extract_batch nhận ảnh uint8. Dùng extract_features cho ảnh dạng thực.")

    def one(image: torch.Tensor) -> list[float]:
        values = extract_features(image.float() / 255.0)
        return [values[name] for name in FEATURES]

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, images))
    else:
        rows = [one(image) for image in images]
    return np.asarray(rows, dtype=np.float32)


def feature_index(names: Sequence[str] | None = None) -> dict[str, int]:
    """Vị trí cột của từng đặc trưng trong mảng của `extract_batch`."""
    return {name: FEATURES.index(name) for name in (FEATURES if names is None else names)}


def drop_groups(candidates: Sequence[Mapping[str, str]], dropped: Sequence[str]) -> list[Mapping[str, str]]:
    """Bỏ các ứng viên đặc trưng thuộc nhóm bị loại, dùng cho ablation theo nhóm đặc trưng (task 5.4)."""
    dropped_set = set(dropped)
    unknown = dropped_set - set(FEATURE_GROUPS)
    if unknown:
        raise KeyError(f"Nhóm đặc trưng chưa hỗ trợ: {sorted(unknown)}.")
    return [candidate for candidate in candidates if group_of(candidate["feature"]) not in dropped_set]

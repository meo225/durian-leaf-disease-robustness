"""Họ suy giảm do bên ngoài cài (task 5.8): ImageNet-C qua gói `imagecorruptions` và ImageNet-C-bar của Mintun et al. (R33).

Mục đích: bộ họ chưa thấy v1 (`unseen.py`) do chính nhóm cài, cùng codebase với augmentation. Để họ dùng làm tập test không phụ thuộc vào cách nhóm hiểu
một phép suy giảm, các họ vai trò test chỉ lấy từ hai nguồn này (xem `src.corruption.family_roles`).

- `imagenet_c`: gói `imagecorruptions` 1.1.2 (Apache 2.0), mức nghiêm trọng là số nguyên 1 đến 5.
- `imagenet_c_bar`: mã vendored trong `third_party/imagenet_c_bar` (MIT), mức nghiêm trọng liên tục trên thang 0 đến 10.

Mọi hàm nhận ảnh uint8 H×W×3 và dùng bộ sinh số toàn cục của numpy. Ở đây seed được đặt tạm quanh mỗi lời gọi rồi khôi phục trạng thái cũ, nên cùng ảnh
và cùng seed luôn cho cùng kết quả, không đụng tới các nơi khác dùng np.random.
"""

from __future__ import annotations

import csv
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

EXTERNAL_TYPE = "external"
SOURCE_IMAGENET_C = "imagenet_c"
SOURCE_IMAGENET_C_BAR = "imagenet_c_bar"
SOURCES = (SOURCE_IMAGENET_C, SOURCE_IMAGENET_C_BAR)

# Tên họ trong config có tiền tố nguồn, để không đụng tên với họ của benchmark 4.1 hay bộ v1 (ví dụ defocus_blur).
FAMILY_PREFIX = {SOURCE_IMAGENET_C: "inc_", SOURCE_IMAGENET_C_BAR: "cbar_"}

# 19 họ của `imagecorruptions`. Bản 1.1.2 gọi API đã bị gỡ ở numpy 2 hoặc scikit-image mới (fog, glass_blur, gaussian_blur),
# nên `probe_families` mới là nơi quyết định họ nào dùng được trong môi trường đang chạy.
IMAGENET_C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter", "saturate",
)
IMAGENET_C_SEVERITIES = (1, 2, 3, 4, 5)
# Lưới mức của ImageNet-C-bar: thang 0 đến 10, bước 1. Bản gốc chọn năm mức rời rạc từ thang liên tục này (imagenet_c_bar.csv). Bước 1 thay vì 0.5 để hiệu chỉnh độ khó
# (mỗi ứng viên là một lượt chạy suy luận trên cả validation) không vượt quá vài giờ trên Kaggle.
IMAGENET_C_BAR_GRID = tuple(float(k) for k in range(1, 11))

_BAR_DIR = Path(__file__).resolve().parent / "third_party" / "imagenet_c_bar"


def family_name(source: str, corruption: str) -> str:
    """Tên họ trong config, ví dụ `inc_gaussian_noise` hay `cbar_sparkles`."""
    return FAMILY_PREFIX[source] + corruption


def candidate_values(source: str) -> tuple[float, ...]:
    """Các mức nghiêm trọng được xét khi hiệu chỉnh độ khó (task 5.8)."""
    if source == SOURCE_IMAGENET_C:
        return tuple(float(v) for v in IMAGENET_C_SEVERITIES)
    if source == SOURCE_IMAGENET_C_BAR:
        return IMAGENET_C_BAR_GRID
    raise ValueError(f"Nguồn họ ngoài chưa hỗ trợ: {source}.")


def imagenet_c_bar_reference_levels() -> dict[str, tuple[float, ...]]:
    """Năm mức của bản gốc ImageNet-C-bar cho từng họ, đọc từ `imagenet_c_bar.csv`. Dùng để đối chiếu, không dùng để chọn mức."""
    with (_BAR_DIR / "imagenet_c_bar.csv").open(encoding="utf-8", newline="") as handle:
        return {row[0]: tuple(float(v) for v in row[1:]) for row in csv.reader(handle) if row}


@lru_cache(maxsize=1)
def _bar_families() -> Mapping[str, Any]:
    from src.corruption.third_party.imagenet_c_bar.corrupt import IMAGENET_C_BAR_FAMILIES

    return IMAGENET_C_BAR_FAMILIES


@lru_cache(maxsize=1)
def _imagenet_c_module() -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # pkg_resources deprecated, không liên quan kết quả
            import imagecorruptions
    except ImportError as error:  # pragma: no cover - phụ thuộc môi trường
        raise ImportError(
            "Thiếu gói imagecorruptions. Cài: pip install imagecorruptions==1.1.2 'setuptools<81' (xem requirements.txt)."
        ) from error
    return imagecorruptions


def corruptions_of(source: str) -> tuple[str, ...]:
    """Tên các họ gốc của một nguồn, chưa lọc theo môi trường."""
    if source == SOURCE_IMAGENET_C:
        return IMAGENET_C_CORRUPTIONS
    if source == SOURCE_IMAGENET_C_BAR:
        return tuple(_bar_families())
    raise ValueError(f"Nguồn họ ngoài chưa hỗ trợ: {source}.")


@contextmanager
def _seeded_numpy(seed: int) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


def apply_external_array(image: np.ndarray, source: str, corruption: str, value: float, seed: int) -> np.ndarray:
    """Áp một họ ngoài lên ảnh uint8 H×W×3. Trả về uint8 cùng kích thước."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ảnh phải là mảng uint8 dạng H×W×3.")
    if corruption not in corruptions_of(source):
        raise ValueError(f"Họ {corruption} không có trong nguồn {source}.")
    with _seeded_numpy(seed):
        if source == SOURCE_IMAGENET_C:
            severity = int(round(value))
            if severity != value or severity not in IMAGENET_C_SEVERITIES:
                raise ValueError(f"Mức của ImageNet-C là số nguyên từ 1 đến 5, nhận {value}.")
            out = _imagenet_c_module().corrupt(image, severity=severity, corruption_name=corruption)
        else:
            if image.shape[0] != image.shape[1]:
                raise ValueError("ImageNet-C-bar cần ảnh vuông.")
            if not 0.0 < float(value) <= 10.0:
                raise ValueError(f"Mức của ImageNet-C-bar nằm trong (0, 10], nhận {value}.")
            out = _bar_families()[corruption](float(value), image.shape[0])(image)
    out = np.asarray(out)
    if out.shape != image.shape:
        raise RuntimeError(f"{source}/{corruption} đổi kích thước ảnh {image.shape} thành {out.shape}.")
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_external(image: torch.Tensor, spec: Mapping[str, Any], seed: int) -> torch.Tensor:
    """Áp một điều kiện loại `external` lên ảnh C×H×W trong khoảng 0 đến 1. Kết quả đã làm tròn 8 bit."""
    array = torch.round(image.clamp(0.0, 1.0) * 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    out = apply_external_array(array, str(spec["source"]), str(spec["corruption"]), float(spec["value"]), seed)
    return torch.from_numpy(out).permute(2, 0, 1).float() / 255.0


def probe_families(size: int = 224, seed: int = 0, repeats: int = 2) -> dict[str, dict[str, Any]]:
    """Chạy thử từng họ của cả hai nguồn trên một ảnh tổng hợp. Trả về {tên họ: {"ok", "error", "seconds_per_image"}}.

    Cần chạy một lần trong môi trường thật (notebook) trước khi hiệu chỉnh. Họ lỗi bị loại và ghi lại, không được vá kết quả. Thời gian mỗi ảnh (trung bình `repeats` lần,
    ở mức giữa của nguồn) dùng cho quy tắc loại họ quá chậm ở `unseen_families_v2.eligible_candidates`.
    """
    rng = np.random.RandomState(seed)
    image = (rng.rand(size, size, 3) * 255).astype(np.uint8)
    middle = {SOURCE_IMAGENET_C: 3.0, SOURCE_IMAGENET_C_BAR: 5.0}
    status: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        for corruption in corruptions_of(source):
            name = family_name(source, corruption)
            try:
                start = time.perf_counter()
                for repeat in range(repeats):
                    apply_external_array(image, source, corruption, middle[source], seed + repeat)
                status[name] = {"ok": True, "error": None, "seconds_per_image": (time.perf_counter() - start) / repeats}
            except Exception as error:  # noqa: BLE001 - ghi lại mọi lỗi để loại họ
                status[name] = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:120]}", "seconds_per_image": float("nan")}
    return status


def split_family_name(name: str) -> tuple[str, str]:
    """Từ tên họ trong config về (nguồn, họ gốc). Ví dụ `cbar_sparkles` thành (`imagenet_c_bar`, `sparkles`)."""
    for source, prefix in FAMILY_PREFIX.items():
        if name.startswith(prefix):
            return source, name[len(prefix):]
    raise ValueError(f"Không nhận ra nguồn của họ {name}.")

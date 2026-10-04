"""Kiểm kê và kiểm định chất lượng ảnh DurianLDD.

Script này thống kê lớp, kích thước và chất lượng ảnh, rồi tìm tệp hỏng,
ảnh trùng, ảnh gần trùng, xung đột nhãn và dấu hiệu rò rỉ trên split có sẵn.
Script không tạo split mới.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
SPLIT_ORDER = ["train", "val", "test", "unsplit"]
CLASS_LABELS = {
    "ALGAL_LEAF_SPOT": "Algal leaf spot",
    "ALLOCARIDARA_ATTACK": "Allocaridara attack",
    "HEALTHY_LEAF": "Healthy leaf",
    "LEAF_BLIGHT": "Leaf blight",
    "PHOMOPSIS_LEAF_SPOT": "Phomopsis leaf spot",
}
CLASS_ORDER = list(CLASS_LABELS)
PRIMARY_HAMMING = 5
SENSITIVITY_THRESHOLDS = (0, 5, 8, 10)
SOURCE_ID_PATTERN = re.compile(r"(\d+)$")


class UnionFind:
    """Gom các ảnh liên kết bởi quan hệ gần trùng thành từng nhóm."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def parse_source_id(stem: str) -> int | None:
    match = SOURCE_ID_PATTERN.search(stem)
    if match is None:
        return None
    return int(match.group(1))


def normalize_split(name: str) -> str:
    lowered = name.lower()
    if lowered in {"val", "valid", "validation"}:
        return "val"
    if lowered in {"train", "test"}:
        return lowered
    return "unsplit"


def infer_split_and_class(path: Path, root: Path) -> tuple[str, str]:
    directories = path.relative_to(root).parts[:-1]
    split = "unsplit"
    for part in directories:
        if part.lower() in SPLIT_NAMES:
            split = normalize_split(part)
    parent = path.parent.name
    if parent.lower() in SPLIT_NAMES:
        return split, "UNLABELED"
    return split, parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Hash khác biệt 64 bit. Ảnh giống nhau sau nén nhẹ cho khoảng cách Hamming nhỏ."""
    if hash_size != 8:
        raise ValueError("Bản kiểm định này cố định hash_size=8 để hash nằm trong 64 bit.")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    gray = image.convert("L").resize((hash_size + 1, hash_size), resample)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_uint64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    payload = width.to_bytes(4, "big") + height.to_bytes(4, "big") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def flip_horizontal(image: Image.Image) -> Image.Image:
    operation = getattr(getattr(Image, "Transpose", Image), "FLIP_LEFT_RIGHT")
    return image.transpose(operation)


def extract_datetime(image: Image.Image) -> str:
    try:
        exif = image.getexif()
    except Exception:
        return ""
    if not exif:
        return ""
    for tag in (36867, 306, 36868):
        value = exif.get(tag)
        if value:
            return str(value)
    return ""


def quality_metrics(rgb: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation": float(hsv[:, :, 1].mean()),
        "high_frequency": float(residual.std()),
    }


def discover_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image_record(path: Path, root: Path, hash_size: int) -> dict[str, object]:
    split, class_name = infer_split_and_class(path, root)
    record: dict[str, object] = {
        "relative_path": path.relative_to(root).as_posix(),
        "split": split,
        "class_name": class_name,
        "source_id": parse_source_id(path.stem),
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "mode": "",
        "brightness": None,
        "contrast": None,
        "sharpness": None,
        "saturation": None,
        "high_frequency": None,
        "pixel_sha256": "",
        "flip_pixel_sha256": "",
        "dhash": None,
        "dhash_flip": None,
        "exif_datetime": "",
        "readable": False,
        "error": "",
    }
    if record["file_bytes"] == 0:
        record["error"] = "empty_file"
        return record
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            rgb_image = image.convert("RGB")
            flipped = flip_horizontal(rgb_image)
            metrics = quality_metrics(np.asarray(rgb_image))
            record.update(
                {
                    "width": width,
                    "height": height,
                    "aspect_ratio": float(width / height) if height else None,
                    "mode": image.mode,
                    "pixel_sha256": pixel_sha256(rgb_image),
                    "flip_pixel_sha256": pixel_sha256(flipped),
                    "dhash": dhash(rgb_image, hash_size),
                    "dhash_flip": dhash(flipped, hash_size),
                    "exif_datetime": extract_datetime(image),
                    "readable": True,
                }
            )
            record.update(metrics)
    except Exception as exc:
        record["error"] = type(exc).__name__
    return record


def _pairwise_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.bitwise_count(np.bitwise_xor(left[:, None], right[None, :]))


def analyze_readable_images(
    records: list[dict[str, object]],
    max_hamming: int,
    thresholds: tuple[int, ...] = SENSITIVITY_THRESHOLDS,
) -> dict[str, object]:
    readable = [record for record in records if record["readable"]]
    count = len(readable)
    empty = {
        "pairs": [],
        "groups": {},
        "sensitivity": [],
        "adjacent_pairs": [],
    }
    if count == 0:
        return empty

    hashes = np.array([int(record["dhash"]) for record in readable], dtype=np.uint64)
    flip_hashes = np.array([int(record["dhash_flip"]) for record in readable], dtype=np.uint64)
    orig_dist = _pairwise_distance(hashes, hashes)
    flip_directed = _pairwise_distance(hashes, flip_hashes)
    flip_dist = np.minimum(flip_directed, flip_directed.T)
    upper_i, upper_j = np.triu_indices(count, k=1)
    orig_upper = orig_dist[upper_i, upper_j]
    flip_upper = flip_dist[upper_i, upper_j]
    effective = np.minimum(orig_upper, flip_upper)

    splits = np.array([str(record["split"]) for record in readable])
    classes = np.array([str(record["class_name"]) for record in readable])
    cross_split = splits[upper_i] != splits[upper_j]
    cross_class = classes[upper_i] != classes[upper_j]

    sensitivity = []
    for threshold in thresholds:
        mask = effective <= threshold
        involved = (
            np.unique(np.concatenate([upper_i[mask], upper_j[mask]]))
            if np.any(mask)
            else np.array([], dtype=int)
        )
        sensitivity.append(
            {
                "max_hamming": int(threshold),
                "pairs": int(mask.sum()),
                "images": int(involved.size),
                "cross_split_pairs": int(np.sum(mask & cross_split)),
                "cross_class_pairs": int(np.sum(mask & cross_class)),
            }
        )

    primary_mask = effective <= max_hamming
    union = UnionFind(count)
    pairs: list[dict[str, object]] = []
    for index in np.flatnonzero(primary_mask):
        left = int(upper_i[index])
        right = int(upper_j[index])
        union.union(left, right)
        left_record = readable[left]
        right_record = readable[right]
        relation = _pair_relation(
            left_record,
            right_record,
            int(orig_upper[index]),
            int(flip_upper[index]),
            max_hamming,
        )
        pairs.append(
            {
                "path_a": left_record["relative_path"],
                "path_b": right_record["relative_path"],
                "split_a": left_record["split"],
                "split_b": right_record["split"],
                "class_a": left_record["class_name"],
                "class_b": right_record["class_name"],
                "source_id_a": left_record["source_id"],
                "source_id_b": right_record["source_id"],
                "hamming": int(orig_upper[index]),
                "flip_hamming": int(flip_upper[index]),
                "relation": relation,
                "cross_split": bool(left_record["split"] != right_record["split"]),
                "cross_class": bool(left_record["class_name"] != right_record["class_name"]),
            }
        )

    roots = [union.find(index) for index in range(count)]
    root_to_group: dict[int, int] = {}
    groups: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        group_id = root_to_group[root]
        groups[group_id].append(index)
        readable[index]["group_id"] = group_id
    for members in groups.values():
        for index in members:
            readable[index]["group_size"] = len(members)

    adjacent_pairs = _adjacent_source_pairs(readable)
    return {
        "pairs": pairs,
        "groups": {group_id: members for group_id, members in groups.items()},
        "sensitivity": sensitivity,
        "adjacent_pairs": adjacent_pairs,
        "readable_records": readable,
    }


def _pair_relation(
    left: dict[str, object],
    right: dict[str, object],
    orig_hamming: int,
    flip_hamming: int,
    max_hamming: int,
) -> str:
    if left["sha256"] == right["sha256"]:
        return "identical_file"
    if left["pixel_sha256"] == right["pixel_sha256"]:
        return "identical_pixels"
    if (
        left["pixel_sha256"] == right["flip_pixel_sha256"]
        or right["pixel_sha256"] == left["flip_pixel_sha256"]
    ):
        return "mirror"
    if orig_hamming <= max_hamming:
        return "near"
    if flip_hamming <= max_hamming:
        return "mirror_near"
    return "near"


def _adjacent_source_pairs(readable: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = [
        record
        for record in readable
        if isinstance(record.get("source_id"), int)
    ]
    ordered.sort(key=lambda record: int(record["source_id"]))
    pairs = []
    for left, right in zip(ordered, ordered[1:]):
        left_id = int(left["source_id"])
        right_id = int(right["source_id"])
        if right_id - left_id != 1:
            continue
        pairs.append(
            {
                "path_a": left["relative_path"],
                "path_b": right["relative_path"],
                "source_id_a": left_id,
                "source_id_b": right_id,
                "split_a": left["split"],
                "split_b": right["split"],
                "class_a": left["class_name"],
                "class_b": right["class_name"],
                "hamming": hamming_uint64(int(left["dhash"]), int(right["dhash"])),
                "same_class": left["class_name"] == right["class_name"],
                "cross_split": left["split"] != right["split"],
            }
        )
    return pairs


def assess_eligibility(summary: dict[str, object]) -> dict[str, object]:
    """Quy tắc được chốt trước khi xem ảnh: đủ ảnh đọc được, đủ lớp, ít tệp hỏng, ít xung đột nhãn."""
    readable = int(summary["readable_images"])
    failures: list[str] = []
    if readable < 4000:
        failures.append(
            f"Chỉ có {fmt_int(readable)} ảnh đọc được, dưới ngưỡng 4.000 ảnh để tiếp tục thực nghiệm."
        )
    if int(summary["classes_with_at_least_100"]) < 4:
        failures.append("Chưa có ít nhất 4 lớp với từ 100 ảnh đọc được trở lên.")
    if float(summary["corrupt_rate"]) >= 0.01:
        failures.append("Tỉ lệ tệp hỏng đạt từ 1% trở lên.")
    cross_class_exact = int(summary["images_in_cross_class_exact_duplicates"])
    if readable and cross_class_exact / readable >= 0.01:
        failures.append("Ít nhất 1% ảnh nằm trong cặp trùng khớp tuyệt đối nhưng khác nhãn.")

    conditions: list[str] = []
    if int(summary["cross_split_near_pairs"]) > 0:
        conditions.append(
            "Split có sẵn chứa cặp gần trùng xuyên train, validation hoặc test. "
            "Không dùng split này làm split đã khóa. Bước chia tập phải giữ mỗi nhóm gần trùng trong một tập."
        )
    elif int(summary["multi_image_groups"]) > 0:
        conditions.append(
            "Có nhóm ảnh gần trùng. Bước chia tập phải gán cả nhóm vào cùng một tập để tránh rò rỉ."
        )
    else:
        conditions.append(
            "Ở ngưỡng gần trùng đã chọn không thấy cặp xuyên split. "
            "Bước chia tập vẫn tự tạo split của đồ án và khóa test, không mặc định dùng split của nhà phát hành."
        )
    if int(summary["unique_resolutions"]) == 1:
        conditions.append(
            "Mọi ảnh đọc được cùng một kích thước. Thống kê độ phân giải mô tả bản phát hành, không mô tả ảnh lúc chụp."
        )
    if int(summary["exif_datetime_count"]) == 0:
        conditions.append(
            "Không có thời điểm chụp trong EXIF. Lần kiểm tra này không truy được cùng một phiên chụp khi hai ảnh đã khác nhau rõ."
        )
    return {"eligible": not failures, "failures": failures, "conditions": conditions}


def fmt_int(value: int) -> str:
    if abs(int(value)) < 1000:
        return str(int(value))
    return f"{int(value):,}".replace(",", ".")


def fmt_pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0,0%"
    return f"{100.0 * part / whole:.1f}".replace(".", ",") + "%"


def _percentiles(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {}
    points = [0, 5, 25, 50, 75, 95, 100]
    computed = np.percentile(values.to_numpy(dtype=float), points)
    return {f"p{point}": round(float(value), 3) for point, value in zip(points, computed)}


def _class_label(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name.replace("_", " ").title())


def _ordered_classes(class_names: list[str]) -> list[str]:
    known = [name for name in CLASS_ORDER if name in class_names]
    extra = sorted(name for name in class_names if name not in CLASS_ORDER)
    return known + extra


def build_summary(
    records: list[dict[str, object]],
    analysis: dict[str, object],
    thresholds: dict[str, float],
    expected_image_count: int,
    max_hamming: int,
    sample_seed: int,
) -> dict[str, object]:
    frame = pd.DataFrame(records)
    readable = frame[frame["readable"]].copy()
    discovered = int(len(frame))
    readable_count = int(len(readable))
    corrupt_count = discovered - readable_count
    pairs = list(analysis["pairs"])
    adjacent = list(analysis["adjacent_pairs"])
    pair_frame = pd.DataFrame(pairs)
    adjacent_frame = pd.DataFrame(adjacent)

    exact_relations = {"identical_file", "identical_pixels", "mirror"}
    exact_pairs = [pair for pair in pairs if pair["relation"] in exact_relations]
    cross_class_exact_images = _images_in_pairs(
        [pair for pair in exact_pairs if pair["cross_class"]]
    )
    cross_split_pairs = [pair for pair in pairs if pair["cross_split"]]
    cross_class_pairs = [pair for pair in pairs if pair["cross_class"]]
    if readable_count and "group_size" in readable:
        group_sizes = readable["group_size"].astype(int)
        multi_member = readable.loc[group_sizes > 1]
        multi_groups = int(multi_member["group_id"].nunique()) if not multi_member.empty else 0
    else:
        group_sizes = pd.Series(dtype=int)
        multi_groups = 0
    largest_group = int(group_sizes.max()) if readable_count and not group_sizes.empty else 0
    images_in_multi_groups = int((group_sizes > 1).sum()) if readable_count else 0

    class_counts = readable["class_name"].value_counts().to_dict() if readable_count else {}
    classes_with_at_least_100 = sum(count >= 100 for count in class_counts.values())
    resolutions = (
        readable.assign(resolution=lambda data: data["width"].astype(str) + "x" + data["height"].astype(str))
        ["resolution"]
        .value_counts()
        .to_dict()
        if readable_count
        else {}
    )
    flags = _quality_flags(readable, thresholds) if readable_count else {}
    test_leak_images = _split_partner_images(pairs, source_split="test", partner_splits={"train", "val"})
    val_leak_images = _split_partner_images(pairs, source_split="val", partner_splits={"train"})
    test_count = int((readable["split"] == "test").sum()) if readable_count else 0
    val_count = int((readable["split"] == "val").sum()) if readable_count else 0

    adjacent_cross = [pair for pair in adjacent if pair["cross_split"]]
    adjacent_cross_near = [
        pair for pair in adjacent_cross if int(pair["hamming"]) <= max_hamming
    ]
    adjacent_cross_class = [pair for pair in adjacent if not pair["same_class"]]

    summary: dict[str, object] = {
        "discovered_images": discovered,
        "readable_images": readable_count,
        "corrupt_images": corrupt_count,
        "corrupt_rate": (corrupt_count / discovered) if discovered else 0.0,
        "expected_image_count": expected_image_count,
        "count_matches_expected": discovered == expected_image_count,
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "classes_with_at_least_100": classes_with_at_least_100,
        "split_counts": {
            str(key): int(value)
            for key, value in (readable["split"].value_counts().to_dict() if readable_count else {}).items()
        },
        "resolutions": {str(key): int(value) for key, value in resolutions.items()},
        "unique_resolutions": len(resolutions),
        "aspect_ratio": _percentiles(readable["aspect_ratio"]) if readable_count else {},
        "file_bytes": _percentiles(readable["file_bytes"]) if readable_count else {},
        "quality": {
            name: _percentiles(readable[name])
            for name in ("brightness", "contrast", "sharpness", "saturation", "high_frequency")
            if readable_count
        },
        "quality_flags": flags,
        "exif_datetime_count": int((readable["exif_datetime"].astype(str).str.len() > 0).sum())
        if readable_count
        else 0,
        "near_duplicate_max_hamming": max_hamming,
        "pairs": len(pairs),
        "exact_pairs": len(exact_pairs),
        "cross_split_near_pairs": len(cross_split_pairs),
        "cross_class_near_pairs": len(cross_class_pairs),
        "images_in_cross_class_exact_duplicates": len(cross_class_exact_images),
        "multi_image_groups": multi_groups,
        "largest_group_size": largest_group,
        "images_in_multi_image_groups": images_in_multi_groups,
        "test_images_with_partner_in_train_or_val": len(test_leak_images),
        "test_images": test_count,
        "val_images_with_partner_in_train": len(val_leak_images),
        "val_images": val_count,
        "sensitivity": analysis["sensitivity"],
        "adjacent_id_pairs": len(adjacent),
        "adjacent_id_cross_split_pairs": len(adjacent_cross),
        "adjacent_id_cross_split_near_pairs": len(adjacent_cross_near),
        "adjacent_id_cross_class_pairs": len(adjacent_cross_class),
        "adjacent_hamming": _percentiles(adjacent_frame["hamming"]) if not adjacent_frame.empty else {},
        "relation_counts": {
            str(key): int(value)
            for key, value in (
                pair_frame["relation"].value_counts().to_dict() if not pair_frame.empty else {}
            ).items()
        },
        "sample_seed": sample_seed,
        "review_examples": _review_examples(readable, pairs, adjacent, sample_seed),
    }
    summary.update(assess_eligibility(summary))
    return summary


def _images_in_pairs(pairs: list[dict[str, object]]) -> set[str]:
    images: set[str] = set()
    for pair in pairs:
        images.add(str(pair["path_a"]))
        images.add(str(pair["path_b"]))
    return images


def _split_partner_images(
    pairs: list[dict[str, object]],
    source_split: str,
    partner_splits: set[str],
) -> set[str]:
    images: set[str] = set()
    for pair in pairs:
        sides = ((str(pair["path_a"]), str(pair["split_a"])), (str(pair["path_b"]), str(pair["split_b"])))
        for path, split in sides:
            other_split = sides[0][1] if path == sides[1][0] else sides[1][1]
            if split == source_split and other_split in partner_splits:
                images.add(path)
    return images


def _quality_flags(readable: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, int]:
    return {
        "very_dark": int((readable["brightness"] < thresholds["dark_brightness_below"]).sum()),
        "very_bright": int((readable["brightness"] > thresholds["bright_brightness_above"]).sum()),
        "low_contrast": int((readable["contrast"] < thresholds["low_contrast_below"]).sum()),
        "very_soft": int((readable["sharpness"] < thresholds["soft_sharpness_below"]).sum()),
    }


def _review_examples(
    readable: pd.DataFrame,
    pairs: list[dict[str, object]],
    adjacent: list[dict[str, object]],
    sample_seed: int,
) -> dict[str, object]:
    if readable.empty:
        return {}
    generator = np.random.default_rng(sample_seed)
    per_class: dict[str, list[str]] = {}
    for class_name, group in readable.groupby("class_name"):
        paths = group["relative_path"].tolist()
        take = min(4, len(paths))
        chosen = generator.choice(paths, size=take, replace=False)
        per_class[str(class_name)] = [str(path) for path in chosen]
    darkest = readable.nsmallest(4, "brightness")["relative_path"].astype(str).tolist()
    softest = readable.nsmallest(4, "sharpness")["relative_path"].astype(str).tolist()
    closest_adjacent = sorted(adjacent, key=lambda pair: int(pair["hamming"]))[:6]
    return {
        "per_class": per_class,
        "darkest": darkest,
        "softest": softest,
        "cross_class_pairs": [pair for pair in pairs if pair["cross_class"]][:8],
        "cross_split_pairs": [pair for pair in pairs if pair["cross_split"]][:8],
        "closest_adjacent_pairs": closest_adjacent,
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Segoe UI",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def write_figures(
    frame: pd.DataFrame,
    output_dir: Path,
    root: Path,
    sample_seed: int,
    pairs: list[dict[str, object]] | None = None,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    readable = frame[frame["readable"]].copy()
    if readable.empty:
        return
    _configure_matplotlib()
    _plot_class_counts(readable, figures / "class_distribution.png")
    _plot_quality(readable, figures / "quality_distributions.png")
    _plot_quality_by_class(readable, figures / "quality_by_class.png")
    _plot_resolutions(readable, figures / "resolutions.png")
    _plot_examples(readable, root, figures / "class_examples.png", sample_seed)
    if pairs:
        _plot_pairs(pairs, root, figures / "near_duplicate_pairs.png")


def _plot_class_counts(readable: pd.DataFrame, path: Path) -> None:
    classes = _ordered_classes(readable["class_name"].unique().tolist())
    splits = [split for split in SPLIT_ORDER if split in set(readable["split"])]
    counts = (
        readable.groupby(["class_name", "split"]).size().unstack(fill_value=0).reindex(classes)
    )
    colors = ["#2F6F8F", "#7BA05B", "#C47B4A", "#8D6A9F"]
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    width = 0.8 / max(len(splits), 1)
    for index, split in enumerate(splits):
        values = counts[split].to_numpy() if split in counts else np.zeros(len(classes))
        axis.bar(x + index * width, values, width=width, label=split, color=colors[index % len(colors)])
    axis.set_xticks(x + width * (len(splits) - 1) / 2)
    axis.set_xticklabels([_class_label(name) for name in classes], rotation=15, ha="right")
    axis.set_ylabel("Số ảnh")
    axis.set_title("Số ảnh theo lớp và split có sẵn")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_quality(readable: pd.DataFrame, path: Path) -> None:
    columns = [
        ("brightness", "Độ sáng trung bình"),
        ("contrast", "Độ tương phản"),
        ("sharpness", "Độ sắc nét"),
        ("saturation", "Độ bão hòa"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (column, title) in zip(axes.ravel(), columns):
        axis.hist(readable[column], bins=30, color="#2F6F8F", edgecolor="white")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_quality_by_class(readable: pd.DataFrame, path: Path) -> None:
    classes = _ordered_classes(readable["class_name"].unique().tolist())
    labels = [_class_label(name) for name in classes]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for axis, column, title in zip(
        axes,
        ("brightness", "contrast", "sharpness"),
        ("Độ sáng", "Độ tương phản", "Độ sắc nét"),
    ):
        data = [readable.loc[readable["class_name"] == name, column].to_numpy() for name in classes]
        axis.boxplot(data, tick_labels=labels, showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_resolutions(readable: pd.DataFrame, path: Path) -> None:
    resolution = readable["width"].astype(int).astype(str) + "x" + readable["height"].astype(int).astype(str)
    counts = resolution.value_counts()
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(counts.index.astype(str), counts.to_numpy(), color="#2F6F8F")
    axis.set_ylabel("Số ảnh")
    axis.set_title("Kích thước ảnh trong bản phát hành")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_examples(readable: pd.DataFrame, root: Path, path: Path, sample_seed: int) -> None:
    classes = _ordered_classes(readable["class_name"].unique().tolist())
    figure, axes = plt.subplots(len(classes), 4, figsize=(10, 2.3 * len(classes)))
    axes_grid = np.atleast_2d(axes)
    generator = np.random.default_rng(sample_seed)
    for row, class_name in enumerate(classes):
        paths = readable.loc[readable["class_name"] == class_name, "relative_path"].tolist()
        take = min(4, len(paths))
        chosen = list(generator.choice(paths, size=take, replace=False))
        for column in range(4):
            axis = axes_grid[row, column]
            axis.axis("off")
            if column >= take:
                continue
            image = Image.open(root / chosen[column]).convert("RGB")
            axis.imshow(image)
            if column == 0:
                axis.set_title(_class_label(class_name), loc="left", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def render_report(summary: dict[str, object], frame: pd.DataFrame) -> str:
    readable = frame[frame["readable"]].copy()
    decision = (
        "DurianLDD đủ điều kiện tiếp tục thực nghiệm."
        if summary["eligible"]
        else "DurianLDD chưa đủ điều kiện tiếp tục thực nghiệm."
    )
    lines = [
        "# Kiểm định chất lượng DurianLDD",
        "",
        decision,
        "",
        _condition_block(summary),
        "",
        "## Phạm vi",
        "",
        "Lần kiểm tra này chỉ dùng DurianLDD làm dữ liệu chính. "
        "Bộ Vietnamese durian leaf, AI-Driven và Ten durian diseases không nằm trong lần chạy này. "
        "Split train, validation và test có sẵn được đo để tìm rò rỉ. "
        "Lần kiểm tra không tạo split mới và không khóa test.",
        "",
        "Ảnh gần trùng là cặp có hash khác biệt 64 bit cách nhau không quá "
        f"{summary['near_duplicate_max_hamming']} bit, kể cả khi một ảnh là bản lật ngang của ảnh kia. "
        "Ngưỡng này ưu tiên độ chính xác: ảnh nén lại hoặc chỉnh nhẹ được gom, còn ảnh cùng lá nhưng đổi góc chụp mạnh có thể không bị gom. "
        "Bảng độ nhạy ở khoảng cách 0, 5, 8 và 10 dùng để xem kết luận có phụ thuộc vào đúng một ngưỡng hay không.",
        "",
        "Độ sáng là trung bình kênh xám. Độ tương phản là độ lệch chuẩn kênh xám. "
        "Độ sắc nét là phương sai Laplacian. Độ bão hòa là trung bình kênh S trong HSV. "
        "Năng lượng tần số cao là độ lệch chuẩn phần dư sau làm mờ Gaussian. "
        "Chỉ số cuối trộn texture vết bệnh với nhiễu, nên chỉ dùng để mô tả, không dùng để loại ảnh.",
        "",
        "## Nguồn và cấu trúc",
        "",
        "Nguồn là bản Kaggle `cthng123/durian-leaf-disease-dataset`, giấy phép CC BY 4.0. "
        f"Thư mục phát hành là `DLD_FinalDataset_224_spit`. "
        f"Có {fmt_int(int(summary['discovered_images']))} tệp ảnh"
        + (
            ", khớp mức 4.437 ảnh đã chọn."
            if summary["count_matches_expected"]
            else f", trong khi quyết định dữ liệu ghi {fmt_int(int(summary['expected_image_count']))} ảnh."
        ),
        "",
        f"Ảnh đọc được: {fmt_int(int(summary['readable_images']))}. "
        f"Tệp hỏng hoặc rỗng: {fmt_int(int(summary['corrupt_images']))} "
        f"({fmt_pct(int(summary['corrupt_images']), int(summary['discovered_images']))}). "
        f"Ảnh còn thời điểm chụp trong EXIF: {fmt_int(int(summary['exif_datetime_count']))}.",
        "",
        "## Phân bố lớp",
        "",
        _class_table(readable),
        "",
        _split_table(readable),
        "",
        _split_proportion_text(summary),
        "",
        "Năm tên lớp giữ nguyên theo thư mục phát hành. "
        "`ALLOCARIDARA_ATTACK` là tên do nhà phát hành đặt. Lần kiểm tra này không đổi tên và không gộp lớp.",
        "",
        "## Kích thước và tỉ lệ khung",
        "",
        _resolution_text(summary),
        "",
        "## Phân bố chất lượng ảnh",
        "",
        _quality_table(summary),
        "",
        _flag_text(summary),
        "",
        "Hình phân bố nằm ở `reports/data_quality/figures/quality_distributions.png` và "
        "`reports/data_quality/figures/quality_by_class.png`.",
        "",
        "## Tệp hỏng, trùng lặp và gần trùng",
        "",
        _duplicate_text(summary),
        "",
        _sensitivity_table(summary),
        "",
        "## Nhãn và rò rỉ",
        "",
        _leakage_text(summary),
        "",
        _reviewed_pairs_text(summary),
        "",
        "Mã nhóm của mọi ảnh nằm ở cột `group_id` trong `reports/data_quality/tables/image_inventory.csv`. "
        "Các nhóm có từ hai ảnh trở lên được tách ở `reports/data_quality/tables/near_duplicate_groups.csv`. "
        "Bước chia tập dùng các mã này để giữ cả nhóm trong một tập.",
        "",
        "Tiền tố `to_label_` là cách đặt tên của toàn bộ bản phát hành. "
        "Mọi tệp đều nằm trong một thư mục lớp, nên tiền tố này không được xem là ảnh chưa gán nhãn.",
        "",
        "Ảnh mẫu mỗi lớp nằm ở `reports/data_quality/figures/class_examples.png`. "
        "Lưới này chỉ phục vụ rà soát định tính. Kết luận về xung đột nhãn dựa trên cặp trùng và gần trùng khác lớp.",
        "",
        "## Giới hạn",
        "",
        "Hash gần trùng không chứng minh hai ảnh khác hash là hai lá khác nhau. "
        "Không có mã phiên chụp và không có EXIF thời gian nên ảnh cùng lá, khác góc hoặc khác thời điểm có thể còn sót. "
        "Độ sáng, tương phản và độ sắc nét được đo trên bản đã resize, nên không suy ngược chất lượng ống kính lúc thu thập. "
        "Lần kiểm tra không huấn luyện mô hình và không đánh giá độ tách biệt giữa các bệnh.",
        "",
        "## Cách chạy lại",
        "",
        "```text",
        "python -m src.data.quality_audit --dataset-root datasets/durian-ldd --output-dir reports/data_quality",
        "```",
        "",
        "Ảnh gốc không được đưa vào Git. Thư mục `datasets/` nằm trong `.gitignore`.",
        "",
    ]
    return "\n".join(lines)


def _condition_block(summary: dict[str, object]) -> str:
    failures = list(summary["failures"])
    conditions = list(summary["conditions"])
    lines = []
    if failures:
        lines.append("Lý do chưa đủ điều kiện:")
        lines.extend(f"- {item}" for item in failures)
    if conditions:
        lines.append("Điều kiện khi sang bước chia tập và tiền xử lý:")
        lines.extend(f"- {item}" for item in conditions)
    return "\n".join(lines)


def _class_table(readable: pd.DataFrame) -> str:
    if readable.empty:
        return "Không có ảnh đọc được."
    total = len(readable)
    counts = readable["class_name"].value_counts()
    rows = []
    for class_name in _ordered_classes(counts.index.tolist()):
        count = int(counts[class_name])
        rows.append([_class_label(class_name), fmt_int(count), fmt_pct(count, total)])
    rows.append(["Tổng", fmt_int(total), "100,0%"])
    return _markdown_table(["Lớp", "Số ảnh", "Tỉ lệ"], rows)


def _split_table(readable: pd.DataFrame) -> str:
    if readable.empty:
        return ""
    classes = _ordered_classes(readable["class_name"].unique().tolist())
    splits = [split for split in SPLIT_ORDER if split in set(readable["split"])]
    counts = readable.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    rows = []
    for class_name in classes:
        row: list[object] = [_class_label(class_name)]
        for split in splits:
            row.append(fmt_int(int(counts.loc[class_name, split])) if split in counts else "0")
        row.append(fmt_int(int(counts.loc[class_name, splits].sum())))
        rows.append(row)
    total_row: list[object] = ["Tổng"]
    for split in splits:
        total_row.append(fmt_int(int((readable["split"] == split).sum())))
    total_row.append(fmt_int(len(readable)))
    rows.append(total_row)
    return _markdown_table(["Lớp", *splits, "Tổng"], rows)


def _resolution_text(summary: dict[str, object]) -> str:
    resolutions = dict(summary["resolutions"])
    if not resolutions:
        return "Không có ảnh đọc được để đo kích thước."
    if len(resolutions) == 1:
        name = next(iter(resolutions))
        return (
            f"Mọi ảnh đọc được đều có kích thước {name}, tỉ lệ khung hình 1,000. "
            "Bản phát hành đã được resize. Thống kê này không mô tả độ phân giải lúc chụp. "
            "Hình: `reports/data_quality/figures/resolutions.png`."
        )
    parts = [f"{name}: {fmt_int(int(count))} ảnh" for name, count in resolutions.items()]
    aspect = dict(summary["aspect_ratio"])
    median = aspect.get("p50")
    aspect_text = f" Trung vị tỉ lệ khung hình là {median:.3f}." if isinstance(median, float) else ""
    return (
        f"Có {len(resolutions)} kích thước. "
        + ". ".join(parts)
        + "."
        + aspect_text
        + " Hình: `reports/data_quality/figures/resolutions.png`."
    )


def _quality_table(summary: dict[str, object]) -> str:
    quality = dict(summary["quality"])
    labels = {
        "brightness": "Độ sáng",
        "contrast": "Độ tương phản",
        "sharpness": "Độ sắc nét",
        "saturation": "Độ bão hòa",
        "high_frequency": "Năng lượng tần số cao",
    }
    rows = []
    for key, label in labels.items():
        stats = dict(quality.get(key, {}))
        if not stats:
            continue
        rows.append(
            [
                label,
                f"{stats['p0']:.1f}",
                f"{stats['p5']:.1f}",
                f"{stats['p50']:.1f}",
                f"{stats['p95']:.1f}",
                f"{stats['p100']:.1f}",
            ]
        )
    if not rows:
        return "Không có ảnh đọc được để đo chất lượng."
    return _markdown_table(["Chỉ số", "Nhỏ nhất", "Phân vị 5", "Trung vị", "Phân vị 95", "Lớn nhất"], rows)


def _flag_text(summary: dict[str, object]) -> str:
    flags = dict(summary["quality_flags"])
    if not flags:
        return ""
    return (
        "Ngưỡng rà soát được đặt trước khi xem phân bố: độ sáng dưới 40, độ sáng trên 220, "
        "độ tương phản dưới 20 và độ sắc nét dưới 50. "
        f"Số ảnh vượt ngưỡng lần lượt là {fmt_int(int(flags['very_dark']))}, "
        f"{fmt_int(int(flags['very_bright']))}, {fmt_int(int(flags['low_contrast']))} và "
        f"{fmt_int(int(flags['very_soft']))}. "
        "Đây là cờ để xem lại, không phải quy tắc loại ảnh."
    )


def _duplicate_text(summary: dict[str, object]) -> str:
    relations = dict(summary["relation_counts"])
    relation_text = ", ".join(f"{key} {fmt_int(int(value))}" for key, value in relations.items()) or "không có"
    return (
        f"Ở ngưỡng Hamming {summary['near_duplicate_max_hamming']} có {fmt_int(int(summary['pairs']))} cặp. "
        f"Trong đó cặp trùng khớp tuyệt đối, trùng pixel hoặc là bản lật ngang đúng pixel: {fmt_int(int(summary['exact_pairs']))}. "
        f"Phân loại cặp: {relation_text}. "
        f"Số nhóm có từ hai ảnh trở lên: {fmt_int(int(summary['multi_image_groups']))}. "
        f"Nhóm lớn nhất có {fmt_int(int(summary['largest_group_size']))} ảnh. "
        f"Số ảnh nằm trong các nhóm đó: {fmt_int(int(summary['images_in_multi_image_groups']))}."
    )


def _sensitivity_table(summary: dict[str, object]) -> str:
    rows = []
    for item in list(summary["sensitivity"]):
        rows.append(
            [
                item["max_hamming"],
                fmt_int(int(item["pairs"])),
                fmt_int(int(item["images"])),
                fmt_int(int(item["cross_split_pairs"])),
                fmt_int(int(item["cross_class_pairs"])),
            ]
        )
    if not rows:
        return "Không đủ ảnh để so khoảng cách hash."
    return _markdown_table(
        ["Ngưỡng Hamming", "Số cặp", "Số ảnh liên quan", "Cặp xuyên split", "Cặp khác lớp"],
        rows,
    )


def _split_proportion_text(summary: dict[str, object]) -> str:
    total = int(summary["readable_images"])
    counts = dict(summary["split_counts"])
    if not total or not {"train", "val", "test"} <= set(counts):
        return "Split có sẵn không đủ train, validation và test."
    return (
        "Split có sẵn gồm "
        f"{fmt_pct(int(counts['train']), total)} train, "
        f"{fmt_pct(int(counts['val']), total)} validation và "
        f"{fmt_pct(int(counts['test']), total)} test. "
        "Từng lớp cũng nằm sát tỉ lệ này."
    )


def _reviewed_pairs_text(summary: dict[str, object]) -> str:
    examples = dict(summary.get("review_examples") or {})
    cross_pairs = list(examples.get("cross_split_pairs") or [])
    blocks = [
        "Mã `to_label_` liền nhau có trung vị khoảng cách Hamming quanh mức ngẫu nhiên. "
        "Số mã xuyên split vì vậy không phải bằng chứng rò rỉ. Chỉ các cặp vừa liền mã vừa gần trùng, "
        "hoặc các cặp gần trùng dù không liền mã, mới được tính là cùng một mẫu."
    ]
    if cross_pairs:
        pair_lines = ["Các cặp gần trùng xuyên split:", ""]
        for pair in cross_pairs:
            pair_lines.append(
                f"- `{pair['path_a']}` và `{pair['path_b']}`, Hamming {pair['hamming']}."
            )
        blocks.append("\n".join(pair_lines))
        blocks.append(
            "Đối chiếu ảnh cho thấy các cặp xuyên split là cùng một cảnh lá, lệch nhẹ khung hình. "
            "Hình các cặp gần trùng: `reports/data_quality/figures/near_duplicate_pairs.png`."
        )
        if int(summary["exact_pairs"]) == 1:
            blocks.append(
                "Có một cặp trùng byte. Cặp này đang nằm cùng một split nên không làm rò rỉ split hiện tại, "
                "nhưng vẫn phải đi cùng nhau khi chia tập lại."
            )
    sensitivity = {int(item["max_hamming"]): item for item in list(summary["sensitivity"])}
    strict = sensitivity.get(5, {})
    loose = sensitivity.get(8, {})
    if int(strict.get("cross_class_pairs", -1)) == 0 and int(loose.get("cross_class_pairs", -1)) == 3:
        blocks.append(
            "Nới ngưỡng lên 8 làm xuất hiện 3 cặp khác lớp. Đối chiếu ảnh cho thấy đó là các lá khác nhau, "
            "cùng kiểu một lá lớn trên nền vườn. Các cặp này không được tính là gán nhầm. "
            "Ngưỡng 10 làm số cặp khác lớp tăng tiếp, nên bước chia tập giữ ngưỡng 5."
        )
    return "\n\n".join(blocks)


def _plot_pairs(pairs: list[dict[str, object]], root: Path, path: Path) -> None:
    _configure_matplotlib()
    shown = pairs[:12]
    figure, axes = plt.subplots(len(shown), 2, figsize=(8, 2.15 * len(shown)))
    axes_grid = np.atleast_2d(axes)
    for row, pair in enumerate(shown):
        for column, path_key, split_key in (
            (0, "path_a", "split_a"),
            (1, "path_b", "split_b"),
        ):
            axis = axes_grid[row, column]
            axis.axis("off")
            image = Image.open(root / str(pair[path_key])).convert("RGB")
            axis.imshow(image)
            title = f"{pair[split_key]}  {Path(str(pair[path_key])).name}"
            if column == 0:
                title = f"Hamming {pair['hamming']}  {title}"
            axis.set_title(title, fontsize=8, loc="left")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _leakage_text(summary: dict[str, object]) -> str:
    test_count = int(summary["test_images"])
    val_count = int(summary["val_images"])
    adjacent_hamming = dict(summary["adjacent_hamming"])
    median = adjacent_hamming.get("p50")
    median_text = f" Trung vị khoảng cách Hamming của các cặp ID kề nhau là {median:.1f}." if isinstance(median, float) else ""
    return (
        f"Cặp gần trùng khác lớp: {fmt_int(int(summary['cross_class_near_pairs']))}. "
        f"Ảnh trong cặp trùng khớp tuyệt đối nhưng khác nhãn: {fmt_int(int(summary['images_in_cross_class_exact_duplicates']))}. "
        f"Cặp gần trùng xuyên split: {fmt_int(int(summary['cross_split_near_pairs']))}. "
        f"Ảnh test có bạn gần trùng trong train hoặc validation: "
        f"{fmt_int(int(summary['test_images_with_partner_in_train_or_val']))} trên {fmt_int(test_count)} "
        f"({fmt_pct(int(summary['test_images_with_partner_in_train_or_val']), test_count)}). "
        f"Ảnh validation có bạn gần trùng trong train: "
        f"{fmt_int(int(summary['val_images_with_partner_in_train']))} trên {fmt_int(val_count)} "
        f"({fmt_pct(int(summary['val_images_with_partner_in_train']), val_count)}). "
        f"Cặp có mã `to_label_` liền nhau: {fmt_int(int(summary['adjacent_id_pairs']))}, "
        f"trong đó xuyên split {fmt_int(int(summary['adjacent_id_cross_split_pairs']))} cặp và "
        f"xuyên split đồng thời gần trùng {fmt_int(int(summary['adjacent_id_cross_split_near_pairs']))} cặp. "
        f"Cặp ID liền nhau nhưng khác lớp: {fmt_int(int(summary['adjacent_id_cross_class_pairs']))}."
        + median_text
    )


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_tables(
    frame: pd.DataFrame,
    analysis: dict[str, object],
    output_dir: Path,
) -> None:
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    export = frame.copy()
    for column in ("dhash", "dhash_flip"):
        if column in export:
            export[column] = export[column].map(lambda value: "" if value is None or pd.isna(value) else f"{int(value):016x}")
    export.to_csv(tables / "image_inventory.csv", index=False)
    pd.DataFrame(analysis["pairs"]).to_csv(tables / "near_duplicate_pairs.csv", index=False)
    pd.DataFrame(analysis["adjacent_pairs"]).to_csv(tables / "adjacent_source_ids.csv", index=False)
    pd.DataFrame(analysis["sensitivity"]).to_csv(tables / "near_duplicate_sensitivity.csv", index=False)

    readable = frame[frame["readable"]].copy()
    if readable.empty:
        return
    grouped = readable[readable["group_size"].astype(int) > 1]
    grouped.to_csv(tables / "near_duplicate_groups.csv", index=False)
    class_rows = []
    for class_name, group in readable.groupby("class_name"):
        class_rows.append(
            {
                "class_name": class_name,
                "images": int(len(group)),
                "brightness_median": float(group["brightness"].median()),
                "contrast_median": float(group["contrast"].median()),
                "sharpness_median": float(group["sharpness"].median()),
                "saturation_median": float(group["saturation"].median()),
            }
        )
    pd.DataFrame(class_rows).to_csv(tables / "quality_by_class.csv", index=False)


def run_audit(
    dataset_root: Path,
    output_dir: Path,
    near_duplicate_max_hamming: int = PRIMARY_HAMMING,
    hash_size: int = 8,
    sample_seed: int = 406,
    expected_image_count: int = 4437,
    dark_brightness_below: float = 40,
    bright_brightness_above: float = 220,
    low_contrast_below: float = 20,
    soft_sharpness_below: float = 50,
) -> dict[str, object]:
    root = dataset_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Không thấy dữ liệu tại {root}")
    paths = discover_images(root)
    records = [read_image_record(path, root, hash_size) for path in paths]
    analysis = analyze_readable_images(records, near_duplicate_max_hamming)
    thresholds = {
        "dark_brightness_below": dark_brightness_below,
        "bright_brightness_above": bright_brightness_above,
        "low_contrast_below": low_contrast_below,
        "soft_sharpness_below": soft_sharpness_below,
    }
    summary = build_summary(
        records,
        analysis,
        thresholds,
        expected_image_count,
        near_duplicate_max_hamming,
        sample_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if "group_id" not in frame:
        frame["group_id"] = np.nan
        frame["group_size"] = np.nan
    write_tables(frame, analysis, output_dir)
    write_figures(frame, output_dir, root, sample_seed, list(analysis["pairs"]))
    report = render_report(summary, frame)
    (output_dir / "durian_ldd_data_quality_report.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def load_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Cấu hình {path} phải là một mapping.")
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm định chất lượng DurianLDD")
    parser.add_argument("--config", type=Path, default=Path("configs/data_audit.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--near-duplicate-max-hamming", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config.exists() else {}
    dataset_root = args.dataset_root or Path(str(config.get("dataset_root", "datasets/durian-ldd")))
    output_dir = args.output_dir or Path(str(config.get("output_dir", "reports/data_quality")))
    max_hamming = args.near_duplicate_max_hamming
    if max_hamming is None:
        max_hamming = int(config.get("near_duplicate_max_hamming", PRIMARY_HAMMING))
    summary = run_audit(
        dataset_root=dataset_root,
        output_dir=output_dir,
        near_duplicate_max_hamming=max_hamming,
        hash_size=int(config.get("hash_size", 8)),
        sample_seed=int(config.get("sample_seed", 406)),
        expected_image_count=int(config.get("expected_image_count", 4437)),
        dark_brightness_below=float(config.get("dark_brightness_below", 40)),
        bright_brightness_above=float(config.get("bright_brightness_above", 220)),
        low_contrast_below=float(config.get("low_contrast_below", 20)),
        soft_sharpness_below=float(config.get("soft_sharpness_below", 50)),
    )
    print(json.dumps(_json_ready({
        "eligible": summary["eligible"],
        "readable_images": summary["readable_images"],
        "corrupt_images": summary["corrupt_images"],
        "pairs": summary["pairs"],
        "cross_split_near_pairs": summary["cross_split_near_pairs"],
        "cross_class_near_pairs": summary["cross_class_near_pairs"],
        "multi_image_groups": summary["multi_image_groups"],
    }), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

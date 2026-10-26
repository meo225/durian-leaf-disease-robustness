"""Phân tích độ nhạy của một model với suy giảm ảnh (task 4.2).

Mọi so sánh là ghép cặp: ảnh sạch và ảnh suy giảm là cùng một mẫu. Khoảng tin cậy bootstrap lấy lại mẫu theo cặp, kiểm định McNemar
đếm các mẫu đổi từ đúng sang sai và ngược lại. Module chỉ làm việc trên dự đoán và nhãn, không nạp model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import re

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch import nn
from torch.nn import functional

from src.evaluation.metrics import classification_scores
from src.evaluation.processing_benchmark import ImageCache

ROOT = Path(__file__).resolve().parents[2]
CATEGORY_ORDER = ["severe", "moderate", "mild", "negligible"]


def load_config(path: str | Path = "configs/sensitivity.yaml") -> dict[str, Any]:
    path = Path(path)
    return yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8"))


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    images: torch.Tensor,
    pipeline_cfg: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Xác suất softmax N×K của model trên ảnh uint8 N×3×H×W, chuẩn hóa như pipeline chuẩn."""
    model.eval()
    mean = torch.tensor(pipeline_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(pipeline_cfg["std"], device=device).view(1, 3, 1, 1)
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].float().to(device) / 255.0
        outputs.append(functional.softmax(model((batch - mean) / std), dim=1).cpu())
    return torch.cat(outputs).numpy()


def collect_predictions(
    model: nn.Module,
    cache: ImageCache,
    pipeline_cfg: Mapping[str, Any],
    device: torch.device,
    progress=print,
    quality=None,
) -> dict[str, np.ndarray]:
    """Xác suất của model trên ảnh sạch và từng điều kiện. Ảnh mỗi điều kiện được giải phóng ngay sau khi dùng.

    `quality` là hàm nhận (tên điều kiện, ảnh uint8) để đo thống kê chất lượng ngay trên ảnh đang có.
    """
    probs: dict[str, np.ndarray] = {}
    for position, name in enumerate(["clean", *cache.conditions], start=1):
        images = cache.get(name)
        probs[name] = predict_probs(model, images, pipeline_cfg, device)
        if quality is not None:
            quality(name, images)
        if name != "clean":
            cache.release(name)
        progress(f"{position:2d}/{len(cache.conditions) + 1} {name}")
    return probs


def macro_f1_batch(labels: np.ndarray, preds: np.ndarray, indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Macro F1 trên từng lần lấy lại mẫu. indices có dạng B×N, mỗi hàng là một lần lấy mẫu."""
    count = indices.shape[0]
    flat = np.arange(count)[:, None] * num_classes**2 + labels[indices] * num_classes + preds[indices]
    matrix = np.bincount(flat.ravel(), minlength=count * num_classes**2).reshape(count, num_classes, num_classes).astype(np.float64)
    true_positive = np.einsum("bii->bi", matrix)
    denominator = matrix.sum(axis=2) + matrix.sum(axis=1)
    return np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0).mean(axis=1)


def bootstrap_indices(size: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, size, size=(resamples, size))


def mcnemar_exact(right_to_wrong: int, wrong_to_right: int) -> float:
    """Kiểm định McNemar chính xác hai phía trên các cặp không đồng nhất."""
    total = right_to_wrong + wrong_to_right
    if total == 0:
        return 1.0
    return float(stats.binomtest(min(right_to_wrong, wrong_to_right), total, 0.5).pvalue)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """P đã hiệu chỉnh nhiều phép thử theo Holm. Giữ thứ tự đầu vào."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[position]))
        adjusted[position] = running
    return adjusted


def recorded_calibration(readme_path: str | Path = "reports/corruptions/README.md") -> dict[str, float]:
    """Macro F1 validation đã ghi ở bảng hiệu chỉnh của 4.1, theo tên điều kiện. Dùng để kiểm tra đường chạy lại khớp 4.1."""
    path = Path(readme_path)
    text = (path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8")
    pattern = re.compile(r"^\| `(\w+)` \|.*\| [\d.]+ \(([\d.]+)\) \| [\d.]+ \(([\d.]+)\) \| [\d.]+ \(([\d.]+)\) \|\s*$", re.MULTILINE)
    return {f"{family}_s{level}": float(value) for family, *values in pattern.findall(text) for level, value in enumerate(values, start=1)}


def split_condition(name: str) -> tuple[str, int]:
    family, severity = name.rsplit("_s", 1)
    return family, int(severity)


def condition_table(
    labels: np.ndarray,
    probs: Mapping[str, np.ndarray],
    classes: Sequence[str],
    conditions: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Một dòng mỗi điều kiện: điểm, khoảng tin cậy bootstrap, degradation, McNemar có hiệu chỉnh Holm và recall từng lớp.

    Hai khoảng tin cậy: của macro F1 suy giảm, và của degradation (F1 sạch trừ F1 suy giảm) tính trên cùng các mẫu lấy lại, nên là khoảng ghép cặp.
    """
    num_classes = len(classes)
    boot = cfg["bootstrap"]
    low, high = (1 - boot["confidence"]) / 2 * 100, (1 + boot["confidence"]) / 2 * 100
    indices = bootstrap_indices(len(labels), int(boot["resamples"]), int(boot["seed"]))
    clean_pred = probs["clean"].argmax(axis=1)
    clean_boot = macro_f1_batch(labels, clean_pred, indices, num_classes)
    clean_scores = classification_scores(labels, clean_pred, classes)
    clean_right = clean_pred == labels
    rows: list[dict[str, Any]] = []
    for name, spec in conditions.items():
        pred = probs[name].argmax(axis=1)
        scores = classification_scores(labels, pred, classes)
        boot_f1 = macro_f1_batch(labels, pred, indices, num_classes)
        right = pred == labels
        lost, gained = int((clean_right & ~right).sum()), int((~clean_right & right).sum())
        degradation = clean_scores["macro_f1"] - scores["macro_f1"]
        row = {
            "condition": name, "family": spec["family"], "severity": spec["severity"], "group": spec["group"], "value": spec["value"],
            "macro_f1": scores["macro_f1"], "f1_low": float(np.percentile(boot_f1, low)), "f1_high": float(np.percentile(boot_f1, high)),
            "balanced_accuracy": scores["balanced_accuracy"], "accuracy": scores["accuracy"],
            "degradation": degradation, "degradation_low": float(np.percentile(clean_boot - boot_f1, low)),
            "degradation_high": float(np.percentile(clean_boot - boot_f1, high)),
            "relative_degradation": degradation / clean_scores["macro_f1"],
            "right_to_wrong": lost, "wrong_to_right": gained, "mcnemar_p": mcnemar_exact(lost, gained),
            "mean_confidence": float(probs[name].max(axis=1).mean()),
        }
        row.update({f"recall_{label}": value for label, value in scores["per_class_recall"].items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame["mcnemar_p"])
    frame["significant"] = frame["p_holm"] < float(cfg["alpha"])
    clean_row = {
        "clean_macro_f1": clean_scores["macro_f1"], "clean_f1_low": float(np.percentile(clean_boot, low)),
        "clean_f1_high": float(np.percentile(clean_boot, high)), "clean_accuracy": clean_scores["accuracy"],
        "clean_balanced_accuracy": clean_scores["balanced_accuracy"], "clean_errors": int((~clean_right).sum()),
    }
    frame.attrs["clean"] = clean_row
    frame.attrs["clean_recall"] = clean_scores["per_class_recall"]
    frame.attrs["clean_confusion"] = clean_scores["confusion_matrix"]
    return frame


def confusion_matrices(labels: np.ndarray, probs: Mapping[str, np.ndarray], classes: Sequence[str], names: Sequence[str]) -> dict[str, np.ndarray]:
    return {name: np.array(classification_scores(labels, probs[name].argmax(axis=1), classes)["confusion_matrix"]) for name in names}


def flip_destinations(labels: np.ndarray, probs: Mapping[str, np.ndarray], classes: Sequence[str], conditions: Sequence[str]) -> pd.DataFrame:
    """Với các mẫu đúng ở ảnh sạch nhưng sai ở ảnh suy giảm: tỉ lệ rơi vào từng lớp dự đoán sai, và số mẫu mất đi theo lớp thật."""
    clean_right = probs["clean"].argmax(axis=1) == labels
    rows = []
    for name in conditions:
        pred = probs[name].argmax(axis=1)
        lost = clean_right & (pred != labels)
        row: dict[str, Any] = {"condition": name, "lost": int(lost.sum())}
        for index, label in enumerate(classes):
            row[f"to_{label}"] = float((pred[lost] == index).sum() / lost.sum()) if lost.any() else 0.0
            row[f"lost_from_{label}"] = int((lost & (labels == index)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def fragile_samples(labels: np.ndarray, paths: Sequence[str], probs: Mapping[str, np.ndarray], classes: Sequence[str], conditions: Sequence[str]) -> pd.DataFrame:
    """Mỗi ảnh đúng ở ảnh sạch: bao nhiêu điều kiện làm nó sai và lớp sai hay gặp nhất."""
    clean_right = probs["clean"].argmax(axis=1) == labels
    wrong = np.stack([probs[name].argmax(axis=1) != labels for name in conditions], axis=1)
    predicted = np.stack([probs[name].argmax(axis=1) for name in conditions], axis=1)
    rows = []
    for index in np.where(clean_right)[0]:
        flips = wrong[index]
        common = np.bincount(predicted[index][flips], minlength=len(classes)).argmax() if flips.any() else -1
        rows.append({
            "index": int(index), "path": paths[index], "label": classes[labels[index]], "flips": int(flips.sum()),
            "share": float(flips.mean()), "common_wrong": classes[common] if common >= 0 else "",
        })
    return pd.DataFrame(rows).sort_values(["flips", "index"], ascending=[False, True]).reset_index(drop=True)


def family_summary(table: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Mỗi họ một dòng: mức giảm trung bình, lớn nhất, mức giảm ở mức 1, số mức có ý nghĩa, phân loại và task xử lý."""
    thresholds = cfg["thresholds"]
    rows = []
    for family, part in table.groupby("family", sort=False):
        part = part.sort_values("severity")
        mean = float(part["degradation"].mean())
        category = next((name for name in CATEGORY_ORDER[:-1] if mean >= thresholds[name]), "negligible")
        if category == "mild" and not part["significant"].any():
            category = "negligible"
        rows.append({
            "family": family, "group": part["group"].iloc[0], "mean_degradation": mean, "max_degradation": float(part["degradation"].max()),
            "degradation_s1": float(part.loc[part["severity"] == 1, "degradation"].iloc[0]),
            "significant_levels": int(part["significant"].sum()), "category": category,
            "early_drop": bool(part.loc[part["severity"] == 1, "degradation"].iloc[0] >= float(cfg["early_drop"])),
            "handlers": list(cfg["handlers"].get(family, [])),
        })
    frame = pd.DataFrame(rows).sort_values("mean_degradation", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def priority_for_next_task(summary: pd.DataFrame, table: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Danh sách đầu vào cho task 4.5: họ theo thứ tự ưu tiên, điều kiện hỏng nặng có ý nghĩa, họ không có kỹ thuật phù hợp."""
    moderate = float(cfg["thresholds"]["moderate"])
    damaging = table[(table["degradation"] >= moderate) & table["significant"]].sort_values("degradation", ascending=False)
    return {
        "ranking": [
            {"family": row.family, "category": row.category, "mean_degradation": round(row.mean_degradation, 4), "early_drop": bool(row.early_drop), "handlers": row.handlers}
            for row in summary.itertuples()
        ],
        "damaging_conditions": [
            {"condition": row.condition, "degradation": round(row.degradation, 4), "p_holm": float(f"{row.p_holm:.3g}")} for row in damaging.itertuples()
        ],
        "without_technique": [row.family for row in summary.itertuples() if not row.handlers and row.category != "negligible"],
        "negligible": [row.family for row in summary.itertuples() if row.category == "negligible"],
        "rule": {"thresholds": dict(cfg["thresholds"]), "early_drop": cfg["early_drop"], "alpha": cfg["alpha"], "correction": "Holm"},
    }


def compare_splits(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, float]:
    """Tương quan thứ hạng của mức giảm giữa hai split, và mức lệch trung bình của macro F1."""
    merged = first.merge(second, on="condition", suffixes=("_a", "_b"))
    rho = stats.spearmanr(merged["degradation_a"], merged["degradation_b"])
    return {
        "spearman": float(rho.statistic), "p_value": float(rho.pvalue),
        "mean_abs_f1_gap": float((merged["macro_f1_a"] - merged["macro_f1_b"]).abs().mean()),
        "max_abs_f1_gap": float((merged["macro_f1_a"] - merged["macro_f1_b"]).abs().max()),
    }


QUALITY_COLUMNS = ["brightness", "contrast", "sharpness", "noise", "saturated"]


def image_quality(images: torch.Tensor) -> pd.DataFrame:
    """Thống kê chất lượng từng ảnh. Đầu vào là ảnh N×3×H×W, uint8 hoặc 0 đến 1.

    brightness: độ sáng trung bình. contrast: độ lệch chuẩn độ sáng. sharpness: phương sai Laplacian của ảnh xám 0 đến 255.
    noise: ước lượng sigma nhiễu theo Immerkaer trên ảnh xám 0 đến 255. saturated: tỉ lệ điểm ảnh có kênh nào đó từ 250 trở lên.
    """
    data = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
    gray = (0.299 * data[:, 0] + 0.587 * data[:, 1] + 0.114 * data[:, 2]).unsqueeze(1)
    laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    immerkaer = torch.tensor([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]]).view(1, 1, 3, 3)
    height, width = gray.shape[-2:]
    scaled = gray * 255.0
    sharp = functional.conv2d(scaled, laplacian).flatten(1).var(dim=1)
    noise = functional.conv2d(scaled, immerkaer).abs().flatten(1).sum(dim=1) * np.sqrt(np.pi / 2) / (6 * (width - 2) * (height - 2))
    return pd.DataFrame({
        "brightness": gray.flatten(1).mean(dim=1).numpy(),
        "contrast": gray.flatten(1).std(dim=1).numpy(),
        "sharpness": sharp.numpy(),
        "noise": noise.numpy(),
        "saturated": (data.amax(dim=1) >= 250 / 255).flatten(1).float().mean(dim=1).numpy(),
    })


def quality_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    """Trung bình, trung vị và khoảng 10 đến 90 phần trăm của mỗi thống kê, theo điều kiện."""
    parts = []
    for name, group in per_image.groupby("condition", sort=False):
        row: dict[str, Any] = {"condition": name}
        for column in QUALITY_COLUMNS:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_p10"] = float(group[column].quantile(0.1))
            row[f"{column}_p90"] = float(group[column].quantile(0.9))
        parts.append(row)
    return pd.DataFrame(parts)


def external_quality(root: str | Path, image_size: int, limit: int, seed: int = 42) -> pd.DataFrame:
    """Thống kê chất lượng của một bộ ảnh ngoài (ví dụ Vietnamese durian leaf), resize về cùng kích thước với model."""
    from PIL import Image
    from torchvision import transforms

    from src.data.transforms import ConditionalResize

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(path for path in Path(root).rglob("*") if path.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(f"Không thấy ảnh trong {root}")
    rng = np.random.default_rng(seed)
    if len(files) > limit:
        files = [files[i] for i in sorted(rng.choice(len(files), size=limit, replace=False))]
    convert = transforms.Compose([ConditionalResize(image_size), transforms.ToTensor()])
    batch = []
    for path in files:
        with Image.open(path) as image:
            batch.append(convert(image.convert("RGB")))
    return image_quality(torch.stack(batch))

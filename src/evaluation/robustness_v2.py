"""Đánh giá bốn hệ trên bộ họ chưa thấy v2 (task 5.11). Các hàm thuần dữ liệu, notebook 16 lo phần chạy mô hình.

Bốn hệ: không xử lý, oracle, bộ chọn và huấn luyện lại. Phạm vi và kỳ vọng đăng ký trước ở `docs/task-5.11-robustness-v2.md`, tham số khóa ở
`configs/robustness_v2.yaml`. Chỉ validation và chỉ họ vai trò train, val: họ vai trò test cùng tập ảnh test để dành cho P6.1.

Quy ước dữ liệu giống `control_compare`: một cách đo là ma trận dự đoán (1 + C) × N, hàng đầu là ảnh sạch, C hàng sau là điều kiện theo thứ tự `names`.
Mọi so sánh ghép cặp theo ảnh (cùng chỉ số bootstrap cho mọi cách). Thước đo chính của mỗi họ là macro F1 trung bình ba mức.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.corruption.benchmark import ROOT, load_conditions
from src.evaluation import seed_summary as S
from src.evaluation.control_compare import bootstrap_indices, split_name, verdict
from src.evaluation.intervals import per_class_recall_wilson, spearman_with_ci
from src.evaluation.processing_benchmark import ImageCache, predict
from src.selector.ablation import condition_oracle_predictions, oracle_predictions
from src.selector.calibrate import KEEP, CalibrationTable, assign_selector, gather_predictions, macro_f1_rows
from src.selector.rules import RuleSelector

SETTINGS_PATH = "configs/robustness_v2.yaml"
ALLOWED_ROLES = ("train", "val")
SYSTEMS = ("none", "selector", "oracle_family", "oracle_condition", "oracle_image")
ALL_FAMILIES = "tất cả họ"


# --- Cấu hình và điều kiện ----------------------------------------------------------------------------------------------------------------


def load_settings(path: str | Path = SETTINGS_PATH) -> dict[str, Any]:
    """Đọc `configs/robustness_v2.yaml` và từ chối cấu hình mở tập test hay họ vai trò test.

    5.11 là bước chốt cấu hình, nên không được thấy kết quả của họ vai trò test: nếu thấy rồi thì mọi chỉnh sửa sau đó làm P6.1 hết giá trị.
    """
    path = Path(path)
    settings = yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8"))
    if settings.get("split") != "val":
        raise PermissionError(f"5.11 chỉ dùng validation, config ghi split = {settings.get('split')!r}.")
    roles = tuple(settings.get("roles") or ())
    if not roles or not set(roles) <= set(ALLOWED_ROLES):
        raise PermissionError(f"5.11 chỉ dùng họ vai trò {ALLOWED_ROLES}, config ghi {roles}. Họ vai trò test để dành cho P6.1.")
    stats = settings["statistics"]
    if stats["k"] <= 0 or stats["bootstrap_resamples"] < 100 or stats["min_reference_drop"] <= 0:
        raise ValueError("statistics có giá trị không hợp lệ.")
    return settings


def evaluation_conditions(settings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Điều kiện của các họ trong phạm vi 5.11 (config v2 đã khóa, kiểm mã băm trong `load_conditions`)."""
    conditions = load_conditions(settings["unseen_config"], settings["split"], roles=tuple(settings["roles"]))
    if not conditions:
        raise ValueError("Không có điều kiện nào trong phạm vi vai trò đã chọn.")
    leaked = sorted({spec["family"] for spec in conditions.values() if spec.get("role") not in ALLOWED_ROLES})
    if leaked:
        raise PermissionError(f"Họ ngoài phạm vi 5.11: {leaked}.")
    return conditions


def family_order(names: Sequence[str]) -> list[str]:
    """Họ theo thứ tự xuất hiện đầu tiên trong danh sách điều kiện."""
    return list(dict.fromkeys(split_name(name)[0] for name in names))


def _family_weights(names: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Ma trận C × F: trọng số 1/số mức nếu điều kiện thuộc họ đó, để nhân với điểm từng điều kiện ra trung bình theo họ."""
    families = family_order(names)
    weights = np.zeros((len(names), len(families)))
    for row, name in enumerate(names):
        weights[row, families.index(split_name(name)[0])] = 1.0
    return families, weights / weights.sum(axis=0, keepdims=True)


# --- Chạy model trên ảnh đã dựng ---------------------------------------------------------------------------------------------------------


class RetainedCache(ImageCache):
    """`ImageCache` không bao giờ nhả ảnh đã dựng.

    Họ ngoài chạy trên CPU và có họ mất hơn nửa giây mỗi ảnh. Notebook dựng mỗi điều kiện đúng một lần rồi chạy mọi model trên cùng ảnh,
    nên cũng bảo đảm mọi hệ nhìn đúng cùng ảnh suy giảm. `collect` gọi `release` sau mỗi điều kiện, ở đây thành không làm gì.
    """

    def release(self, condition: str) -> None:
        return None


def model_matrix(model: nn.Module, cache: ImageCache, names: Sequence[str], pipeline_cfg: Mapping[str, Any], device: torch.device, batch_size: int = 64) -> np.ndarray:
    """Ma trận dự đoán (1 + C) × N của một model: hàng đầu ảnh sạch, C hàng sau là các điều kiện theo thứ tự `names`. Không xử lý ảnh trước khi nhận diện."""
    rows = [predict(model, cache.get(name), None, pipeline_cfg, device, batch_size=batch_size) for name in ("clean", *names)]
    matrix = np.stack(rows).astype(np.int64)
    if matrix.shape != (len(names) + 1, len(cache)):
        raise RuntimeError(f"Ma trận có dạng {matrix.shape}, cần {(len(names) + 1, len(cache))}.")
    return matrix


# --- Điểm theo họ kèm bootstrap ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyScores:
    """Macro F1 trung bình ba mức theo họ: giá trị điểm (F) và R mẫu bootstrap theo ảnh (R × F). Cộng, trừ và trung bình giữ nguyên ghép cặp."""

    families: tuple[str, ...]
    point: np.ndarray
    boot: np.ndarray

    def __post_init__(self) -> None:
        if self.boot.ndim != 2 or self.boot.shape[1] != len(self.families) or self.point.shape != (len(self.families),):
            raise ValueError("point cần dạng (F,) và boot cần dạng (R, F).")

    def minus(self, other: "FamilyScores") -> "FamilyScores":
        if self.families != other.families or self.boot.shape != other.boot.shape:
            raise ValueError("Hai điểm số khác họ hoặc khác số mẫu bootstrap, không ghép cặp được.")
        return FamilyScores(self.families, self.point - other.point, self.boot - other.boot)

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(self.boot, [tail, 1.0 - tail], axis=0)
        return low, high

    def overall(self) -> tuple[float, float, float]:
        """Trung bình các họ (mỗi họ một phiếu): điểm và khoảng tin cậy 95%."""
        samples = self.boot.mean(axis=1)
        return float(self.point.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def mean_scores(items: Sequence[FamilyScores]) -> FamilyScores:
    """Trung bình theo seed của một nhánh. Mẫu bootstrap thứ r của mọi seed dùng cùng tập ảnh, nên trung bình từng mẫu là khoảng của nhánh."""
    if not items:
        raise ValueError("Cần ít nhất một điểm số.")
    if any(item.families != items[0].families or item.boot.shape != items[0].boot.shape for item in items):
        raise ValueError("Các seed phải cùng họ và cùng số mẫu bootstrap.")
    return FamilyScores(items[0].families, np.mean([item.point for item in items], axis=0), np.mean([item.boot for item in items], axis=0))


def family_scores(labels: np.ndarray, matrix: np.ndarray, names: Sequence[str], classes: int, indices: np.ndarray) -> FamilyScores:
    """Điểm theo họ của một cách đo. `indices` (R × N) là chỉ số bootstrap dùng chung cho mọi cách (`control_compare.bootstrap_indices`)."""
    if matrix.shape != (len(names) + 1, len(labels)):
        raise ValueError(f"Ma trận dự đoán có dạng {matrix.shape}, cần {(len(names) + 1, len(labels))}.")
    families, weights = _family_weights(names)
    point = macro_f1_rows(labels, matrix[1:], classes) @ weights
    boot = np.stack([macro_f1_rows(labels[chosen], matrix[1:, chosen], classes) for chosen in indices]) @ weights
    return FamilyScores(tuple(families), point, boot)


def clean_score(labels: np.ndarray, matrix: np.ndarray, classes: int) -> float:
    return float(macro_f1_rows(labels, matrix[:1], classes)[0])


def recovered_fraction(system: np.ndarray, base: np.ndarray, base_clean: float, levels: int, min_drop: float) -> np.ndarray:
    """Phần ΔF1 của `base` mà `system` lấy lại được, theo họ: (hệ − base) / (sạch của base − base), cả hai tính trung bình ba mức.

    Tỉ số của hai tổng ba mức nên ổn định hơn tỉ số từng mức. Họ mà tổng ΔF1 của base (ba mức cộng lại) nhỏ hơn `min_drop` trả về NaN,
    vì chia cho số gần 0 cho tỉ lệ vô nghĩa. Âm nghĩa là hệ làm tệ hơn base.
    """
    drop = base_clean - np.asarray(base, dtype=float)
    result = np.full(drop.shape, np.nan)
    usable = drop * levels >= min_drop
    result[usable] = (np.asarray(system, dtype=float)[usable] - np.asarray(base, dtype=float)[usable]) / drop[usable]
    return result


# --- Bốn hệ trên reference đóng băng ------------------------------------------------------------------------------------------------------


def system_predictions(table: CalibrationTable, selector: RuleSelector) -> dict[str, np.ndarray]:
    """Dự đoán (1 + C) × N của năm cách trên cùng bảng `CalibrationTable`: không xử lý, bộ chọn đã khóa và ba cận oracle (dùng nhãn nên không triển khai được).

    Bộ chọn không được hiệu chỉnh lại ở đây: nó nhìn đặc trưng chất lượng của ảnh suy giảm chưa xử lý, đúng như khi triển khai.
    """
    predictions = {
        "none": table.predictions[table.actions.index(KEEP)],
        "selector": gather_predictions(table, assign_selector(table, selector)),
        "oracle_family": condition_oracle_predictions(table, by_family=True),
        "oracle_condition": condition_oracle_predictions(table),
        "oracle_image": oracle_predictions(table),
    }
    shapes = {name: matrix.shape for name, matrix in predictions.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"Các hệ có dạng khác nhau: {shapes}.")
    return {name: matrix.astype(np.int64) for name, matrix in predictions.items()}


def action_rates(table: CalibrationTable, selector: RuleSelector) -> pd.DataFrame:
    """Tỉ lệ ảnh mỗi hành động được chọn, theo điều kiện. Cho thấy bộ chọn có kích hoạt đúng chỗ, hay lặng im ở họ không có luật."""
    chosen = assign_selector(table, selector)
    rows = []
    for position, name in enumerate(table.conditions):
        row = {"condition": name, "family": table.families[position]}
        row.update({action: float((chosen[position] == index).mean()) for index, action in enumerate(table.actions)})
        row["processed_rate"] = 1.0 - row[KEEP]
        rows.append(row)
    return pd.DataFrame(rows)


# --- Phân phối lớp dự đoán và recall lớp nhỏ ---------------------------------------------------------------------------------------------


def prediction_entropy(predictions: np.ndarray, classes: int) -> float:
    """Entropy Shannon của phân phối lớp dự đoán, chia cho log(số lớp) nên nằm trong [0, 1]. Gần 0 nghĩa là dự đoán dồn vào một lớp (R32)."""
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=classes).astype(float)
    if counts.sum() == 0 or classes < 2:
        raise ValueError("Cần ít nhất một dự đoán và hai lớp.")
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(classes))


def condition_diagnostics(labels: np.ndarray, matrix: np.ndarray, names: Sequence[str], class_names: Sequence[str], small_class: str) -> pd.DataFrame:
    """Mỗi điều kiện một dòng: macro F1, entropy lớp dự đoán, lớp bị dự đoán nhiều nhất và tỉ lệ của nó, recall lớp nhỏ kèm khoảng Wilson 95%.

    Entropy của nhãn thật ghi riêng (`label_entropy`) để so: dự đoán thấp hơn nhãn thật nhiều là dấu hiệu dồn lớp.
    """
    if small_class not in class_names:
        raise ValueError(f"Không có lớp {small_class} trong {list(class_names)}.")
    classes = len(class_names)
    label_entropy = prediction_entropy(labels, classes)
    f1 = macro_f1_rows(labels, matrix[1:], classes)
    rows = []
    for position, name in enumerate(names):
        preds = matrix[position + 1].astype(np.int64)
        counts = np.bincount(preds, minlength=classes)
        recall = per_class_recall_wilson(labels, preds, class_names)[small_class]
        family, severity = split_name(name)
        rows.append({
            "condition": name, "family": family, "severity": severity, "macro_f1": float(f1[position]),
            "pred_entropy": prediction_entropy(preds, classes), "label_entropy": label_entropy,
            "top_class": class_names[int(counts.argmax())], "top_share": float(counts.max() / counts.sum()),
            "small_support": int(recall["support"]), "small_recall": recall["recall"], "small_low": recall["low"], "small_high": recall["high"],
        })
    return pd.DataFrame(rows)


# --- So sánh ---------------------------------------------------------------------------------------------------------------------------------


def paired_table(system: FamilyScores, base: FamilyScores) -> pd.DataFrame:
    """Hiệu hệ trừ base theo họ và trung bình các họ, kèm khoảng bootstrap ghép theo ảnh và kết luận theo khoảng (không có seed nên không dùng quy tắc SD)."""
    delta = system.minus(base)
    low, high = delta.interval()
    rows = [
        {"family": family, "system": float(system.point[i]), "base": float(base.point[i]), "delta": float(delta.point[i]), "low": float(low[i]), "high": float(high[i]),
         "kết luận": verdict(float(delta.point[i]), float(low[i]), float(high[i]))}
        for i, family in enumerate(system.families)
    ]
    value, lo, hi = delta.overall()
    rows.append({"family": ALL_FAMILIES, "system": float(system.point.mean()), "base": float(base.point.mean()), "delta": value, "low": lo, "high": hi, "kết luận": verdict(value, lo, hi)})
    return pd.DataFrame(rows)


def _seed_matrix(items: Mapping[int, FamilyScores]) -> tuple[tuple[str, ...], np.ndarray]:
    """Điểm từng seed, dạng (số seed × F), cùng thứ tự họ."""
    first = next(iter(items.values()))
    if any(item.families != first.families for item in items.values()):
        raise ValueError("Các seed khác thứ tự họ.")
    return first.families, np.stack([items[seed].point for seed in sorted(items)])


def _with_overall(families: Sequence[str], matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    return [*families, ALL_FAMILIES], np.concatenate([matrix, matrix.mean(axis=1, keepdims=True)], axis=1)


def compare_seed_arms(a: Mapping[int, FamilyScores], b: Mapping[int, FamilyScores], k: float = S.K) -> pd.DataFrame:
    """Hai nhánh nhiều seed, theo họ: hiệu trung bình a − b, SD gộp của seed, tỉ số hiệu/SD và kết luận theo quy tắc 5.9 (chỉ kết luận khi hiệu > k·SD)."""
    families, points_a = _seed_matrix(a)
    other_families, points_b = _seed_matrix(b)
    if families != other_families:
        raise ValueError("Hai nhánh khác họ.")
    names, points_a = _with_overall(families, points_a)
    _, points_b = _with_overall(families, points_b)
    rows = []
    for column, family in enumerate(names):
        stats_a, stats_b = S.seed_stats(points_a[:, column]), S.seed_stats(points_b[:, column])
        delta = stats_a["mean"] - stats_b["mean"]
        sd = S.pooled_sd(stats_a["sd"], stats_b["sd"])
        rows.append({"family": family, "a": stats_a["mean"], "b": stats_b["mean"], "delta": delta, "pooled_sd": sd,
                     "ratio": abs(delta) / sd if sd and not math.isnan(sd) else float("nan"), "kết luận": S.decide(delta, sd, k)})
    return pd.DataFrame(rows)


def compare_single_to_seed_arm(single: FamilyScores, reference: Mapping[int, FamilyScores], k: float = S.K) -> pd.DataFrame:
    """Nhánh một seed so với nhánh tham chiếu nhiều seed, theo họ. SD lấy từ nhánh tham chiếu, như 5.10."""
    families, points = _seed_matrix(reference)
    if single.families != families:
        raise ValueError("Hai nhánh khác họ.")
    names, points = _with_overall(families, points)
    _, values = _with_overall(families, single.point[None, :])
    rows = []
    for column, family in enumerate(names):
        rows.append({"family": family, "single": float(values[0, column]), **S.compare_single_to_arm(float(values[0, column]), points[:, column], k)})
    return pd.DataFrame(rows)


def seed_spread(items: Mapping[int, FamilyScores]) -> pd.DataFrame:
    """Trung bình, SD seed, nhỏ nhất và lớn nhất theo họ của một nhánh nhiều seed."""
    families, points = _seed_matrix(items)
    names, points = _with_overall(families, points)
    return pd.DataFrame([{"family": family, **S.seed_stats(points[:, column])} for column, family in enumerate(names)])


# --- MSD và kết quả -------------------------------------------------------------------------------------------------------------------------


def family_means(frame: pd.DataFrame) -> pd.Series:
    """Macro F1 trung bình theo họ từ bảng điều kiện của một lần train (cột condition, family, macro_f1), bỏ dòng ảnh sạch."""
    rows = frame[frame["condition"] != "clean"]
    if rows.empty:
        raise ValueError("Bảng không có điều kiện suy giảm nào.")
    return rows.groupby("family", sort=False)["macro_f1"].mean()


def seen_effect(arm_frames: Sequence[pd.DataFrame], base_frames: Sequence[pd.DataFrame]) -> dict[str, float]:
    """Mức hơn baseline theo họ của một nhánh: trung bình các seed của nhánh trừ trung bình các seed của baseline. Dùng cho 14 họ đã thấy ở 5.9."""
    if not arm_frames or not base_frames:
        raise ValueError("Cần ít nhất một bảng cho mỗi nhánh.")
    arm = pd.concat([family_means(frame) for frame in arm_frames], axis=1).mean(axis=1)
    base = pd.concat([family_means(frame) for frame in base_frames], axis=1).mean(axis=1)
    if set(arm.index) != set(base.index):
        raise ValueError("Hai nhánh khác tập họ.")
    return {family: float(arm[family] - base[family]) for family in arm.index}


def msd_vectors(record: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """MSD theo họ từ `reports/unseen_v2/msd.json`: `ta_seen` (14 họ đã thấy tới TrivialAugment), `ta_new` và `aug_new` (họ v2 tới TrivialAugment và tới augmentation suy giảm)."""
    return {
        "ta_seen": {family: float(entry["msd"]) for family, entry in record["seen_to_trivialaugment"].items()},
        "ta_new": {entry["family"]: float(entry["msd_ta"]) for entry in record["new_families"]},
        "aug_new": {entry["family"]: float(entry["msd"]) for entry in record["new_families"]},
    }


def msd_correlation(msd: Mapping[str, float], effect: Mapping[str, float], min_families: int = 5) -> dict[str, float]:
    """Spearman giữa MSD và một hiệu ứng (ví dụ phần TrivialAugment hơn baseline) trên các họ có cả hai giá trị, kèm khoảng bootstrap theo cặp.

    Với ít họ khoảng rất rộng và phải báo như vậy. Dưới `min_families` họ thì từ chối: hệ số không còn mang thông tin.
    """
    shared = sorted(set(msd) & set(effect))
    pairs = [(msd[name], effect[name]) for name in shared if not (math.isnan(msd[name]) or math.isnan(effect[name]))]
    if len(pairs) < min_families:
        raise ValueError(f"Chỉ có {len(pairs)} họ có đủ cả hai giá trị, cần ít nhất {min_families}.")
    x, y = zip(*pairs)
    return dict(spearman_with_ci(x, y))


# --- Ghép mọi bảng --------------------------------------------------------------------------------------------------------------------------

THREE_SEED_ARMS = ("baseline", "degradation", "trivialaugment")
SINGLE_SEED_ARMS = ("trivialaugment_geometric", "degradation_leave_group")
BASE_ARM = "baseline"
ORACLES = ("oracle_family", "oracle_condition", "oracle_image")
DIAGNOSTIC_SYSTEMS = ("none", "selector", "oracle_condition")


def _levels_per_family(names: Sequence[str]) -> int:
    counts = pd.Series([split_name(name)[0] for name in names]).value_counts()
    if counts.nunique() != 1:
        raise ValueError(f"Các họ có số mức khác nhau: {counts.to_dict()}.")
    return int(counts.iloc[0])


def condition_scores(labels: np.ndarray, matrix: np.ndarray, names: Sequence[str], classes: int) -> pd.DataFrame:
    """Macro F1 từng điều kiện, kể cả ảnh sạch (dòng `clean`)."""
    f1 = macro_f1_rows(labels, matrix, classes)
    rows = [{"condition": "clean", "family": "clean", "severity": 0, "macro_f1": float(f1[0])}]
    for position, name in enumerate(names, start=1):
        family, severity = split_name(name)
        rows.append({"condition": name, "family": family, "severity": severity, "macro_f1": float(f1[position])})
    return pd.DataFrame(rows)


def _fraction_rows(label: str, system: FamilyScores, base: FamilyScores, base_clean: float, levels: int, min_drop: float) -> pd.DataFrame:
    return pd.DataFrame({
        "system": label, "family": system.families, "base_f1": base.point, "base_drop": base_clean - base.point, "system_f1": system.point,
        "recovered": recovered_fraction(system.point, base.point, base_clean, levels, min_drop),
    })


def build_tables(
    labels: np.ndarray,
    names: Sequence[str],
    class_names: Sequence[str],
    reference: Mapping[str, np.ndarray],
    retrained: Mapping[str, Mapping[int, np.ndarray]],
    settings: Mapping[str, Any],
    side_check: Mapping[str, np.ndarray] | None = None,
) -> dict[str, pd.DataFrame]:
    """Mọi bảng của 5.11 từ các ma trận dự đoán (1 + C) × N.

    `reference`: năm hệ trên reference đóng băng (`SYSTEMS`). `retrained`: {nhánh: {seed: ma trận}}, đủ ba nhánh ba seed và các nhánh một seed.
    `side_check`: năm hệ trên baseline mới seed 42 (kiểm tra phụ), hoặc None. Mọi hệ chấm trên cùng ảnh và cùng chỉ số bootstrap.
    """
    stats = settings["statistics"]
    classes = len(class_names)
    if set(reference) != set(SYSTEMS):
        raise ValueError(f"reference cần đủ {SYSTEMS}, có {sorted(reference)}.")
    for arm in THREE_SEED_ARMS:
        if len(retrained.get(arm, {})) < 2:
            raise ValueError(f"Nhánh {arm} cần ít nhất 2 seed để có SD.")
    unknown = set(retrained) - set(THREE_SEED_ARMS) - set(SINGLE_SEED_ARMS)
    if unknown:
        raise ValueError(f"Nhánh không nhận ra: {sorted(unknown)}.")
    levels = _levels_per_family(names)
    indices = bootstrap_indices(len(labels), int(stats["bootstrap_resamples"]), int(stats["bootstrap_seed"]))

    ref_scores = {name: family_scores(labels, matrix, names, classes, indices) for name, matrix in reference.items()}
    seed_scores = {arm: {seed: family_scores(labels, matrix, names, classes, indices) for seed, matrix in runs.items()} for arm, runs in retrained.items()}
    arm_means = {arm: mean_scores(list(runs.values())) for arm, runs in seed_scores.items()}
    k, min_drop = float(stats["k"]), float(stats["min_reference_drop"])

    tables: dict[str, pd.DataFrame] = {}

    # Điểm từng điều kiện của mọi hệ và mọi seed.
    rows = [condition_scores(labels, matrix, names, classes).assign(system=name, seed=42, model="reference_v1") for name, matrix in reference.items()]
    for arm, runs in retrained.items():
        rows += [condition_scores(labels, matrix, names, classes).assign(system=arm, seed=seed, model="retrained") for seed, matrix in runs.items()]
    tables["condition_scores"] = pd.concat(rows, ignore_index=True)

    # Bộ chọn và oracle so với không xử lý, cùng nền reference.
    tables["systems_vs_none"] = pd.concat([paired_table(ref_scores[name], ref_scores["none"]).assign(system=name) for name in ("selector", *ORACLES)], ignore_index=True)

    # Các nhánh huấn luyện lại.
    comparisons = []
    for a, b in S.PAIRS_5_9:
        comparisons.append(compare_seed_arms(seed_scores[a], seed_scores[b], k).assign(pair=f"{a} − {b}"))
    for arm in SINGLE_SEED_ARMS:
        if arm in seed_scores:
            single = next(iter(seed_scores[arm].values()))
            for reference_arm in (BASE_ARM, "degradation"):
                comparisons.append(compare_single_to_seed_arm(single, seed_scores[reference_arm], k).assign(pair=f"{arm} − {reference_arm}"))
    tables["retrained_comparison"] = pd.concat(comparisons, ignore_index=True)
    tables["retrained_vs_baseline_ci"] = pd.concat(
        [paired_table(arm_means[arm], arm_means[BASE_ARM]).assign(system=arm) for arm in arm_means if arm != BASE_ARM], ignore_index=True
    )
    tables["seed_spread"] = pd.concat([seed_spread(seed_scores[arm]).assign(system=arm) for arm in THREE_SEED_ARMS], ignore_index=True)

    # Phần ΔF1 của nền lấy lại được. Nền của bộ chọn và oracle là reference v1, nền của huấn luyện lại là baseline mới (trung bình seed).
    clean_ref = clean_score(labels, reference["none"], classes)
    clean_base = float(np.mean([clean_score(labels, matrix, classes) for matrix in retrained[BASE_ARM].values()]))
    fractions = [_fraction_rows(name, ref_scores[name], ref_scores["none"], clean_ref, levels, min_drop).assign(base="reference_v1") for name in ("selector", *ORACLES)]
    fractions += [_fraction_rows(arm, arm_means[arm], arm_means[BASE_ARM], clean_base, levels, min_drop).assign(base="baseline_5.9") for arm in arm_means if arm != BASE_ARM]
    tables["recovered_fraction"] = pd.concat(fractions, ignore_index=True)

    # Phân phối lớp dự đoán và recall lớp nhỏ: reference, bộ chọn, oracle theo điều kiện và ba nhánh seed 42 (một seed để bảng gọn).
    small = str(stats["small_class"])
    diagnostics = [condition_diagnostics(labels, reference[name], names, class_names, small).assign(system=name) for name in DIAGNOSTIC_SYSTEMS]
    diagnostics += [condition_diagnostics(labels, retrained[arm][42], names, class_names, small).assign(system=arm) for arm in THREE_SEED_ARMS if 42 in retrained[arm]]
    tables["diagnostics"] = pd.concat(diagnostics, ignore_index=True)

    if side_check is not None:
        if set(side_check) != set(SYSTEMS):
            raise ValueError(f"side_check cần đủ {SYSTEMS}.")
        side_scores = {name: family_scores(labels, matrix, names, classes, indices) for name, matrix in side_check.items()}
        tables["side_check_new_baseline"] = pd.concat([paired_table(side_scores[name], side_scores["none"]).assign(system=name) for name in ("selector", *ORACLES)], ignore_index=True)
    return tables


def correlation_table(
    vectors: Mapping[str, Mapping[str, float]],
    seen_ta_effect: Mapping[str, float],
    new_effects: Mapping[str, Mapping[str, float]],
    scope_families: Sequence[str],
    min_families: int = 5,
) -> pd.DataFrame:
    """Spearman giữa MSD và mức hơn baseline.

    TrivialAugment: 14 họ đã thấy (MSD `ta_seen`, mức hơn từ 5.9) cộng các họ v2 trong phạm vi (MSD `ta_new`). Tên họ đã thấy và họ v2 không trùng nhau.
    Suy giảm: chỉ họ v2 trong phạm vi (MSD `aug_new`), vì 14 họ đã thấy nằm trong chính augmentation đó nên mức hơn không nói gì về họ lạ.
    """
    scope = set(scope_families)
    if set(seen_ta_effect) & scope:
        raise ValueError("Họ đã thấy và họ v2 trùng tên.")
    specs = {
        "trivialaugment": ({**vectors["ta_seen"], **{f: v for f, v in vectors["ta_new"].items() if f in scope}}, {**seen_ta_effect, **new_effects["trivialaugment"]}),
        "degradation": ({f: v for f, v in vectors["aug_new"].items() if f in scope}, dict(new_effects["degradation"])),
    }
    rows = []
    for arm, (msd, effect) in specs.items():
        rows.append({"arm": arm, "msd": "trivialaugment" if arm == "trivialaugment" else "degradation_aug", **msd_correlation(msd, effect, min_families)})
    return pd.DataFrame(rows)

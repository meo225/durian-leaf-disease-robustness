"""Hiệu chỉnh ngưỡng của bộ chọn trên validation (task 5.3). Chỉ dùng validation, không mở tập test.

Hai giai đoạn, như bản đồ phục hồi (task 4.5):

A (cần GPU, `collect`): với ảnh sạch và từng điều kiện của benchmark 4.1, tính đặc trưng chất lượng và dự đoán của reference model đóng băng
   cho mỗi hành động có thể chọn (giữ nguyên và các cấu hình xử lý ảnh đã khóa). Kết quả là một bảng (`CalibrationTable`).
B (chạy trên CPU, `fit`): tìm hành động, đặc trưng và ngưỡng của từng luật trên bảng đó. Vì dự đoán của mọi hành động đã có sẵn, mỗi bộ ngưỡng
   được chấm bằng cách tra bảng, không chạy lại model.

Mục tiêu là macro F1 trung bình trên 42 điều kiện suy giảm, có hai ràng buộc trên ảnh sạch (giảm F1 và tỉ lệ kích hoạt). Một luật chỉ được giữ
nếu tăng mục tiêu ít nhất `min_gain`, nên kết quả âm (không luật nào đáng giữ) là kết quả hợp lệ. `cross_validate` kiểm tra độ ổn định của ngưỡng.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from src.evaluation.recovery_map import predict_probabilities
from src.selector.features import FEATURES, drop_groups, extract_batch
from src.selector.pipeline import apply_action
from src.selector.rules import Clause, Rule, RuleSelector

CLEAN = "clean"
KEEP = "keep"


@dataclass(frozen=True)
class Settings:
    """Tham số của bước hiệu chỉnh, lấy từ mục `calibration` của configs/selector.yaml."""

    threshold_points: int = 41
    guard_points: int = 9
    min_gain: float = 0.002
    max_clean_f1_drop: float = 0.005
    max_clean_trigger_rate: float = 0.02

    @classmethod
    def from_config(cls, calibration: Mapping[str, Any]) -> "Settings":
        known = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in calibration.items() if key in known})


@dataclass
class CalibrationTable:
    """Đặc trưng và dự đoán trên ảnh sạch và các điều kiện suy giảm. Chỉ số điều kiện 0 là ảnh sạch, chỉ số hành động 0 là `keep`."""

    conditions: list[str]
    families: list[str]
    actions: list[str]
    features: np.ndarray  # C × N × F, theo thứ tự FEATURES
    predictions: np.ndarray  # A × C × N, nhãn dự đoán
    probabilities: np.ndarray | None  # A × C × N × K
    labels: np.ndarray  # N
    paths: list[str]
    classes: list[str]

    def __post_init__(self) -> None:
        if self.conditions[0] != CLEAN or self.actions[0] != KEEP:
            raise ValueError("Điều kiện đầu phải là `clean` và hành động đầu phải là `keep`.")
        count, images = len(self.conditions), len(self.labels)
        if self.features.shape != (count, images, len(FEATURES)):
            raise ValueError(f"Đặc trưng có dạng {self.features.shape}, cần {(count, images, len(FEATURES))}.")
        if self.predictions.shape != (len(self.actions), count, images):
            raise ValueError(f"Dự đoán có dạng {self.predictions.shape}, cần {(len(self.actions), count, images)}.")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def feature(self, name: str) -> np.ndarray:
        """Giá trị một đặc trưng, C × N."""
        return self.features[:, :, FEATURES.index(name)]

    def subset(self, indices: np.ndarray) -> "CalibrationTable":
        """Bảng chỉ gồm các ảnh được chọn, giữ nguyên thứ tự điều kiện và hành động."""
        return CalibrationTable(
            self.conditions, self.families, self.actions, self.features[:, indices], self.predictions[:, :, indices],
            None if self.probabilities is None else self.probabilities[:, :, indices],
            self.labels[indices], [self.paths[i] for i in indices], self.classes,
        )

    def save(self, path: str | Path) -> None:
        arrays: dict[str, Any] = {
            "features": self.features, "predictions": self.predictions, "labels": self.labels, "paths": np.array(self.paths),
            "meta_json": np.array(json.dumps({"conditions": self.conditions, "families": self.families, "actions": self.actions, "classes": self.classes, "features": list(FEATURES)}, ensure_ascii=False)),
        }
        if self.probabilities is not None:
            arrays["probabilities"] = self.probabilities
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationTable":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            if meta["features"] != list(FEATURES):
                raise ValueError("Thứ tự đặc trưng trong file khác với src.selector.features.FEATURES.")
            return cls(
                meta["conditions"], meta["families"], meta["actions"], data["features"], data["predictions"],
                data["probabilities"] if "probabilities" in data else None, data["labels"], [str(p) for p in data["paths"]], meta["classes"],
            )


# --- Giai đoạn A: thu thập ---------------------------------------------------------------------------------------------------------------


def collect(
    model: nn.Module,
    cache: Any,
    conditions: Mapping[str, Mapping[str, Any]],
    actions: Mapping[str, Mapping[str, Any] | None],
    pipeline_cfg: Mapping[str, Any],
    device: torch.device,
    workers: int = 1,
    progress: Callable[[str], None] = print,
    on_condition_done: Callable[[CalibrationTable], None] | None = None,
) -> CalibrationTable:
    """Đặc trưng và dự đoán của reference model cho mọi hành động, trên ảnh sạch và từng điều kiện.

    `cache` là `ImageCache` của processing_benchmark trên validation (thuộc tính labels, paths, classes và phương thức get, release).
    Đặc trưng tính trên ảnh suy giảm chưa xử lý, đúng thứ bộ chọn nhìn thấy khi triển khai.
    """
    names = [CLEAN, *conditions]
    action_names = [KEEP, *[name for name in actions if name != KEEP]]
    features, predictions, probabilities = [], [], []
    for position, name in enumerate(names):
        images = cache.get(name)
        features.append(extract_batch(images, workers))
        per_action = []
        for action in action_names:
            spec = actions[action]
            processor = None if spec is None else (lambda image, spec=spec: apply_action(image, spec))
            per_action.append(predict_probabilities(model, images, processor, pipeline_cfg, device, workers=workers))
        predictions.append(np.stack([p.argmax(axis=1) for p in per_action]).astype(np.int8))
        probabilities.append(np.stack(per_action).astype(np.float16))
        cache.release(name)
        progress(f"{name}: {len(images)} ảnh, {len(action_names)} hành động")
        if on_condition_done is not None and position < len(names) - 1:
            on_condition_done(_assemble(cache, conditions, names[: position + 1], action_names, features, predictions, probabilities))
    return _assemble(cache, conditions, names, action_names, features, predictions, probabilities)


def _assemble(cache: Any, conditions: Mapping[str, Mapping[str, Any]], names: Sequence[str], action_names: Sequence[str],
              features: Sequence[np.ndarray], predictions: Sequence[np.ndarray], probabilities: Sequence[np.ndarray]) -> CalibrationTable:
    return CalibrationTable(
        conditions=list(names),
        families=[CLEAN if name == CLEAN else str(conditions[name]["family"]) for name in names],
        actions=list(action_names),
        features=np.stack(features).astype(np.float32),
        predictions=np.stack(predictions, axis=1),
        probabilities=np.stack(probabilities, axis=1),
        labels=np.asarray(cache.labels, dtype=np.int64),
        paths=list(cache.paths),
        classes=list(cache.classes),
    )


# --- Giai đoạn B: chấm điểm và tìm ngưỡng ----------------------------------------------------------------------------------------------------


def macro_f1_rows(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    """Macro F1 của từng hàng trong `predictions` (C × N), cùng định nghĩa với sklearn (mọi lớp, zero_division = 0)."""
    count, images = predictions.shape
    offset = (np.arange(count)[:, None] * num_classes * num_classes)
    index = offset + labels[None, :].astype(np.int64) * num_classes + predictions.astype(np.int64)
    matrix = np.bincount(index.ravel(), minlength=count * num_classes * num_classes).reshape(count, num_classes, num_classes)
    true_positive = np.diagonal(matrix, axis1=1, axis2=2).astype(np.float64)
    false_positive = matrix.sum(axis=1) - true_positive
    false_negative = matrix.sum(axis=2) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
    return f1.mean(axis=1)


def gather_predictions(table: CalibrationTable, assignment: np.ndarray) -> np.ndarray:
    """Dự đoán C × N khi mỗi ảnh dùng hành động có chỉ số cho trước trong `assignment` (C × N)."""
    count, images = assignment.shape
    return table.predictions[assignment, np.arange(count)[:, None], np.arange(images)[None, :]]


def score(table: CalibrationTable, assignment: np.ndarray) -> np.ndarray:
    """Macro F1 của từng điều kiện (độ dài C, phần tử đầu là ảnh sạch) theo một phép gán hành động."""
    return macro_f1_rows(table.labels, gather_predictions(table, assignment), table.num_classes)


def assign_selector(table: CalibrationTable, selector: RuleSelector) -> np.ndarray:
    """Chỉ số hành động mà bộ chọn chọn cho từng ảnh ở từng điều kiện (C × N)."""
    count, images = table.features.shape[:2]
    flat = {name: table.feature(name).reshape(-1) for name in FEATURES}
    chosen = selector.select_batch(flat).reshape(count, images)
    lookup = {name: index for index, name in enumerate(table.actions)}
    unknown = set(np.unique(chosen)) - set(lookup)
    if unknown:
        raise ValueError(f"Bộ chọn chọn hành động không có trong bảng: {sorted(unknown)}.")
    return np.vectorize(lookup.__getitem__)(chosen).astype(np.int64)


def threshold_grid(pooled: np.ndarray, clean: np.ndarray, points: int) -> np.ndarray:
    """Lưới ngưỡng theo phân vị của đặc trưng trên mọi điều kiện, cộng thêm phân vị cao của ảnh sạch để ngưỡng nằm sát biên của ảnh sạch."""
    grid = np.quantile(pooled, np.linspace(0.01, 0.99, points))
    grid = np.concatenate([grid, np.quantile(clean, [0.5, 0.9, 0.95, 0.99, 0.999])])
    return np.unique(grid.astype(np.float32))


def _holds(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    return values >= threshold if op == ">=" else values <= threshold


def fit(
    table: CalibrationTable,
    rules: Sequence[Mapping[str, Any]],
    settings: Settings,
    dropped_groups: Sequence[str] = (),
) -> tuple[RuleSelector, dict[str, Any]]:
    """Chọn hành động, đặc trưng và ngưỡng của từng luật, lần lượt theo thứ tự, mỗi luật tối ưu mục tiêu với các luật trước đã cố định.

    `dropped_groups` loại các nhóm đặc trưng khỏi tập ứng viên, dùng cho ablation (task 5.4). Trả về bộ chọn và nhật ký tìm kiếm.
    """
    count, images = table.features.shape[:2]
    action_index = {name: i for i, name in enumerate(table.actions)}
    assignment = np.zeros((count, images), dtype=np.int64)
    open_mask = np.ones((count, images), dtype=bool)
    none_scores = score(table, assignment)
    clean_floor = none_scores[0] - settings.max_clean_f1_drop
    objective = float(none_scores[1:].mean())
    log: dict[str, Any] = {"none_objective": objective, "none_clean_f1": float(none_scores[0]), "clean_f1_floor": float(clean_floor), "rules": []}
    chosen: list[Rule] = []

    for spec in rules:
        entry: dict[str, Any] = {"name": spec["name"], "evaluated": 0, "accepted": False}
        log["rules"].append(entry)
        candidates = drop_groups(spec["candidates"], dropped_groups)
        guards = drop_groups(spec.get("guard", []), dropped_groups)
        if not candidates:
            entry["reason"] = "không còn đặc trưng ứng viên sau khi bỏ nhóm"
            continue
        guard_masks = _guard_options(table, guards, settings.guard_points)
        best: dict[str, Any] | None = None
        for action in spec["actions"]:
            for candidate in candidates:
                values = table.feature(candidate["feature"])
                for threshold in threshold_grid(values.ravel(), values[0], settings.threshold_points):
                    primary = _holds(values, candidate["op"], float(threshold))
                    for guard_clause, guard_mask in guard_masks:
                        fired = primary & open_mask if guard_mask is None else primary & guard_mask & open_mask
                        if fired[0].mean() > settings.max_clean_trigger_rate:
                            continue
                        trial = np.where(fired, action_index[action], assignment)
                        scores = score(table, trial)
                        entry["evaluated"] += 1
                        if scores[0] < clean_floor:
                            continue
                        value = float(scores[1:].mean())
                        key = (value, -int(fired.sum()))
                        if best is None or key > best["key"]:
                            best = {"key": key, "value": value, "action": action, "candidate": candidate, "threshold": float(threshold),
                                    "guard": guard_clause, "fired": fired, "clean_f1": float(scores[0]), "clean_trigger_rate": float(fired[0].mean())}
        if best is None:
            entry["reason"] = "không bộ ngưỡng nào thỏa ràng buộc trên ảnh sạch"
            continue
        extras = drop_groups(spec.get("extra_guard", []), dropped_groups)
        if extras:
            entry["objective_before_extra_guard"] = best["value"]
            best = _add_extra_guard(table, best, extras, action_index, assignment, clean_floor, settings, entry)
        gain = best["value"] - objective
        entry.update({"best_objective": best["value"], "gain": gain, "action": best["action"], "feature": best["candidate"]["feature"],
                      "op": best["candidate"]["op"], "threshold": best["threshold"], "guard": best["guard"]})
        if gain < settings.min_gain:
            entry["reason"] = f"tăng mục tiêu {gain:+.4f}, dưới min_gain {settings.min_gain}"
            continue
        clauses = [Clause(best["candidate"]["feature"], best["candidate"]["op"], best["threshold"])]
        if best["guard"] is not None:
            clauses.append(Clause(best["guard"]["feature"], best["guard"]["op"], best["guard"]["threshold"]))
        if best.get("extra_guard") is not None:
            clauses.append(Clause(best["extra_guard"]["feature"], best["extra_guard"]["op"], best["extra_guard"]["threshold"]))
        chosen.append(Rule(str(spec["name"]), tuple(clauses), best["action"]))
        assignment = np.where(best["fired"], action_index[best["action"]], assignment)
        open_mask &= ~best["fired"]
        objective = best["value"]
        entry.update({"accepted": True, "clean_f1": best["clean_f1"], "clean_trigger_rate": best["clean_trigger_rate"], "triggered_images": int(best["fired"].sum())})
    log["objective"] = objective
    return RuleSelector(chosen), log


def _add_extra_guard(
    table: CalibrationTable,
    best: dict[str, Any],
    extras: Sequence[Mapping[str, str]],
    action_index: Mapping[str, int],
    assignment: np.ndarray,
    clean_floor: float,
    settings: Settings,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Thử thêm một mệnh đề chặn thứ hai vào luật vừa tìm được. Chỉ giữ nếu mục tiêu tăng thêm ít nhất `min_gain`.

    Đây là bước chỉnh sau khi lần chạy đầu cho thấy luật mờ làm nét nhầm ảnh bóng đổ và tương phản thấp. Chọn một mệnh đề duy nhất
    từ `extra_guard` của luật, trên cùng ràng buộc ảnh sạch. Kết quả ghi vào nhật ký của luật.
    """
    runner: dict[str, Any] | None = None
    for clause, mask in _guard_options(table, extras, settings.guard_points):
        if clause is None:
            continue
        fired = best["fired"] & mask
        scores = score(table, np.where(fired, action_index[best["action"]], assignment))
        entry["evaluated"] += 1
        if scores[0] < clean_floor:
            continue
        value = float(scores[1:].mean())
        if runner is None or (value, -int(fired.sum())) > (runner["value"], -int(runner["fired"].sum())):
            runner = {"value": value, "clause": clause, "fired": fired, "clean_f1": float(scores[0]), "clean_trigger_rate": float(fired[0].mean())}
    if runner is None or runner["value"] < best["value"] + settings.min_gain:
        entry["extra_guard"] = None
        entry["extra_guard_gain"] = None if runner is None else runner["value"] - best["value"]
        return best
    entry["extra_guard"] = runner["clause"]
    entry["extra_guard_gain"] = runner["value"] - best["value"]
    return {**best, "value": runner["value"], "fired": runner["fired"], "clean_f1": runner["clean_f1"],
            "clean_trigger_rate": runner["clean_trigger_rate"], "extra_guard": runner["clause"]}


def _guard_options(table: CalibrationTable, guards: Sequence[Mapping[str, str]], points: int) -> list[tuple[dict[str, Any] | None, np.ndarray | None]]:
    """Các lựa chọn mệnh đề chặn: không có mệnh đề chặn, hoặc một đặc trưng với một ngưỡng trên lưới thưa."""
    options: list[tuple[dict[str, Any] | None, np.ndarray | None]] = [(None, None)]
    for guard in guards:
        values = table.feature(guard["feature"])
        for threshold in np.unique(np.quantile(values.ravel(), np.linspace(0.05, 0.95, points)).astype(np.float32)):
            options.append(({"feature": guard["feature"], "op": guard["op"], "threshold": float(threshold)}, _holds(values, guard["op"], float(threshold))))
    return options


# --- Báo cáo và kiểm tra độ ổn định --------------------------------------------------------------------------------------------------------------


def report(table: CalibrationTable, selector: RuleSelector) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bảng theo điều kiện (F1 không xử lý, F1 bộ chọn, tỉ lệ kích hoạt) và tóm tắt: bộ chọn, từng hành động cố định, và cận trên theo ảnh."""
    count, images = table.features.shape[:2]
    chosen = assign_selector(table, selector)
    none = score(table, np.zeros((count, images), dtype=np.int64))
    selected = score(table, chosen)
    frame = pd.DataFrame({
        "condition": table.conditions, "family": table.families, "f1_none": none, "f1_selector": selected, "delta": selected - none,
        "trigger_rate": (chosen != 0).mean(axis=1),
    })
    for index, action in enumerate(table.actions[1:], start=1):
        frame[f"share_{action}"] = (chosen == index).mean(axis=1)
    fixed = {action: float(score(table, np.full((count, images), index))[1:].mean()) for index, action in enumerate(table.actions)}
    clean_fixed = {action: float(score(table, np.full((count, images), index))[0]) for index, action in enumerate(table.actions)}
    # Cận trên: với mỗi ảnh và điều kiện chọn một hành động cho dự đoán đúng nếu có (dùng nhãn, không triển khai được).
    correct = table.predictions == table.labels[None, None, :]
    oracle_pred = np.where(correct.any(axis=0), table.labels[None, :], table.predictions[0])
    summary = {
        "objective_none": float(none[1:].mean()), "objective_selector": float(selected[1:].mean()), "gain": float((selected - none)[1:].mean()),
        "clean_f1_none": float(none[0]), "clean_f1_selector": float(selected[0]), "clean_trigger_rate": float(frame["trigger_rate"].iloc[0]),
        "conditions_better": int((frame["delta"].iloc[1:] > 1e-9).sum()), "conditions_worse": int((frame["delta"].iloc[1:] < -1e-9).sum()),
        "fixed_action_objective": fixed, "fixed_action_clean_f1": clean_fixed,
        "image_level_upper_bound": float(macro_f1_rows(table.labels, oracle_pred, table.num_classes)[1:].mean()),
    }
    return frame, summary


def cross_validate(
    table: CalibrationTable,
    rules: Sequence[Mapping[str, Any]],
    settings: Settings,
    folds: int,
    seed: int,
    dropped_groups: Sequence[str] = (),
) -> dict[str, Any]:
    """Hiệu chỉnh trên các phần ảnh còn lại và chấm trên phần giữ lại, lặp theo từng phần (chia theo lớp).

    Cho biết bộ ngưỡng có tổng quát hóa sang ảnh chưa dùng để hiệu chỉnh hay không. Các họ suy giảm vẫn là các họ đã thấy.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (train, held) in enumerate(splitter.split(np.zeros(len(table.labels)), table.labels)):
        selector, log = fit(table.subset(train), rules, settings, dropped_groups)
        _, summary = report(table.subset(held), selector)
        rows.append({
            "fold": fold, "rules": [rule.to_dict() for rule in selector.rules], "calibration_objective": log["objective"],
            "heldout_gain": summary["gain"], "heldout_clean_f1_change": summary["clean_f1_selector"] - summary["clean_f1_none"],
            "heldout_trigger_rate_clean": summary["clean_trigger_rate"],
        })
    gains = np.array([row["heldout_gain"] for row in rows])
    return {
        "folds": rows, "heldout_gain_mean": float(gains.mean()), "heldout_gain_std": float(gains.std(ddof=1)) if len(gains) > 1 else 0.0,
        "folds_with_positive_gain": int((gains > 0).sum()),
    }


def save_artifacts(
    output_dir: str | Path,
    selector: RuleSelector,
    search_log: Mapping[str, Any],
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    stability: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Path:
    """Ghi bộ chọn đã hiệu chỉnh, nhật ký tìm kiếm và bảng theo điều kiện. Trả về đường dẫn file bộ chọn.

    `locked` luôn là false khi mới sinh. Khóa là một quyết định sau khi nhóm xem lại kết quả, bằng cách đặt true và commit trước khi mở test.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = selector.to_dict()
    record = {
        "task": "5.3", "split": config["split"], "locked": False, "seed": config["seed"],
        "selector": body, "selector_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "context": dict(context),
    }
    path = out / "selector_calibrated.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log = {"settings": dict(config["calibration"]), "search": search_log, "summary": summary, "stability": stability}
    (out / "calibration_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    frame.to_csv(out / "calibration_conditions_val.csv", index=False)
    return path

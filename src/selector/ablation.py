"""Ablation của bộ chọn xử lý ảnh thích ứng (task 5.4). Chỉ dùng validation, không mở tập test.

Câu hỏi: mức tăng của bộ chọn (task 5.3) đến từ đâu? Mỗi biến thể dưới đây thay đổi một thành phần so với bộ chọn đầy đủ,
hiệu chỉnh lại trên cùng bảng đặc trưng và dự đoán (`CalibrationTable`), nên không chạy lại model.

Các nhóm biến thể (cột `kind`):
- reference: không xử lý, từng hành động cố định, cận trên theo ảnh (oracle, dùng nhãn nên không triển khai được).
- full: bộ chọn đầy đủ hiệu chỉnh lại, và bộ chọn đã khóa ở 5.3 để đối chiếu.
- feature_group: bỏ một nhóm đặc trưng khỏi mọi luật (ứng viên và mệnh đề chặn), rồi hiệu chỉnh lại.
- action: bỏ một hành động xử lý, rồi hiệu chỉnh lại.
- technique: bỏ một kỹ thuật (mọi cấu hình của nó), rồi hiệu chỉnh lại.
- rule: giữ một luật của bộ chọn đầy đủ, bỏ luật còn lại (không hiệu chỉnh lại).
- guard: giữ mệnh đề chính của từng luật, bỏ mệnh đề chặn (không hiệu chỉnh lại).

Mục tiêu và ràng buộc giống task 5.3: macro F1 trung bình 42 điều kiện suy giảm, ảnh sạch không giảm quá `max_clean_f1_drop`.
Khoảng tin cậy dùng bootstrap ghép theo ảnh: mọi biến thể được chấm trên cùng các mẫu lặp, nên chênh lệch giữa hai biến thể có phương sai nhỏ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.selector.calibrate import (
    CalibrationTable,
    Settings,
    assign_selector,
    cross_validate,
    fit,
    gather_predictions,
    macro_f1_rows,
)
from src.selector.features import FEATURE_GROUPS
from src.selector.rules import Rule, RuleSelector

KINDS = ("reference", "full", "feature_group", "action", "technique", "rule", "guard", "oracle")


@dataclass
class Variant:
    """Một biến thể: dự đoán C × N của nó, và bộ chọn nếu có."""

    name: str
    kind: str
    label: str
    predictions: np.ndarray
    selector: RuleSelector | None = None
    search_log: Mapping[str, Any] | None = None


# --- Dựng biến thể -------------------------------------------------------------------------------------------------------------------------


def restrict_actions(rules: Sequence[Mapping[str, Any]], removed: Sequence[str]) -> list[dict[str, Any]]:
    """Luật với các hành động bị bỏ được lọc khỏi danh sách `actions`. Luật hết hành động thì bị loại."""
    removed_set = set(removed)
    kept = []
    for spec in rules:
        actions = [action for action in spec["actions"] if action not in removed_set]
        if actions:
            kept.append({**spec, "actions": actions})
    return kept


def strip_guards(selector: RuleSelector) -> RuleSelector:
    """Bộ chọn chỉ giữ mệnh đề đầu (mệnh đề chính) của mỗi luật."""
    return RuleSelector([Rule(rule.name, rule.clauses[:1], rule.action) for rule in selector.rules], selector.default)


def _fixed_predictions(table: CalibrationTable, index: int) -> np.ndarray:
    return table.predictions[index]


def _selector_variant(table: CalibrationTable, name: str, kind: str, label: str, selector: RuleSelector,
                      log: Mapping[str, Any] | None = None) -> Variant:
    return Variant(name, kind, label, gather_predictions(table, assign_selector(table, selector)), selector, log)


def oracle_predictions(table: CalibrationTable) -> np.ndarray:
    """Cận trên theo ảnh: đúng nếu có ít nhất một hành động dự đoán đúng. Dùng nhãn nên chỉ để so sánh."""
    correct = table.predictions == table.labels[None, None, :]
    return np.where(correct.any(axis=0), table.labels[None, :], table.predictions[0])


def build_variants(
    table: CalibrationTable,
    rules: Sequence[Mapping[str, Any]],
    settings: Settings,
    techniques: Mapping[str, str],
    locked: RuleSelector | None = None,
) -> list[Variant]:
    """Dựng mọi biến thể. `techniques` ánh xạ tên hành động sang kỹ thuật (unsharp, bilateral) để bỏ theo kỹ thuật."""
    variants = [Variant("none", "reference", "Không xử lý", table.predictions[0])]
    for index, action in enumerate(table.actions[1:], start=1):
        variants.append(Variant(f"fixed:{action}", "reference", f"Luôn dùng {action}", _fixed_predictions(table, index)))

    full, full_log = fit(table, rules, settings)
    variants.append(_selector_variant(table, "full", "full", "Bộ chọn đầy đủ", full, full_log))
    if locked is not None:
        variants.append(_selector_variant(table, "locked", "full", "Bộ chọn đã khóa (5.3)", locked))

    for group in FEATURE_GROUPS:
        selector, log = fit(table, rules, settings, dropped_groups=[group])
        variants.append(_selector_variant(table, f"drop_group:{group}", "feature_group", f"Bỏ nhóm {group}", selector, log))

    for action in table.actions[1:]:
        selector, log = fit(table, restrict_actions(rules, [action]), settings)
        variants.append(_selector_variant(table, f"drop_action:{action}", "action", f"Bỏ {action}", selector, log))

    for technique in sorted(set(techniques.values())):
        removed = [action for action, name in techniques.items() if name == technique]
        selector, log = fit(table, restrict_actions(rules, removed), settings)
        variants.append(_selector_variant(table, f"drop_technique:{technique}", "technique", f"Bỏ kỹ thuật {technique}", selector, log))

    for rule in full.rules:
        selector = RuleSelector([rule], full.default)
        variants.append(_selector_variant(table, f"only_rule:{rule.name}", "rule", f"Chỉ luật {rule.name}", selector))
    if any(len(rule.clauses) > 1 for rule in full.rules):
        variants.append(_selector_variant(table, "no_guards", "guard", "Bỏ mệnh đề chặn", strip_guards(full)))

    variants.append(Variant("oracle", "oracle", "Cận trên theo ảnh", oracle_predictions(table)))
    return variants


# --- Chấm điểm -------------------------------------------------------------------------------------------------------------------------------


def objective_of(f1: np.ndarray) -> float:
    """Mục tiêu: macro F1 trung bình các điều kiện suy giảm (bỏ phần tử đầu là ảnh sạch)."""
    return float(f1[1:].mean())


def bootstrap_indices(images: int, resamples: int, seed: int) -> np.ndarray:
    """Chỉ số ảnh của các mẫu bootstrap (resamples × images), dùng chung cho mọi biến thể để phép so sánh có ghép cặp."""
    return np.random.default_rng(seed).integers(0, images, size=(resamples, images))


def objective_samples(table: CalibrationTable, predictions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Mục tiêu trên từng mẫu bootstrap."""
    values = np.empty(len(indices))
    for row, chosen in enumerate(indices):
        values[row] = objective_of(macro_f1_rows(table.labels[chosen], predictions[:, chosen], table.num_classes))
    return values


def _interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(low), float(high)


def _describe(selector: RuleSelector | None) -> str:
    if selector is None:
        return ""
    parts = []
    for rule in selector.rules:
        clauses = " VÀ ".join(f"{c.feature} {c.op} {c.threshold:.4g}" for c in rule.clauses)
        parts.append(f"{rule.name}: {clauses} -> {rule.action}")
    return " | ".join(parts) if parts else "(không luật)"


def summarize(
    table: CalibrationTable,
    variants: Sequence[Variant],
    resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bảng một dòng mỗi biến thể: mục tiêu, chênh lệch so với không xử lý và so với bộ chọn đầy đủ (kèm khoảng tin cậy 95%), ảnh sạch, kích hoạt."""
    by_name = {variant.name: variant for variant in variants}
    indices = bootstrap_indices(len(table.labels), resamples, seed)
    samples = {variant.name: objective_samples(table, variant.predictions, indices) for variant in variants}
    none_samples, full_samples = samples["none"], samples["full"]
    none_f1 = macro_f1_rows(table.labels, by_name["none"].predictions, table.num_classes)
    full_f1 = macro_f1_rows(table.labels, by_name["full"].predictions, table.num_classes)

    rows = []
    for variant in variants:
        f1 = macro_f1_rows(table.labels, variant.predictions, table.num_classes)
        changed = variant.predictions != table.predictions[0]
        low, high = _interval(samples[variant.name])
        gain_low, gain_high = _interval(samples[variant.name] - none_samples)
        delta_low, delta_high = _interval(samples[variant.name] - full_samples)
        rows.append({
            "variant": variant.name, "kind": variant.kind, "label": variant.label,
            "objective": objective_of(f1), "objective_ci_low": low, "objective_ci_high": high,
            "gain_vs_none": objective_of(f1) - objective_of(none_f1), "gain_ci_low": gain_low, "gain_ci_high": gain_high,
            "delta_vs_full": objective_of(f1) - objective_of(full_f1), "delta_ci_low": delta_low, "delta_ci_high": delta_high,
            "clean_f1": float(f1[0]), "clean_f1_change": float(f1[0] - none_f1[0]),
            "changed_share_clean": float(changed[0].mean()), "changed_share_degraded": float(changed[1:].mean()),
            "conditions_better": int((f1[1:] > none_f1[1:] + 1e-9).sum()), "conditions_worse": int((f1[1:] < none_f1[1:] - 1e-9).sum()),
            "rules": _describe(variant.selector),
        })
    return pd.DataFrame(rows)


def add_cross_validation(
    summary: pd.DataFrame,
    table: CalibrationTable,
    rules: Sequence[Mapping[str, Any]],
    settings: Settings,
    techniques: Mapping[str, str],
    folds: int,
    seed: int,
) -> pd.DataFrame:
    """Thêm mức tăng trên phần ảnh giữ lại (cross-validation) cho các biến thể có hiệu chỉnh lại. Biến thể khác để trống."""
    specs: dict[str, tuple[Sequence[Mapping[str, Any]], Sequence[str]]] = {"full": (rules, ())}
    for group in FEATURE_GROUPS:
        specs[f"drop_group:{group}"] = (rules, [group])
    for action in table.actions[1:]:
        specs[f"drop_action:{action}"] = (restrict_actions(rules, [action]), ())
    for technique in sorted(set(techniques.values())):
        removed = [action for action, name in techniques.items() if name == technique]
        specs[f"drop_technique:{technique}"] = (restrict_actions(rules, removed), ())

    result = summary.copy()
    result["cv_gain_mean"] = np.nan
    result["cv_gain_std"] = np.nan
    result["cv_positive_folds"] = np.nan
    for name, (variant_rules, dropped) in specs.items():
        stability = cross_validate(table, variant_rules, settings, folds, seed, dropped_groups=dropped)
        mask = result["variant"] == name
        result.loc[mask, "cv_gain_mean"] = stability["heldout_gain_mean"]
        result.loc[mask, "cv_gain_std"] = stability["heldout_gain_std"]
        result.loc[mask, "cv_positive_folds"] = stability["folds_with_positive_gain"]
    return result


def condition_table(table: CalibrationTable, variants: Sequence[Variant]) -> pd.DataFrame:
    """Macro F1 theo điều kiện (dạng dài): condition, family, variant, f1."""
    frames = []
    for variant in variants:
        f1 = macro_f1_rows(table.labels, variant.predictions, table.num_classes)
        frames.append(pd.DataFrame({"condition": table.conditions, "family": table.families, "variant": variant.name, "f1": f1}))
    return pd.concat(frames, ignore_index=True)


def family_gain(conditions: pd.DataFrame) -> pd.DataFrame:
    """Macro F1 trung bình ba mức của từng họ, trừ đi F1 không xử lý. Dòng là biến thể, cột là họ suy giảm."""
    degraded = conditions[conditions["condition"] != "clean"]
    pivot = degraded.pivot_table(index="variant", columns="family", values="f1", aggfunc="mean", sort=False)
    return pivot.sub(pivot.loc["none"], axis=1)


def save_results(output_dir: str | Path, summary: pd.DataFrame, conditions: pd.DataFrame, variants: Sequence[Variant],
                 context: Mapping[str, Any]) -> Path:
    """Ghi bảng tóm tắt, bảng theo điều kiện, các bộ chọn của biến thể và ngữ cảnh. Trả về đường dẫn bảng tóm tắt."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ablation_summary_val.csv"
    summary.to_csv(path, index=False)
    conditions.to_csv(out / "ablation_conditions_val.csv", index=False)
    selectors = {variant.name: variant.selector.to_dict() for variant in variants if variant.selector is not None}
    record = {"task": "5.4", "split": "val", "context": dict(context), "selectors": selectors}
    (out / "ablation_selectors.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path

"""Giai đoạn B của bản đồ phục hồi (task 4.5): bảng recovery, chi phí ảnh sạch, oracle và so sánh với model huấn luyện lại, trên CPU.

Đầu vào là dự đoán từng ảnh của giai đoạn A (recovery_map.py), dự đoán của reference chưa xử lý ở task 4.2 (để kiểm tra tái hiện), cấu hình đã khóa của
task 4.3 và 4.4, và dự đoán test của model huấn luyện lại. Khoảng tin cậy bootstrap theo ảnh, McNemar chính xác ghép cặp, hiệu chỉnh Holm.
Chạy: python -m src.evaluation.recovery_report configs/recovery_map.yaml <file .npz của giai đoạn A> [thư mục kết quả]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure

from src.evaluation.processing_plots import plot_matrix_heatmap
from src.evaluation.recovery_map import key_of, load_selected
from src.evaluation.recovery_stats import PairedBootstrap, holm, mcnemar_exact, normalized_recovery

ROOT = Path(__file__).resolve().parents[2]
SEVERITIES = (1, 2, 3)


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def recalls(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    """Recall từng lớp."""
    counts = np.bincount(labels, minlength=num_classes)
    hits = np.bincount(labels[predictions == labels], minlength=num_classes)
    return np.where(counts > 0, hits / np.maximum(counts, 1), 0.0)


class Run:
    """Dự đoán từng ảnh của giai đoạn A, đọc từ file .npz."""

    def __init__(self, path: str | Path) -> None:
        data = np.load(_abs(path), allow_pickle=False)
        self.labels = data["labels"].astype(np.int64)
        self.paths = [str(item) for item in data["paths"]]
        self.rows: list[dict[str, Any]] = json.loads(str(data["rows_json"]))
        skip = {"labels", "paths", "rows_json"}
        self._predictions = {name: data[name].astype(np.int64) for name in data.files if name not in skip and not name.endswith("__probs")}

    def predictions(self, technique: str, config_id: str, condition: str) -> np.ndarray:
        return self._predictions[key_of(technique, config_id, condition)]

    def has(self, technique: str, config_id: str, condition: str) -> bool:
        return key_of(technique, config_id, condition) in self._predictions


def recovery_cells(run: Run, bootstrap: PairedBootstrap, stats: Mapping[str, Any], classes: list[str]) -> pd.DataFrame:
    """Một dòng cho mỗi (kỹ thuật, cấu hình, điều kiện suy giảm): recovery, khoảng tin cậy, McNemar và điểm sau xử lý."""
    labels = run.labels
    clean = run.predictions("none", "none", "clean")
    rows = []
    for row in run.rows:
        if row["technique"] == "none" or row["condition"] == "clean":
            continue
        corrupted = run.predictions("none", "none", row["condition"])
        processed = run.predictions(row["technique"], row["config_id"], row["condition"])
        result = normalized_recovery(bootstrap, clean, corrupted, processed, float(stats["min_degradation"]), confidence=float(stats["confidence"]))
        test = mcnemar_exact(corrupted == labels, processed == labels)
        recall = recalls(labels, processed, len(classes))
        family, severity = row["condition"].rsplit("_s", 1)
        record = {
            "technique": row["technique"], "config_id": row["config_id"], "role": row["role"], "condition": row["condition"],
            "family": family, "severity": int(severity), "suitable": bool(row["suitable"]),
            "f1_corrupted": bootstrap.macro_f1(corrupted), "f1_processed": bootstrap.macro_f1(processed),
            "balanced_accuracy": float(recall.mean()), **result, "gains": test["gains"], "harms": test["harms"], "p": test["p"],
        }
        record.update({f"recall_{name}": float(value) for name, value in zip(classes, recall)})
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame["p"].tolist())
    alpha = float(stats["alpha"])
    frame["gain_significant"] = (frame["recovery"] > 0) & (frame["p_holm"] < alpha)
    frame["harm_significant"] = (frame["recovery"] < 0) & (frame["p_holm"] < alpha)
    return frame


def clean_cost(run: Run, bootstrap: PairedBootstrap, cells: pd.DataFrame, stats: Mapping[str, Any], classes: list[str]) -> pd.DataFrame:
    """Thay đổi macro F1 và recall từng lớp trên ảnh sạch khi áp từng cấu hình, kèm khoảng tin cậy và McNemar."""
    labels = run.labels
    clean = run.predictions("none", "none", "clean")
    base_recall = recalls(labels, clean, len(classes))
    base_f1, base_samples = bootstrap.macro_f1(clean), bootstrap.macro_f1_samples(clean)
    used = cells[cells["role"] == "family"].groupby(["technique", "config_id"])["family"].apply(lambda fam: sorted(set(fam[cells.loc[fam.index, "suitable"]]))).to_dict()
    rows = []
    for row in run.rows:
        if row["role"] != "clean":
            continue
        processed = run.predictions(row["technique"], row["config_id"], "clean")
        samples = bootstrap.macro_f1_samples(processed) - base_samples
        low, high = bootstrap.interval(samples, float(stats["confidence"]))
        test = mcnemar_exact(clean == labels, processed == labels)
        recall = recalls(labels, processed, len(classes))
        record = {
            "technique": row["technique"], "config_id": row["config_id"], "used_for_families": ",".join(used.get((row["technique"], row["config_id"]), [])),
            "f1_clean": base_f1, "f1_processed": bootstrap.macro_f1(processed), "delta": bootstrap.macro_f1(processed) - base_f1,
            "delta_low": low, "delta_high": high, "gains": test["gains"], "harms": test["harms"], "p": test["p"],
        }
        record.update({f"delta_recall_{name}": float(value - base) for name, value, base in zip(classes, recall, base_recall)})
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame["p"].tolist())
    return frame


def family_summary(cells: pd.DataFrame, cost: pd.DataFrame, run: Run, bootstrap: PairedBootstrap, stats: Mapping[str, Any], evidence: Mapping[str, Any]) -> pd.DataFrame:
    """Theo (kỹ thuật, họ): recovery trung bình ba mức kèm khoảng tin cậy, số mức có ý nghĩa, chi phí ảnh sạch và kết luận "có bằng chứng"."""
    cost_by_config = cost.set_index(["technique", "config_id"])
    rows = []
    for (technique, family), group in cells[cells["role"] == "family"].groupby(["technique", "family"], sort=False):
        group = group.sort_values("severity")
        samples = np.mean(
            [bootstrap.macro_f1_samples(run.predictions(technique, level["config_id"], level["condition"])) - bootstrap.macro_f1_samples(run.predictions("none", "none", level["condition"])) for _, level in group.iterrows()],
            axis=0,
        )
        low, high = bootstrap.interval(samples, float(stats["confidence"]))
        config_id = group["config_id"].iloc[0]
        price = cost_by_config.loc[(technique, config_id)]
        suitable = bool(group["suitable"].iloc[0])
        significant = int(group["gain_significant"].sum())
        has_evidence = suitable and low > 0 and significant >= int(evidence["min_significant_levels"])
        rows.append({
            "technique": technique, "family": family, "config_id": config_id, "suitable": suitable,
            "f1_corrupted_mean": float(group["f1_corrupted"].mean()), "f1_processed_mean": float(group["f1_processed"].mean()),
            "recovery_mean": float(group["recovery"].mean()), "recovery_low": low, "recovery_high": high,
            "normalized_recovery_mean": float(group["normalized_recovery"].mean()) if group["normalized_recovery"].notna().any() else float("nan"),
            "levels_gain_significant": significant, "levels_harm_significant": int(group["harm_significant"].sum()),
            "clean_delta": float(price["delta"]), "clean_delta_low": float(price["delta_low"]), "clean_delta_high": float(price["delta_high"]),
            "evidence": bool(has_evidence), "costly": bool(price["delta_high"] < -float(evidence["costly_clean_drop"])),
        })
    return pd.DataFrame(rows)


def oracle_choices(
    run: Run, selected: Mapping[str, Any], validation: pd.DataFrame, cells: pd.DataFrame, conditions: list[str], families: list[str], bootstrap: PairedBootstrap
) -> dict[str, dict[str, tuple[str, str, str]]]:
    """Với mỗi điều kiện, (kỹ thuật, cấu hình, tên chiến lược) mà hai kiểu oracle chọn. Ứng viên: kỹ thuật phù hợp với họ, cộng không xử lý.

    oracle_family_val chọn theo họ bằng điểm validation (macro F1 trung bình ba mức của cấu hình đã khóa, so với không xử lý), rồi áp lên test.
    oracle_condition_test chọn theo từng điều kiện bằng chính macro F1 test: cận trên lạc quan, vì chọn trên cùng tập đo.
    """
    val_none = validation.groupby("family")["macro_f1"].mean()
    family_pick: dict[str, tuple[str, str]] = {}
    for family in families:
        best = ("none", "none")
        best_score = float(val_none[family])
        for technique, entry in selected.items():
            if family in entry["per_family"] and float(entry["per_family"][family]["mean_macro_f1"]) > best_score:
                best, best_score = (technique, entry["per_family"][family]["config_id"]), float(entry["per_family"][family]["mean_macro_f1"])
        family_pick[family] = best
    picks: dict[str, dict[str, tuple[str, str, str]]] = {"oracle_family_val": {}, "oracle_condition_test": {}}
    suitable_cells = cells[(cells["role"] == "family") & cells["suitable"]]
    for condition in conditions:
        family = condition.rsplit("_s", 1)[0]
        technique, config_id = family_pick[family]
        picks["oracle_family_val"][condition] = (technique, config_id, "oracle_family_val")
        candidates = [("none", "none", bootstrap.macro_f1(run.predictions("none", "none", condition)))]
        for _, cell in suitable_cells[suitable_cells["condition"] == condition].iterrows():
            candidates.append((cell["technique"], cell["config_id"], float(cell["f1_processed"])))
        top = max(candidates, key=lambda item: item[2])
        picks["oracle_condition_test"][condition] = (top[0], top[1], "oracle_condition_test")
    return picks


def strategy_predictions(
    run: Run, selected: Mapping[str, Any], picks: Mapping[str, Mapping[str, tuple[str, str, str]]], retrained: Mapping[str, np.ndarray] | None, conditions: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Dự đoán trên ảnh sạch và 42 điều kiện của từng chiến lược: không xử lý, hai oracle, kỹ thuật cố định (cấu hình chung) và model huấn luyện lại."""
    strategies: dict[str, dict[str, np.ndarray]] = {"none": {name: run.predictions("none", "none", name) for name in ["clean", *conditions]}}
    for name, per_condition in picks.items():
        strategies[name] = {"clean": run.predictions("none", "none", "clean")}
        strategies[name].update({condition: run.predictions(*choice[:2], condition) if choice[0] != "none" else run.predictions("none", "none", condition) for condition, choice in per_condition.items()})
    for technique, entry in selected.items():
        general = entry["general"]["config_id"]
        if all(run.has(technique, general, name) for name in ["clean", *conditions]):
            strategies[f"fixed_{technique}"] = {name: run.predictions(technique, general, name) for name in ["clean", *conditions]}
    if retrained is not None:
        strategies["retrained"] = retrained
    return strategies


def compare_strategies(
    run: Run, bootstrap: PairedBootstrap, strategies: Mapping[str, Mapping[str, np.ndarray]], conditions: list[str], stats: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bảng theo chiến lược (ảnh sạch, trung bình 42 điều kiện, theo mức) và bảng theo từng điều kiện so với không xử lý."""
    labels = run.labels
    confidence = float(stats["confidence"])
    none = strategies["none"]
    none_means = np.mean([bootstrap.macro_f1_samples(none[name]) for name in conditions], axis=0)
    per_condition, summary = [], []
    for strategy, predictions in strategies.items():
        samples = {name: bootstrap.macro_f1_samples(predictions[name]) for name in ["clean", *conditions]}
        mean_samples = np.mean([samples[name] for name in conditions], axis=0)
        delta_samples = mean_samples - none_means
        rows = []
        for name in conditions:
            test = mcnemar_exact(none[name] == labels, predictions[name] == labels)
            delta = samples[name] - bootstrap.macro_f1_samples(none[name])
            low, high = bootstrap.interval(delta, confidence)
            rows.append({"strategy": strategy, "condition": name, "family": name.rsplit("_s", 1)[0], "severity": int(name.rsplit("_s", 1)[1]),
                         "f1": bootstrap.macro_f1(predictions[name]), "f1_none": bootstrap.macro_f1(none[name]), "delta": bootstrap.macro_f1(predictions[name]) - bootstrap.macro_f1(none[name]),
                         "delta_low": low, "delta_high": high, "gains": test["gains"], "harms": test["harms"], "p": test["p"]})
        frame = pd.DataFrame(rows)
        frame["p_holm"] = holm(frame["p"].tolist())
        per_condition.append(frame)
        clean_test = mcnemar_exact(none["clean"] == labels, predictions["clean"] == labels)
        clean_low, clean_high = bootstrap.interval(samples["clean"] - bootstrap.macro_f1_samples(none["clean"]), confidence)
        mean_low, mean_high = bootstrap.interval(mean_samples, confidence)
        delta_low, delta_high = bootstrap.interval(delta_samples, confidence)
        alpha = float(stats["alpha"])
        entry = {
            "strategy": strategy, "f1_clean": bootstrap.macro_f1(predictions["clean"]), "clean_delta": bootstrap.macro_f1(predictions["clean"]) - bootstrap.macro_f1(none["clean"]),
            "clean_delta_low": clean_low, "clean_delta_high": clean_high, "clean_p": clean_test["p"],
            "f1_mean_42": float(frame["f1"].mean()), "f1_mean_42_low": mean_low, "f1_mean_42_high": mean_high,
            "delta_mean_42": float(frame["delta"].mean()), "delta_mean_42_low": delta_low, "delta_mean_42_high": delta_high,
            "conditions_better": int(((frame["delta"] > 0) & (frame["p_holm"] < alpha)).sum()), "conditions_worse": int(((frame["delta"] < 0) & (frame["p_holm"] < alpha)).sum()),
        }
        for severity in SEVERITIES:
            entry[f"f1_mean_s{severity}"] = float(frame[frame["severity"] == severity]["f1"].mean())
        summary.append(entry)
    return pd.DataFrame(summary), pd.concat(per_condition, ignore_index=True)


def compare_retrained_with_oracle(run: Run, bootstrap: PairedBootstrap, strategies: Mapping[str, Mapping[str, np.ndarray]], conditions: list[str], stats: Mapping[str, Any]) -> pd.DataFrame:
    """Model huấn luyện lại so với oracle chọn theo họ trên validation, từng điều kiện: hiệu macro F1, khoảng tin cậy, McNemar."""
    labels = run.labels
    rows = []
    for name in conditions:
        base, other = strategies["oracle_family_val"][name], strategies["retrained"][name]
        difference = bootstrap.macro_f1_samples(other) - bootstrap.macro_f1_samples(base)
        low, high = bootstrap.interval(difference, float(stats["confidence"]))
        test = mcnemar_exact(base == labels, other == labels)
        rows.append({"condition": name, "family": name.rsplit("_s", 1)[0], "f1_oracle_family_val": bootstrap.macro_f1(base), "f1_retrained": bootstrap.macro_f1(other),
                     "delta": bootstrap.macro_f1(other) - bootstrap.macro_f1(base), "delta_low": low, "delta_high": high, "gains": test["gains"], "harms": test["harms"], "p": test["p"]})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame["p"].tolist())
    return frame


def plot_strategy_by_family(per_condition: pd.DataFrame, strategies: list[str], path: str | Path) -> None:
    """Macro F1 trung bình ba mức của từng họ cho các chiến lược được chọn."""
    table = per_condition[per_condition["strategy"].isin(strategies)].groupby(["family", "strategy"])["f1"].mean().unstack("strategy")[strategies]
    fig = Figure(figsize=(0.9 * len(table) + 3, 4.2))
    ax = fig.subplots()
    width = 0.8 / len(strategies)
    for index, strategy in enumerate(strategies):
        ax.bar(np.arange(len(table)) + index * width, table[strategy].values, width, label=strategy)
    ax.set_xticks(np.arange(len(table)) + 0.4 - width / 2)
    ax.set_xticklabels(table.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("macro F1 trung bình ba mức (test)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def build_report(config_path: str | Path, run_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Tính mọi bảng, ghi CSV, JSON và hình vào thư mục kết quả. Trả về tóm tắt."""
    cfg = yaml.safe_load(_abs(config_path).read_text(encoding="utf-8"))
    out = _abs(output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stats, evidence = cfg["statistics"], cfg["evidence"]
    run = Run(run_path)
    classes = [str(name) for name in yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))["classes"]]
    selected = load_selected(cfg["selected"])
    bootstrap = PairedBootstrap(run.labels, len(classes), int(stats["bootstrap"]), int(stats["seed"]))
    conditions = [row["condition"] for row in run.rows if row["technique"] == "none" and row["condition"] != "clean"]
    families = list(dict.fromkeys(name.rsplit("_s", 1)[0] for name in conditions))

    # Kiểm tra tái hiện: dự đoán "không xử lý" của giai đoạn A phải trùng với reference ở task 4.2 trên cùng ảnh.
    split = cfg["split"]
    mismatch: dict[str, int] = {}
    same_samples = False
    if _abs(cfg["reference_predictions"]).exists():
        reference = np.load(_abs(cfg["reference_predictions"]), allow_pickle=True)
        same_samples = bool(
            len(reference[f"labels_{split}"]) == len(run.labels)
            and np.array_equal(reference[f"labels_{split}"], run.labels)
            and [str(item) for item in reference[f"paths_{split}"]] == run.paths
        )
        if same_samples:
            for index, name in enumerate(str(item) for item in reference["conditions"]):
                mismatch[name] = int((reference[f"probs_{split}"][index].astype(np.float32).argmax(axis=1) != run.predictions("none", "none", name)).sum())

    cells = recovery_cells(run, bootstrap, stats, classes)
    cost = clean_cost(run, bootstrap, cells, stats, classes)
    family = family_summary(cells, cost, run, bootstrap, stats, evidence)
    validation = pd.read_csv(_abs(cfg["reference_validation_table"]))
    picks = oracle_choices(run, selected, validation, cells, conditions, families, bootstrap)
    retrained_path = _abs(cfg["retrained"]["predictions"])
    retrained = None
    if retrained_path.exists():
        saved = np.load(retrained_path, allow_pickle=False)
        retrained = {name: saved[f"{cfg['retrained']['prefix']}__{name}"].astype(np.int64) for name in ["clean", *conditions]}
        if not np.array_equal(saved["labels"], run.labels):
            raise RuntimeError("Nhãn của model huấn luyện lại khác bản đồ phục hồi: không phải cùng ảnh.")
    strategies = strategy_predictions(run, selected, picks, retrained, conditions)
    summary, per_condition = compare_strategies(run, bootstrap, strategies, conditions, stats)

    cells.to_csv(out / "recovery_map_test.csv", index=False)
    cost.to_csv(out / "clean_cost_test.csv", index=False)
    family.to_csv(out / "recovery_family_test.csv", index=False)
    summary.to_csv(out / "strategies_test.csv", index=False)
    per_condition.to_csv(out / "strategies_by_condition_test.csv", index=False)
    pd.DataFrame([{"condition": name, **dict(zip(["technique", "config_id"], picks[kind][name][:2])), "oracle": kind} for kind in picks for name in conditions]).to_csv(out / "oracle_choices_test.csv", index=False)
    if retrained is not None:
        compare_retrained_with_oracle(run, bootstrap, strategies, conditions, stats).to_csv(out / "retrained_vs_oracle_test.csv", index=False)

    mapped = cells[cells["role"] == "family"].assign(value=lambda d: d["recovery"])
    plot_matrix_heatmap(mapped, "recovery", families, out / "recovery_heatmap.png", "Recovery trên test (F1 sau xử lý trừ F1 suy giảm). Ô khung đen là cặp phù hợp.")
    plot_matrix_heatmap(mapped, "normalized_recovery", families, out / "normalized_recovery_heatmap.png", "Normalized recovery trên test. Ô trống: degradation dưới ngưỡng.", limit=1.0)
    chosen = [name for name in ["none", "oracle_family_val", "retrained"] if name in strategies]
    plot_strategy_by_family(per_condition, chosen, out / "strategies_by_family.png")

    keep = family[family["evidence"]]
    document = {
        "split": split, "reference_model": Path(cfg["train_config"]).stem.removeprefix("train_"), "bootstrap": int(stats["bootstrap"]), "confidence": float(stats["confidence"]), "alpha": float(stats["alpha"]),
        "same_samples_as_task_4_2": same_samples, "mismatch_vs_task_4_2": {"total": int(sum(mismatch.values())), "max_per_condition": int(max(mismatch.values(), default=0)), "by_condition": mismatch},
        "rule": evidence, "cells": int(len(cells)), "evidence": keep[["technique", "family", "config_id", "recovery_mean", "recovery_low", "recovery_high", "levels_gain_significant", "clean_delta", "clean_delta_low", "clean_delta_high", "costly"]].round(4).to_dict(orient="records"),
        "no_evidence_suitable": family[family["suitable"] & ~family["evidence"]][["technique", "family", "recovery_mean", "recovery_low", "recovery_high", "levels_gain_significant"]].round(4).to_dict(orient="records"),
        "strategies": summary.round(4).to_dict(orient="records"),
        "oracle_condition_test_note": "Cận trên lạc quan: chọn kỹ thuật theo từng điều kiện bằng chính macro F1 test.",
    }
    (out / "recovery_summary.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"cells": cells, "clean_cost": cost, "family": family, "strategies": summary, "per_condition": per_condition, "summary": document, "output_dir": out}


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Dùng: python -m src.evaluation.recovery_report configs/recovery_map.yaml <file .npz> [thư mục kết quả]")
    built = build_report(*sys.argv[1:])
    print(f"Đã ghi {len(built['cells'])} ô vào {built['output_dir']}")

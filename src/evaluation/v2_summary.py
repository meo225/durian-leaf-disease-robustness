"""Bảng tóm tắt bộ họ chưa thấy v2 từ file đã commit, chỉ đọc, không chạy mô hình.

- Độ nhạy của nhóm gần/xa với ngưỡng: ngưỡng đăng ký trước là MSD leave-one-out lớn nhất, nên một họ ngoại lai kéo nó lên. Bảng này đổi ngưỡng sang thứ hai lớn nhất và trung vị để xem
  nhãn nào đổi. Đây là phân tích thăm dò, nhãn đã khóa không đổi.
- Độ khó thực của Swin-Tiny theo nhóm: mức được khớp bằng mô hình thay thế, nên độ khó trên reference có thể lệch theo nhóm.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

from src.evaluation.msd import MSDResult, classify_near_far

REPORT_DIR = "reports/unseen_v2"
THRESHOLD_RULES = ("max", "second_max", "median")


def _result(entry: Mapping[str, Any]) -> MSDResult:
    return MSDResult(entry["msd"], entry.get("nearest", ""), entry["ci_low"], entry["ci_high"], entry.get("n_samples", 0))


def thresholds(seen_leave_one_out: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    values = sorted(entry["msd"] for entry in seen_leave_one_out.values())
    if len(values) < 2:
        raise ValueError("Cần ít nhất hai họ đã thấy.")
    return {"max": values[-1], "second_max": values[-2], "median": float(median(values))}


def threshold_sensitivity(msd_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Mỗi họ mới: MSD, khoảng tin cậy và nhãn dưới từng quy tắc ngưỡng. `changed` là nhãn khác nhãn đăng ký trước."""
    cuts = thresholds(msd_report["seen_leave_one_out"])
    rows = []
    for family in msd_report["new_families"]:
        result = _result(family)
        labels = {rule: classify_near_far(result, cuts[rule]) for rule in THRESHOLD_RULES}
        rows.append(
            {
                "family": family["family"],
                "msd": round(result.value, 3),
                "ci_low": round(result.ci_low, 3),
                "ci_high": round(result.ci_high, 3),
                **{f"label_{rule}": label for rule, label in labels.items()},
                "changed": any(labels[rule] != labels["max"] for rule in THRESHOLD_RULES),
            }
        )
    return sorted(rows, key=lambda row: row["msd"])


def read_realised(path: str | Path) -> list[dict[str, Any]]:
    """Đọc `realised_difficulty.csv` theo vị trí cột để không phụ thuộc tiêu đề tiếng Việt."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [
            {"family": row[0], "level": int(row[1]), "target": float(row[3]), "surrogate": float(row[5]), "swin": float(row[6])}
            for row in reader
        ]


def difficulty_by_group(realised: list[dict[str, Any]], groups: Mapping[str, str]) -> list[dict[str, Any]]:
    """ΔF1 trung bình của Swin-Tiny và sai lệch tuyệt đối so với mục tiêu, theo nhóm và mức."""
    rows = []
    for group in sorted(set(groups.values())):
        for level in sorted({item["level"] for item in realised}):
            items = [item for item in realised if groups[item["family"]] == group and item["level"] == level]
            if not items:
                continue
            rows.append(
                {
                    "group": group,
                    "level": level,
                    "n_families": len(items),
                    "target": round(mean(item["target"] for item in items), 3),
                    "surrogate_delta_f1": round(mean(item["surrogate"] for item in items), 3),
                    "swin_delta_f1": round(mean(item["swin"] for item in items), 3),
                    "swin_abs_error": round(mean(abs(item["swin"] - item["target"]) for item in items), 3),
                }
            )
    return rows


def role_counts(families: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in sorted({spec["group"] for spec in families.values()}):
        members = [spec for spec in families.values() if spec["group"] == group]
        rows.append({"group": group, **{role: sum(spec["role"] == role for spec in members) for role in ("train", "val", "test")}})
    return rows


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary(report_dir: str | Path = REPORT_DIR, config_path: str | Path = "configs/unseen_families_v2.yaml") -> dict[str, Path]:
    import yaml

    report_dir = Path(report_dir)
    msd_report = json.loads((report_dir / "msd.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    groups = {name: spec["group"] for name, spec in cfg["families"].items()}
    return {
        "threshold_sensitivity": write_csv(threshold_sensitivity(msd_report), report_dir / "msd_threshold_sensitivity.csv"),
        "difficulty_by_group": write_csv(difficulty_by_group(read_realised(report_dir / "realised_difficulty.csv"), groups), report_dir / "difficulty_by_group.csv"),
        "role_counts": write_csv(role_counts(cfg["families"]), report_dir / "role_counts.csv"),
    }


if __name__ == "__main__":
    for name, path in write_summary().items():
        print(name, path)

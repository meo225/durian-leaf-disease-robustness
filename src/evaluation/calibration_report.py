"""Kiểm tra và biểu đồ cho kết quả hiệu chỉnh mức suy giảm trên validation (tasks 4.1 và 5.5).

Đầu vào là bản ghi của `robustness.run_calibration`: {"clean_macro_f1", "conditions": {"<họ>_s<mức>": {"macro_f1", "per_class_recall", ...}}}.
Các ngưỡng kiểm tra là hằng số có tên, để người đọc thấy "đạt" nghĩa là gì và chỉnh được khi cần.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from matplotlib.figure import Figure
from matplotlib.patches import Patch

CHANCE_MACRO_F1 = 0.2  # đoán bừa trên 5 lớp
MONOTONIC_TOLERANCE = 0.005  # cho phép điểm tăng nhẹ giữa hai mức do nhiễu đo
COLLAPSE_DROP = 0.4  # mức 1 mà đã mất từng này F1 là quá mạnh
MILD_DROP = 0.03  # mức 3 mà mất ít hơn từng này F1 là quá nhẹ
HARSH_ABOVE_CHANCE = 0.1  # mức 3 mà F1 chỉ cao hơn đoán bừa từng này là quá nặng


def split_name(name: str) -> tuple[str, int]:
    """`fog_s2` thành ("fog", 2)."""
    family, severity = name.rsplit("_s", 1)
    return family, int(severity)


def scores_by_family(record: Mapping[str, Any]) -> dict[str, list[float]]:
    """Macro F1 của mỗi họ theo thứ tự mức tăng dần."""
    grouped: dict[str, dict[int, float]] = {}
    for name, entry in record["conditions"].items():
        family, severity = split_name(name)
        grouped.setdefault(family, {})[severity] = float(entry["macro_f1"])
    return {family: [levels[k] for k in sorted(levels)] for family, levels in grouped.items()}


def check_levels(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Mỗi họ một dòng: điểm các mức, mức giảm so với sạch và các cờ cần xem lại.

    Cờ: `non_monotonic` (điểm tăng theo mức), `collapsed_s1` (mức 1 đã mất quá nhiều), `too_mild` (mức 3 gần như không ảnh hưởng),
    `too_harsh` (mức 3 gần mức đoán bừa). `status` là "OK" khi không có cờ.
    """
    clean = float(record["clean_macro_f1"])
    rows = []
    for family, scores in scores_by_family(record).items():
        flags = []
        if any(later > earlier + MONOTONIC_TOLERANCE for earlier, later in zip(scores, scores[1:])):
            flags.append("non_monotonic")
        if clean - scores[0] >= COLLAPSE_DROP:
            flags.append("collapsed_s1")
        if clean - scores[-1] < MILD_DROP:
            flags.append("too_mild")
        if scores[-1] <= CHANCE_MACRO_F1 + HARSH_ABOVE_CHANCE:
            flags.append("too_harsh")
        rows.append({"family": family, "scores": scores, "drop_s3": clean - scores[-1], "flags": flags, "status": ", ".join(flags) or "OK"})
    return rows


def plot_severity_curves(record: Mapping[str, Any], path: str | Path, dpi: int = 150) -> None:
    """Macro F1 theo mức cho từng họ, kèm đường F1 sạch và đường đoán bừa."""
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    for family, scores in scores_by_family(record).items():
        ax.plot(range(1, len(scores) + 1), scores, marker="o", label=family)
    ax.axhline(float(record["clean_macro_f1"]), color="gray", linestyle="--", label="sạch")
    ax.axhline(CHANCE_MACRO_F1, color="red", linestyle=":", label="đoán bừa")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Mức")
    ax.set_ylabel("Macro F1 (validation)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)


def plot_level_comparison(
    groups: Mapping[str, Mapping[str, Any]], path: str | Path, severity: int = 3, dpi: int = 150
) -> None:
    """Macro F1 ở một mức của các họ trong nhiều bản ghi (ví dụ họ đã thấy và họ chưa thấy), cùng một trục để so độ khó."""
    entries = []
    for group, record in groups.items():
        for family, scores in scores_by_family(record).items():
            entries.append((group, family, scores[severity - 1]))
    entries.sort(key=lambda item: item[2])
    palette = {group: f"C{index}" for index, group in enumerate(groups)}
    fig = Figure(figsize=(7, 0.32 * len(entries) + 1.2))
    ax = fig.subplots()
    ax.barh([family for _, family, _ in entries], [score for _, _, score in entries], color=[palette[group] for group, _, _ in entries])
    ax.axvline(CHANCE_MACRO_F1, color="red", linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Macro F1 ở mức {severity} (validation)")
    ax.legend(handles=[Patch(color=color, label=group) for group, color in palette.items()], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)


def plot_per_class_recall(record: Mapping[str, Any], path: str | Path, severity: int = 3, dpi: int = 150) -> None:
    """Recall từng lớp ở một mức, mỗi họ một dòng, để thấy suy giảm làm hỏng lớp nào."""
    families = list(scores_by_family(record))
    first = next(iter(record["conditions"].values()))
    classes = list(first["per_class_recall"])
    matrix = [[record["conditions"][f"{family}_s{severity}"]["per_class_recall"][name] for name in classes] for family in families]
    fig = Figure(figsize=(1.4 * len(classes) + 1.5, 0.45 * len(families) + 1.5))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)), [name.replace("_", "\n") for name in classes], fontsize=7)
    ax.set_yticks(range(len(families)), families, fontsize=8)
    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Recall từng lớp ở mức {severity}", fontsize=9)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)

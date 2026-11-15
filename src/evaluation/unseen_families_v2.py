"""Dựng và khóa bộ họ chưa thấy v2 (task 5.7 và 5.8). Các hàm thuần dữ liệu, notebook 13 và 14 lo phần chạy mô hình.

Luồng:
1. `eligible_candidates`: từ kết quả `probe_families`, giữ họ chạy được và đủ nhanh.
2. Notebook 13 đo ΔF1 của mô hình thay thế trên mọi mức ứng viên (`candidate_conditions`, `delta_f1_table`), rồi `difficulty.match_families` chọn ba mức.
3. Notebook 14 đo MSD (`src.evaluation.msd`), xếp nhóm gần/xa, chia vai trò (`family_roles.assign_roles`) rồi `assemble_config` và `render_config` ghi cấu hình đã khóa.

Quy tắc nằm ở `registration` trong `configs/unseen_families_v2.yaml`, được đăng ký trước khi chạy và nằm trong phần băm sha256 của khóa.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.corruption.benchmark import ROOT, load_config
from src.corruption.external import EXTERNAL_TYPE, FAMILY_PREFIX, SOURCES, split_family_name
from src.corruption.family_lock import PROTOCOL, compute_lock

CONFIG_PATH = "configs/unseen_families_v2.yaml"
HEADER = """# Bộ họ chưa thấy v2 (task 5.7 và 5.8). Thay thế bộ v1 (configs/unseen_corruption.yaml) làm bộ đánh giá chính, v1 giữ nguyên để đối chiếu.
# Mọi quy tắc nằm ở `registration` và được đăng ký trước khi chạy. Mức do notebook 13 khớp theo ΔF1 của ConvNeXt-Tiny, nhóm gần/xa và vai trò do notebook 14
# lấy từ MSD của Swin-Tiny. Họ chỉ lấy từ ImageNet-C (imagecorruptions) và ImageNet-C-bar, không có họ nhóm tự cài.
# `lock.sha256` là mã băm của protocol, version, seed, registration và families. Sửa bất kỳ phần nào sau khi khóa thì load_conditions báo lỗi.
# Mở tập test cần FINAL_FAMILY_EVAL=1. Không chỉnh mức, nhóm hay vai trò sau khi đã thấy kết quả. Thử thêm thì ghi là phân tích thăm dò.
"""


# Các quy tắc mà mã cài cố định. Config ghi lại để đăng ký trước, và `load_registration` từ chối config ghi khác với thứ mã thực sự làm.
IMPLEMENTED_RULES = {
    ("msd", "threshold_rule"): "max_leave_one_out",
    ("msd", "feature"): "forward_head_pre_logits",
    ("msd", "augmentations"): ["trivialaugment_wide", "degradation_aug"],
    ("roles", "stratify_by"): "group",
    ("difficulty", "metric"): "macro_f1_drop",
}


def load_registration(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    cfg = load_config(config_path)
    if cfg.get("protocol") != PROTOCOL:
        raise ValueError(f"{config_path} không phải config giao thức {PROTOCOL}.")
    registration = dict(cfg["registration"])
    for (section, key), expected in IMPLEMENTED_RULES.items():
        if registration[section][key] != expected:
            raise ValueError(f"registration.{section}.{key} = {registration[section][key]!r}, nhưng mã chỉ cài {expected!r}.")
    return registration


def eligible_candidates(
    probe: Mapping[str, Mapping[str, Any]],
    max_seconds_per_image: float,
    sources: Sequence[str] = SOURCES,
) -> tuple[list[str], dict[str, str]]:
    """Họ ứng viên (tên có tiền tố nguồn) chạy được và có thời gian mỗi ảnh không quá ngưỡng. Trả về (họ giữ lại, {họ bị loại: lý do}).

    Ngưỡng thời gian là quy tắc đăng ký trước, quyết định theo chi phí tính toán chứ không theo độ khó hay kết quả: đánh giá cuối chạy mọi họ trên 888 ảnh test, ba mức, nhiều hệ thống.
    """
    prefixes = tuple(FAMILY_PREFIX[source] for source in sources)
    kept: list[str] = []
    excluded: dict[str, str] = {}
    for name in sorted(probe):
        if not name.startswith(prefixes):
            continue
        entry = probe[name]
        if not entry["ok"]:
            excluded[name] = f"lỗi khi chạy thử: {entry['error']}"
        elif not entry["seconds_per_image"] <= max_seconds_per_image:
            excluded[name] = f"chậm: {entry['seconds_per_image']:.2f} giây mỗi ảnh, quá ngưỡng {max_seconds_per_image:g}"
        else:
            kept.append(name)
    return kept, excluded


def candidate_grid(registration: Mapping[str, Any], source: str) -> tuple[float, ...]:
    """Lưới mức ứng viên của một nguồn, đọc từ `registration.candidates` để config là nguồn duy nhất."""
    return tuple(float(v) for v in registration["candidates"][f"{source}_severities"])


def candidate_conditions(families: Sequence[str], registration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Mọi (họ, mức) ứng viên thành điều kiện để chạy qua `evaluate_conditions`. Tên `<họ>_c<mức>` (c là candidate), khác `_s<mức>` của điều kiện đã khóa."""
    conditions: dict[str, dict[str, Any]] = {}
    for family in families:
        source, corruption = split_family_name(family)
        for value in candidate_grid(registration, source):
            conditions[f"{family}_c{value:g}"] = {
                "type": EXTERNAL_TYPE,
                "source": source,
                "corruption": corruption,
                "value": value,
                "family": family,
                "severity": value,
                "group": "candidate",
            }
    return conditions


def matched_conditions(matches: Mapping[str, Mapping[str, Any]], groups: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Điều kiện `<họ>_s<mức>` của các họ đã khớp mức, cùng tên với điều kiện sẽ nằm trong config khóa nên cùng nội dung ảnh suy giảm (seed lấy từ tên điều kiện)."""
    conditions: dict[str, dict[str, Any]] = {}
    for family in sorted(matches):
        source, corruption = split_family_name(family)
        for severity, value in enumerate(matches[family]["values"], start=1):
            conditions[f"{family}_s{severity}"] = {
                "type": EXTERNAL_TYPE,
                "source": source,
                "corruption": corruption,
                "value": float(value),
                "family": family,
                "severity": severity,
                "group": (groups or {}).get(family, "plan"),
            }
    return conditions


def delta_f1_table(results: Mapping[str, Mapping[str, Any]], conditions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[float, float]]:
    """Từ kết quả `evaluate_conditions["conditions"]` về {họ: {mức: ΔF1}}."""
    table: dict[str, dict[float, float]] = {}
    for name, entry in results.items():
        spec = conditions[name]
        table.setdefault(spec["family"], {})[float(spec["value"])] = float(entry["degradation"])
    return table


def assemble_config(
    registration: Mapping[str, Any],
    matches: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, str],
    roles: Mapping[str, str],
    seed: int,
    inputs: Mapping[str, str] | None = None,
    locked_on: str | None = None,
) -> dict[str, Any]:
    """Cấu hình v2 đã khóa. `matches` là kết quả `match_families` (họ giữ lại), `groups` là nhóm near/far/ambiguous, `roles` là train/val/test.

    `lock.sha256` tính trên protocol, version, seed, registration, families. `lock.inputs` là mã sha256 của các file báo cáo làm đầu vào, chỉ để truy vết.
    """
    families = {}
    for family in sorted(matches):
        source, corruption = split_family_name(family)
        families[family] = {
            "type": EXTERNAL_TYPE,
            "source": source,
            "corruption": corruption,
            "group": groups[family],
            "role": roles[family],
            "levels": [float(v) for v in matches[family]["values"]],
        }
    cfg: dict[str, Any] = {
        "protocol": PROTOCOL,
        "version": 2,
        "locked": True,
        "seed": int(seed),
        "registration": dict(registration),
        "families": families,
    }
    cfg["lock"] = {"sha256": compute_lock(cfg), "locked_on": locked_on or dt.date.today().isoformat(), "inputs": dict(inputs or {})}
    return cfg


def render_config(cfg: Mapping[str, Any]) -> str:
    """Văn bản YAML của cấu hình, có chú thích đầu file."""
    return HEADER + "\n" + yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_sources(families: Sequence[str]) -> dict[str, str]:
    """{tên họ: nguồn} cho `assign_roles`."""
    return {family: split_family_name(family)[0] for family in families}


def dump_config(cfg: Mapping[str, Any], path: str | Path = CONFIG_PATH) -> Path:
    target = Path(path)
    target = target if target.is_absolute() else ROOT / target
    target.write_text(render_config(cfg), encoding="utf-8")
    return target

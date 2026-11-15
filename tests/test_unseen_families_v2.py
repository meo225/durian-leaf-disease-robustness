import copy

import pytest
import yaml

from src.corruption import load_conditions
from src.corruption.external import IMAGENET_C_BAR_GRID, IMAGENET_C_GRID, SOURCE_IMAGENET_C, SOURCE_IMAGENET_C_BAR, candidate_values
from src.corruption.family_lock import FINAL_EVAL_ENV, PROTOCOL, compute_lock, guard_split, is_v2, verify_lock
from src.evaluation import unseen_families_v2 as U

REGISTRATION = U.load_registration()
MATCHES = {
    "cbar_sparkles": {"ok": True, "values": [2.0, 4.0, 7.0]},
    "inc_gaussian_noise": {"ok": True, "values": [1.0, 3.0, 5.0]},
    "cbar_perlin_noise": {"ok": True, "values": [3.0, 5.0, 8.0]},
}
GROUPS = {"cbar_sparkles": "near", "inc_gaussian_noise": "far", "cbar_perlin_noise": "ambiguous"}
ROLES = {"cbar_sparkles": "test", "inc_gaussian_noise": "val", "cbar_perlin_noise": "train"}


@pytest.fixture
def locked_cfg():
    return U.assemble_config(REGISTRATION, MATCHES, GROUPS, ROLES, seed=42, inputs={"calibration": "abc"}, locked_on="2026-11-20")


@pytest.fixture
def locked_path(tmp_path, locked_cfg):
    path = tmp_path / "v2.yaml"
    U.dump_config(locked_cfg, path)
    return path


def test_registration_is_consistent_with_code_constants():
    assert tuple(float(v) for v in REGISTRATION["candidates"]["imagenet_c_severities"]) == IMAGENET_C_GRID
    assert U.candidate_grid(REGISTRATION, SOURCE_IMAGENET_C_BAR) == IMAGENET_C_BAR_GRID
    assert U.candidate_grid(REGISTRATION, SOURCE_IMAGENET_C) == candidate_values(SOURCE_IMAGENET_C)
    assert REGISTRATION["difficulty"]["quantiles"] == sorted(REGISTRATION["difficulty"]["quantiles"])
    assert sum(REGISTRATION["roles"]["fractions"].values()) == pytest.approx(1.0)
    assert REGISTRATION["surrogate"]["model"] != REGISTRATION["reference"]["model"]


def test_repo_config_is_an_unlocked_preregistration_skeleton():
    cfg = yaml.safe_load(open(U.CONFIG_PATH, encoding="utf-8"))
    assert is_v2(cfg) and cfg["protocol"] == PROTOCOL and cfg["locked"] is False and cfg["families"] == {}
    assert load_conditions(U.CONFIG_PATH, "val") == {}
    with pytest.raises(PermissionError):
        load_conditions(U.CONFIG_PATH, "test")


def test_load_registration_rejects_rules_the_code_does_not_implement(tmp_path):
    cfg = yaml.safe_load(open(U.CONFIG_PATH, encoding="utf-8"))
    cfg["registration"]["msd"]["threshold_rule"] = "median_leave_one_out"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold_rule"):
        U.load_registration(path)


def test_load_registration_rejects_other_protocols():
    with pytest.raises(ValueError):
        U.load_registration("configs/unseen_corruption.yaml")


def test_eligible_candidates_apply_cost_rule_and_report_reasons():
    probe = {
        "cbar_fast": {"ok": True, "error": None, "seconds_per_image": 0.1},
        "cbar_slow": {"ok": True, "error": None, "seconds_per_image": 3.0},
        "inc_broken": {"ok": False, "error": "TypeError: x", "seconds_per_image": float("nan")},
        "inc_fast": {"ok": True, "error": None, "seconds_per_image": 0.01},
        "other_family": {"ok": True, "error": None, "seconds_per_image": 0.0},
    }
    kept, excluded = U.eligible_candidates(probe, 0.5)
    assert kept == ["cbar_fast", "inc_fast"]
    assert set(excluded) == {"cbar_slow", "inc_broken"} and "chậm" in excluded["cbar_slow"] and "lỗi" in excluded["inc_broken"]
    assert U.eligible_candidates(probe, 0.5, sources=[SOURCE_IMAGENET_C])[0] == ["inc_fast"]


def test_candidate_conditions_cover_the_registered_grid():
    conditions = U.candidate_conditions(["cbar_sparkles", "inc_gaussian_noise"], REGISTRATION)
    assert len(conditions) == len(IMAGENET_C_BAR_GRID) + len(IMAGENET_C_GRID)
    spec = conditions["cbar_sparkles_c7"]
    assert (spec["type"], spec["source"], spec["corruption"], spec["value"], spec["family"]) == ("external", SOURCE_IMAGENET_C_BAR, "sparkles", 7.0, "cbar_sparkles")
    assert "inc_gaussian_noise_c5" in conditions and "inc_gaussian_noise_c0.1" in conditions


def test_matched_conditions_use_locked_names_and_values():
    conditions = U.matched_conditions(MATCHES, GROUPS)
    assert len(conditions) == 9 and conditions["cbar_sparkles_s3"]["value"] == 7.0 and conditions["cbar_sparkles_s3"]["group"] == "near"
    assert conditions["inc_gaussian_noise_s1"]["source"] == SOURCE_IMAGENET_C
    # cùng tên và cùng nội dung với điều kiện load_conditions đọc lại từ config khóa
    cfg = U.assemble_config(REGISTRATION, MATCHES, GROUPS, ROLES, seed=1)
    assert {name: spec["value"] for name, spec in conditions.items()} == {f"{f}_s{k}": v for f, spec in cfg["families"].items() for k, v in enumerate(spec["levels"], 1)}


def test_delta_f1_table_groups_by_family_and_value():
    conditions = U.candidate_conditions(["inc_gaussian_noise"], REGISTRATION)
    results = {name: {"degradation": 0.05 * i} for i, name in enumerate(conditions)}
    table = U.delta_f1_table(results, conditions)
    position = IMAGENET_C_GRID.index(3.0)
    assert list(table) == ["inc_gaussian_noise"] and table["inc_gaussian_noise"][3.0] == pytest.approx(0.05 * position)
    assert set(table["inc_gaussian_noise"]) == set(IMAGENET_C_GRID)


def test_assembled_config_has_levels_roles_and_valid_lock(locked_cfg, locked_path):
    assert locked_cfg["locked"] is True and locked_cfg["lock"]["locked_on"] == "2026-11-20"
    assert list(locked_cfg["families"]) == ["cbar_perlin_noise", "cbar_sparkles", "inc_gaussian_noise"]
    assert locked_cfg["families"]["cbar_sparkles"] == {"type": "external", "source": "imagenet_c_bar", "corruption": "sparkles", "group": "near", "role": "test", "levels": [2.0, 4.0, 7.0]}
    verify_lock(locked_cfg)
    text = locked_path.read_text(encoding="utf-8")
    assert text.startswith("# Bộ họ chưa thấy v2") and "sha256" in text
    reloaded = yaml.safe_load(text)
    assert compute_lock(reloaded) == reloaded["lock"]["sha256"] == locked_cfg["lock"]["sha256"]
    conditions = load_conditions(locked_path, "val")
    assert len(conditions) == 9 and conditions["cbar_sparkles_s3"]["value"] == 7.0 and conditions["cbar_sparkles_s3"]["role"] == "test"
    assert set(load_conditions(locked_path, "val", roles=["val", "train"])) == {f"{f}_s{k}" for f in ("inc_gaussian_noise", "cbar_perlin_noise") for k in (1, 2, 3)}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: cfg["families"]["cbar_sparkles"]["levels"].__setitem__(0, 1.0),
        lambda cfg: cfg["families"]["cbar_sparkles"].__setitem__("role", "val"),
        lambda cfg: cfg["families"]["cbar_sparkles"].__setitem__("group", "far"),
        lambda cfg: cfg["registration"]["difficulty"].__setitem__("tolerance", 0.5),
        lambda cfg: cfg["registration"]["msd"].__setitem__("n_images", 5),
        lambda cfg: cfg.__setitem__("seed", 1),
        lambda cfg: cfg["families"].pop("inc_gaussian_noise"),
    ],
)
def test_any_change_after_locking_is_detected(locked_cfg, tmp_path, mutate):
    changed = copy.deepcopy(locked_cfg)
    mutate(changed)
    with pytest.raises(PermissionError, match="sha256"):
        verify_lock(changed)
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(PermissionError):
        load_conditions(path, "val")


def test_lock_ignores_metadata_that_is_not_part_of_the_payload(locked_cfg):
    other = copy.deepcopy(locked_cfg)
    other["lock"]["inputs"] = {"calibration": "zzz"}
    other["lock"]["locked_on"] = "2030-01-01"
    verify_lock(other)


def test_locked_without_hash_is_rejected(locked_cfg):
    broken = copy.deepcopy(locked_cfg)
    broken["lock"] = {}
    with pytest.raises(PermissionError, match="thiếu"):
        verify_lock(broken)


def test_test_split_needs_lock_hash_and_final_eval_env(locked_cfg, locked_path, monkeypatch):
    monkeypatch.delenv(FINAL_EVAL_ENV, raising=False)
    with pytest.raises(PermissionError, match=FINAL_EVAL_ENV):
        load_conditions(locked_path, "test")
    monkeypatch.setenv(FINAL_EVAL_ENV, "0")
    with pytest.raises(PermissionError, match=FINAL_EVAL_ENV):
        load_conditions(locked_path, "test")
    monkeypatch.setenv(FINAL_EVAL_ENV, "1")
    assert len(load_conditions(locked_path, "test")) == 9
    # val và train không bị chặn dù thiếu biến môi trường
    monkeypatch.delenv(FINAL_EVAL_ENV)
    guard_split(locked_cfg, "val")
    guard_split(locked_cfg, "train")
    unlocked = {**locked_cfg, "locked": False}
    monkeypatch.setenv(FINAL_EVAL_ENV, "1")
    with pytest.raises(PermissionError, match="chưa khóa"):
        guard_split(unlocked, "test")


def test_other_protocols_are_not_affected_by_the_v2_guard(monkeypatch):
    monkeypatch.delenv(FINAL_EVAL_ENV, raising=False)
    assert len(load_conditions("configs/unseen_corruption.yaml", "test")) == 18
    assert len(load_conditions("configs/corruption.yaml", "test")) == 42


def test_registration_records_the_v2_1_amendment_and_keeps_the_matching_rule():
    amendment, = REGISTRATION["amendments"]
    assert amendment["id"] == "v2.1" and "reports/unseen_v2/pass1_v2.0" in amendment["disclosure"]
    # Quy tắc khớp độ khó giữ nguyên như bản đăng ký đầu: phân vị, dung sai, lưới ImageNet-C-bar.
    assert REGISTRATION["difficulty"]["quantiles"] == [0.2, 0.5, 0.8] and REGISTRATION["difficulty"]["tolerance"] == 0.05
    assert REGISTRATION["candidates"]["max_seconds_per_image"] == 0.5
    assert REGISTRATION["candidates"]["imagenet_c_bar_severities"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

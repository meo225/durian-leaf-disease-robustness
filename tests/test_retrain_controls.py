import json
import math

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision import transforms

from src.corruption import load_conditions
from src.evaluation import seed_summary as S
from src.evaluation.leave_group import load_pairwise_matrix, select_leave_group
from src.training import run_control as C
from src.training.control_augmentation import GEOMETRIC_OPS, GeometricTrivialAugment, operation_names
from src.training.run import ROOT
from src.training.run_generic_aug import build_train_steps

ARMS = ("baseline", "degradation", "trivialaugment", "trivialaugment_geometric", "degradation_leave_group")
FAMILIES = list(dict.fromkeys(spec["family"] for spec in load_conditions("configs/corruption.yaml", "train").values()))


def _cfg(arm):
    return C.load_arm_config(f"configs/controls/{arm}.yaml")


def _matrix(near_anchor=("noise", "sensor_noise")):
    """Symmetric 14x14 distances: `blur` is close to `near_anchor`, everything else is far."""
    matrix = {a: {b: 0.0 if a == b else 10.0 for b in FAMILIES} for a in FAMILIES}
    for index, other in enumerate(near_anchor, start=1):
        matrix["blur"][other] = matrix[other]["blur"] = float(index)
    return matrix


# --- protocol ---------------------------------------------------------------------------------------------------------


def test_all_arms_share_the_protocol_and_differ_only_in_augmentation():
    configs = {arm: _cfg(arm)[0] for arm in ARMS}
    assert C.protocol_differences(configs) == []
    for cfg in configs.values():
        assert cfg["epochs"] == 30
        assert cfg["early_stopping"] is False
        assert cfg["validation"] == {"corruption_config": "configs/corruption.yaml", "severity": 2}


def test_arm_file_cannot_override_a_protocol_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"protocol": "configs/controls/protocol.yaml", "arm": "x", "model_name": "x", "augmentation": {"method": "none"}, "epochs": 12}))
    with pytest.raises(ValueError, match="epochs"):
        C.load_arm_config(bad)


def test_protocol_rejects_early_stopping_and_unknown_methods():
    cfg = dict(_cfg("baseline")[0])
    with pytest.raises(ValueError, match="early_stopping"):
        C.check_protocol({**cfg, "early_stopping": True})
    with pytest.raises(ValueError, match="không hỗ trợ"):
        C.check_protocol({**cfg, "augmentation": {"method": "randaugment"}})
    with pytest.raises(ValueError, match="leave_group"):
        C.check_protocol({**cfg, "augmentation": {"method": "none", "leave_group": {}}})


def test_degradation_arm_matches_the_task_4_5_config():
    old = yaml.safe_load((ROOT / "configs/train_swin_tiny_aug.yaml").read_text(encoding="utf-8"))
    new = _cfg("degradation")[0]
    assert new["model_name"] == old["model_name"]
    assert {key: value for key, value in new["augmentation"].items() if key != "method"} == old["augmentation"]
    for key in ("epochs", "learning_rate", "weight_decay", "optimizer", "scheduler", "num_workers", "backbone", "pretrained"):
        assert new[key] == old[key]
    assert new["validation"]["severity"] == old["validation"]["severity"]
    assert new["validation"]["corruption_config"] == old["augmentation"]["corruption_config"]


def test_trivialaugment_arm_matches_the_task_5_6_config():
    old = yaml.safe_load((ROOT / "configs/train_swin_tiny_trivialaugment.yaml").read_text(encoding="utf-8"))
    new = _cfg("trivialaugment")[0]
    assert new["model_name"] == old["model_name"]
    assert new["augmentation"] == old["augmentation"]
    assert new["validation"] == old["validation"]


def test_plans_cover_the_registered_runs():
    assert [(s.arm, s.seed) for s in C.PLAN_5_10] == [("trivialaugment_geometric", 42), ("degradation_leave_group", 42)]
    assert len(C.PLAN_5_9) == 9 and {s.seed for s in C.PLAN_5_9} == {42, 43, 44}
    assert C.SMOKE in C.PLAN_5_9


# --- augmentation -----------------------------------------------------------------------------------------------------


def test_geometric_trivialaugment_keeps_only_geometric_operations():
    augmentation = GeometricTrivialAugment()
    assert tuple(operation_names(augmentation)) == GEOMETRIC_OPS
    optical = {"Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize", "AutoContrast", "Equalize"}
    assert not optical & set(operation_names(augmentation))
    assert "Brightness" in operation_names(transforms.TrivialAugmentWide())


def test_geometric_trivialaugment_returns_an_image_of_the_same_size():
    torch.manual_seed(0)
    image = Image.new("RGB", (64, 64), (120, 200, 40))
    augmentation = GeometricTrivialAugment()
    for _ in range(40):
        out = augmentation(image)
        assert out.size == image.size and out.mode == "RGB"


def test_baseline_loader_adds_nothing_to_the_reference_pipeline():
    _, pipeline_cfg = _cfg("baseline")
    steps = build_train_steps(pipeline_cfg, nn.Identity())
    kinds = [type(step).__name__ for step in steps]
    assert kinds == ["ConditionalResize", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation", "Identity", "ToTensor", "Normalize"]
    # Identity returns the input object untouched.
    image = Image.new("RGB", (8, 8))
    assert nn.Identity()(image) is image


def test_build_augmentation_per_arm():
    assert C.build_augmentation(_cfg("baseline")[0]) is None
    degradation = C.build_augmentation(_cfg("degradation")[0])
    assert len(degradation.specs) == 42 and degradation.families() == FAMILIES
    assert type(C.build_augmentation(_cfg("trivialaugment")[0])) is transforms.TrivialAugmentWide
    assert isinstance(C.build_augmentation(_cfg("trivialaugment_geometric")[0]), GeometricTrivialAugment)


# --- leave-group-out --------------------------------------------------------------------------------------------------


def test_select_leave_group_takes_anchor_and_nearest_families():
    group = select_leave_group(_matrix(), "blur", 3)
    assert group["families"] == ["blur", "noise", "sensor_noise"]
    assert group["distance_to_anchor"] == {"blur": 0.0, "noise": 1.0, "sensor_noise": 2.0}
    assert len(group["remaining_families"]) == len(FAMILIES) - 3
    assert group["mean_distance_inside"] < group["mean_distance_to_rest"]


def test_select_leave_group_breaks_ties_by_name_and_is_deterministic():
    flat = {a: {b: 0.0 if a == b else 5.0 for b in FAMILIES} for a in FAMILIES}
    first = select_leave_group(flat, "blur", 3)["families"]
    assert first == ["blur", *sorted(f for f in FAMILIES if f != "blur")[:2]]
    assert select_leave_group(flat, "blur", 3)["families"] == first


def test_select_leave_group_validates_input():
    matrix = _matrix()
    with pytest.raises(ValueError, match="mỏ neo"):
        select_leave_group(matrix, "fog", 3)
    for size in (0, len(FAMILIES)):
        with pytest.raises(ValueError, match="Kích thước"):
            select_leave_group(matrix, "blur", size)
    asymmetric = {a: dict(row) for a, row in matrix.items()}
    asymmetric["blur"]["noise"] = 99.0
    with pytest.raises(ValueError, match="đối xứng"):
        select_leave_group(asymmetric, "blur", 3)
    broken = {a: dict(row) for a, row in matrix.items()}
    del broken["blur"]["noise"]
    with pytest.raises(ValueError, match="đủ cột"):
        select_leave_group(broken, "blur", 3)


def test_load_pairwise_matrix_explains_the_missing_task_5_7_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="5.7"):
        load_pairwise_matrix(tmp_path / "msd.json")
    (tmp_path / "msd.json").write_text(json.dumps({"other": {}}))
    with pytest.raises(KeyError, match="seen_pairwise_distance"):
        load_pairwise_matrix(tmp_path / "msd.json")


def _leave_group_cfg(tmp_path, matrix):
    path = tmp_path / "msd.json"
    path.write_text(json.dumps({"seen_pairwise_distance": matrix}))
    cfg = json.loads(json.dumps(_cfg("degradation_leave_group")[0]))
    cfg["augmentation"]["leave_group"]["matrix"] = str(path)
    return cfg


def test_leave_group_arm_removes_exactly_the_selected_families(tmp_path):
    cfg = _leave_group_cfg(tmp_path, _matrix())
    group = C.resolve_leave_group(cfg)
    assert group["families"] == ["blur", "noise", "sensor_noise"]
    augmentation = C.build_augmentation(cfg, group)
    assert set(augmentation.families()) == set(FAMILIES) - set(group["families"])
    assert len(augmentation.specs) == 3 * (len(FAMILIES) - 3)
    record = C.augmentation_record(cfg, augmentation, group)
    assert record["excluded_families"] == group["families"] and record["conditions"] == 33


def test_leave_group_rejects_a_manual_exclusion_and_unknown_families(tmp_path):
    cfg = _leave_group_cfg(tmp_path, _matrix())
    cfg["augmentation"]["exclude_families"] = ["jpeg"]
    with pytest.raises(ValueError, match="exclude_families"):
        C.resolve_leave_group(cfg)
    renamed = {("fog" if a == "noise" else a): {("fog" if b == "noise" else b): v for b, v in row.items()} for a, row in _matrix().items()}
    with pytest.raises(ValueError, match="benchmark 4.1"):
        C.resolve_leave_group(_leave_group_cfg(tmp_path, renamed))


def test_arms_without_leave_group_resolve_to_none():
    assert C.resolve_leave_group(_cfg("degradation")[0]) is None


# --- reusing the seed-42 runs -----------------------------------------------------------------------------------------


def _record(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("arm", ["degradation", "trivialaugment"])
def test_old_seed_42_runs_match_the_protocol(arm):
    train_cfg, pipeline_cfg = _cfg(arm)
    record = _record(train_cfg["reuse"]["record"])
    assert C.reusable_problems(record, train_cfg, pipeline_cfg, 42) == []


def test_old_baseline_does_not_match_the_protocol():
    train_cfg, pipeline_cfg = _cfg("baseline")
    old = _record("reports/baselines/swin_tiny/swin_tiny_seed42.json")
    problems = " ".join(C.reusable_problems(old, train_cfg, pipeline_cfg, 42))
    assert "dừng sớm" in problems and "validation hỗn hợp" in problems and "model" in problems


def test_reuse_rejects_changed_runs():
    train_cfg, pipeline_cfg = _cfg("degradation")
    record = _record(train_cfg["reuse"]["record"])
    assert any("seed" in p for p in C.reusable_problems(record, train_cfg, pipeline_cfg, 43))
    assert any("dừng sớm" in p for p in C.reusable_problems({**record, "epochs_run": 12}, train_cfg, pipeline_cfg, 42))
    assert any("learning_rate" in p for p in C.reusable_problems({**record, "hyperparameters": {**record["hyperparameters"], "learning_rate": 1e-3}}, train_cfg, pipeline_cfg, 42))
    assert any("probability" in p for p in C.reusable_problems({**record, "augmentation": {**record["augmentation"], "probability": 0.9}}, train_cfg, pipeline_cfg, 42))
    assert any("excluded_families" in p for p in C.reusable_problems({**record, "augmentation": {**record["augmentation"], "excluded_families": ["blur"]}}, train_cfg, pipeline_cfg, 42))
    clean_selected = {key: value for key, value in record.items() if key not in ("val_set", "val_selection")}
    assert any("validation hỗn hợp" in p for p in C.reusable_problems(clean_selected, train_cfg, pipeline_cfg, 42))


def test_find_run_reuses_seed_42_trains_the_rest(tmp_path):
    for arm in ("degradation", "trivialaugment"):
        found = C.find_run(C.RunSpec(arm, 42), tmp_path)
        assert found["source"] == "reused" and found["record"].is_file() and found["conditions"].is_file()
        assert C.find_run(C.RunSpec(arm, 43), tmp_path) is None
    assert C.find_run(C.RunSpec("baseline", 42), tmp_path) is None
    assert C.find_run(C.RunSpec("trivialaugment_geometric", 42), tmp_path) is None


def test_find_run_prefers_a_new_run_and_needs_both_files(tmp_path):
    train_cfg, _ = _cfg("baseline")
    paths = C.result_paths(train_cfg, 43, tmp_path)
    paths["record"].write_text("{}")
    assert C.find_run(C.RunSpec("baseline", 43), tmp_path) is None
    paths["conditions"].write_text("x")
    assert C.find_run(C.RunSpec("baseline", 43), tmp_path)["source"] == "trained"
    assert paths["record"].name == "swin_tiny_ctrl_baseline_seed43.json"


def test_find_run_refuses_a_reuse_that_no_longer_matches(tmp_path, monkeypatch):
    real = C.load_arm_config

    def tampered(path):
        train_cfg, pipeline_cfg = real(path)
        if "configs/controls/degradation.yaml" in str(path):
            train_cfg = {**train_cfg, "learning_rate": 1e-3}
        return train_cfg, pipeline_cfg

    monkeypatch.setattr(C, "load_arm_config", tampered)
    with pytest.raises(ValueError, match="Train lại"):
        C.find_run(C.RunSpec("degradation", 42), tmp_path)


def test_run_plan_skips_finished_runs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(C, "run", lambda config, seed, require_gpu=True: calls.append((config, seed)))
    trained = C.run_plan(C.PLAN_5_9, tmp_path)
    assert (("configs/controls/degradation.yaml", 42) not in calls) and (("configs/controls/trivialaugment.yaml", 42) not in calls)
    assert len(calls) == 7 and len(trained) == 7
    assert sum(1 for spec in trained if spec.arm == "baseline") == 3


# --- seed summary -----------------------------------------------------------------------------------------------------


def _frame(scores):
    rows = [{"condition": "clean", "family": "clean", "macro_f1": 0.9}]
    rows += [{"condition": f"{family}_s{level}", "family": family, "macro_f1": value} for family, level, value in scores]
    return pd.DataFrame(rows)


def test_condition_objective_ignores_clean_and_filters_families():
    frame = _frame([("blur", 1, 0.8), ("blur", 2, 0.6), ("jpeg", 1, 1.0)])
    assert S.condition_objective(frame) == pytest.approx(0.8)
    assert S.condition_objective(frame, ["blur"]) == pytest.approx(0.7)
    assert S.clean_f1(frame) == 0.9
    with pytest.raises(ValueError):
        S.condition_objective(frame, ["noise"])


def test_seed_stats_use_the_sample_standard_deviation():
    stats = S.seed_stats([0.80, 0.82, 0.84])
    assert stats["mean"] == pytest.approx(0.82) and stats["sd"] == pytest.approx(0.02)
    assert math.isnan(S.seed_stats([0.8])["sd"])
    with pytest.raises(ValueError):
        S.seed_stats([])


def test_decision_rule_needs_more_than_k_standard_deviations():
    assert S.decide(0.06, 0.02) == "vượt"
    assert S.decide(0.05, 0.02) == "chưa kết luận"  # exactly 2.5 SD is not more than 2.5 SD
    assert S.decide(-0.06, 0.02) == "vượt"
    assert S.decide(0.5, float("nan")) == "chưa đủ seed"


def test_compare_arms_pools_the_two_standard_deviations():
    values = {"a": [0.90, 0.92, 0.94], "b": [0.80, 0.82, 0.84], "c": [0.91, 0.92, 0.93]}
    table = S.compare_arms(values, [("a", "b"), ("a", "c")]).set_index("b")
    assert table.loc["b", "delta"] == pytest.approx(0.10) and table.loc["b", "pooled_sd"] == pytest.approx(0.02)
    assert table.loc["b", "ratio"] == pytest.approx(5.0) and table.loc["b", "kết luận"] == "vượt"
    assert table.loc["c", "kết luận"] == "chưa kết luận"


def test_single_seed_arm_borrows_the_reference_standard_deviation():
    out = S.compare_single_to_arm(0.70, [0.80, 0.82, 0.84])
    assert out["delta"] == pytest.approx(-0.12) and out["reference_sd"] == pytest.approx(0.02) and out["kết luận"] == "vượt"
    assert S.compare_single_to_arm(0.81, [0.80, 0.82, 0.84])["kết luận"] == "chưa kết luận"
    assert S.compare_single_to_arm(0.81, [0.80])["kết luận"] == "chưa đủ seed"


# --- validation for the leave-group arm, results from earlier Kaggle sessions --------------------------------------------


def _fake_full_validation(clean=3, names=FAMILIES):
    chunks = 1 + len(names)
    images = torch.cat([torch.full((clean, 3, 2, 2), index, dtype=torch.uint8) for index in range(chunks)])
    return C.run_aug.CachedImages(images, list(range(clean)) * chunks, [0.5] * 3, [0.2] * 3)


def test_leave_group_validation_drops_only_the_held_out_families(monkeypatch):
    train_cfg, pipeline_cfg = _cfg("degradation_leave_group")
    monkeypatch.setattr(C.run_aug, "build_mixed_validation", lambda cfg, pipeline: _fake_full_validation())
    full = C.build_validation(train_cfg, pipeline_cfg, None)
    assert len(full) == 3 * 15
    group = {"families": ["blur", "noise", "sensor_noise"]}
    reduced = C.build_validation(train_cfg, pipeline_cfg, group)
    assert len(reduced) == 3 * (15 - 3)
    kept_chunks = sorted({int(image[0, 0, 0]) for image in reduced.images})
    dropped = [position for position, family in enumerate(FAMILIES, start=1) if family in group["families"]]
    assert kept_chunks == [index for index in range(15) if index not in dropped]
    assert 0 in kept_chunks  # clean images stay
    assert reduced.labels == list(range(3)) * 12


def test_import_previous_results_copies_complete_pairs_without_overwriting(tmp_path):
    source, target = tmp_path / "input" / "earlier", tmp_path / "results"
    source.mkdir(parents=True)
    (source / "swin_tiny_ctrl_baseline_seed42.json").write_text("{}")
    (source / "swin_tiny_ctrl_baseline_seed42_val_conditions.csv").write_text("a")
    (source / "swin_tiny_ctrl_baseline_seed43.json").write_text("{}")  # no CSV: skipped
    (source / "unrelated_model_seed42.json").write_text("{}")
    (source / "unrelated_model_seed42_val_conditions.csv").write_text("a")
    target.mkdir()
    (target / "swin_tiny_ctrl_baseline_seed42.json").write_text("keep")
    copied = C.import_previous_results(tmp_path / "input", target)
    assert copied == ["swin_tiny_ctrl_baseline_seed42_val_conditions.csv"]
    assert (target / "swin_tiny_ctrl_baseline_seed42.json").read_text() == "keep"
    assert not (target / "unrelated_model_seed42.json").exists()


def _write_run(directory, model_name, seed, scores, best_epoch=7, excluded=()):
    stem = f"{model_name}_seed{seed}"
    (directory / f"{stem}.json").write_text(json.dumps({"best_epoch": best_epoch, "epochs_run": 30, "val_macro_f1": 0.9, "val_selection": {"excluded_families": list(excluded)}}))
    _frame(scores).to_csv(directory / f"{stem}_val_conditions.csv", index=False)


def test_run_table_reads_new_runs_and_reused_seed_42(tmp_path):
    scores = [(family, level, 0.5 + 0.01 * index) for index, family in enumerate(FAMILIES) for level in (1, 2, 3)]
    _write_run(tmp_path, "swin_tiny_ctrl_baseline", 43, scores)
    plan = [C.RunSpec("degradation", 42), C.RunSpec("baseline", 43), C.RunSpec("baseline", 44)]
    table = C.run_table(plan, tmp_path)
    assert list(table["arm"]) == ["degradation", "baseline"] and list(table["source"]) == ["reused", "trained"]
    assert list(table["val_families"]) == [14, 14]
    baseline = table.set_index("arm").loc["baseline"]
    assert baseline["objective_42"] == pytest.approx(sum(0.5 + 0.01 * i for i in range(14)) / 14)
    assert baseline["clean_val_f1"] == 0.9 and baseline["best_epoch"] == 7


def test_run_table_splits_the_held_out_group_from_the_rest(tmp_path):
    scores = [(family, level, 0.2 if family == "blur" else 0.8) for family in FAMILIES for level in (1, 2, 3)]
    _write_run(tmp_path, "swin_tiny_aug_leave_group", 42, scores, excluded=["blur", "noise", "sensor_noise"])
    table = C.run_table([C.RunSpec("degradation_leave_group", 42)], tmp_path, group_families=["blur"])
    row = table.iloc[0]
    assert row["objective_group"] == pytest.approx(0.2) and row["objective_rest"] == pytest.approx(0.8)
    assert row["val_families"] == 11

import json

import numpy as np
import pytest
import torch
from selector_helpers import scene, tiny_model
from sklearn.metrics import f1_score

from src.corruption.basic import apply_corruption
from src.selector.calibrate import (
    CLEAN,
    CalibrationTable,
    Settings,
    assign_selector,
    collect,
    cross_validate,
    fit,
    macro_f1_rows,
    report,
    save_artifacts,
    score,
    threshold_grid,
)
from src.selector.features import FEATURES
from src.selector.pipeline import load_selector_config
from src.selector.plots import plot_by_family
from src.selector.rules import Clause, Rule, RuleSelector

CONFIG = load_selector_config("configs/selector.yaml")
SETTINGS = Settings.from_config(CONFIG["calibration"])
ACTIONS = ["keep", "unsharp_strong", "unsharp_mild", "bilateral_wide", "bilateral_narrow"]


def _synthetic_table(images: int = 100, seed: int = 0, helpful: bool = True, separable: bool = True) -> CalibrationTable:
    """Bảng ba điều kiện (sạch, mờ, nhiễu). Khi `helpful`, unsharp chữa mờ và bilateral chữa nhiễu, còn trên ảnh sạch cả hai đều làm hỏng."""
    rng = np.random.default_rng(seed)
    labels = np.arange(images) % 5
    wrong = (labels + 1) % 5
    count = 3
    features = rng.uniform(0.2, 0.8, size=(count, images, len(FEATURES))).astype(np.float32)
    index = {name: FEATURES.index(name) for name in FEATURES}
    if separable:
        features[0, :, index["sharpness"]] = rng.uniform(0.5, 1.0, images)
        features[1, :, index["sharpness"]] = rng.uniform(0.0, 0.1, images)
        features[2, :, index["sharpness"]] = rng.uniform(0.5, 1.0, images)
        features[[0, 1], :, index["noise_sigma"]] = rng.uniform(0.0, 0.01, (2, images))
        features[2, :, index["noise_sigma"]] = rng.uniform(0.05, 0.1, images)
        features[:, :, index["noise_ratio"]] = features[:, :, index["noise_sigma"]]
    predictions = np.zeros((len(ACTIONS), count, images), dtype=np.int8)
    predictions[0, 0] = labels
    predictions[0, 1:] = np.where(np.arange(images) % 10 < 7, wrong, labels)
    for action in range(1, len(ACTIONS)):
        predictions[action, 0] = wrong if helpful else labels
        predictions[action, 1:] = predictions[0, 1:]
    if helpful:
        predictions[ACTIONS.index("unsharp_strong"), 1] = labels
        predictions[ACTIONS.index("bilateral_wide"), 2] = labels
    return CalibrationTable(
        [CLEAN, "blur_s1", "noise_s1"], [CLEAN, "blur", "noise"], list(ACTIONS), features, predictions, None, labels,
        [f"ảnh_{i}.jpg" for i in range(images)], [f"lớp_{i}" for i in range(5)],
    )


def test_fast_macro_f1_matches_sklearn_including_empty_classes():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 60)
    predictions = np.stack([rng.integers(0, 5, 60), np.where(rng.random(60) < 0.8, labels, 0), np.full(60, 2)])
    expected = [f1_score(labels, row, labels=list(range(5)), average="macro", zero_division=0) for row in predictions]
    np.testing.assert_allclose(macro_f1_rows(labels, predictions, 5), expected)


def test_threshold_grid_is_sorted_unique_and_reaches_the_clean_tail():
    values = np.random.default_rng(0).normal(size=(3, 500))
    grid = threshold_grid(values.ravel(), values[0], 21)
    assert np.all(np.diff(grid) > 0) and grid.max() >= np.quantile(values[0], 0.999) - 1e-6


def test_fit_finds_the_helpful_actions_and_features():
    table = _synthetic_table()
    selector, log = fit(table, CONFIG["rules"], SETTINGS)
    rules = {rule.name: rule for rule in selector.rules}
    assert rules["noise"].action == "bilateral_wide" and rules["noise"].clauses[0].feature in {"noise_sigma", "noise_ratio"}
    assert rules["blur"].action == "unsharp_strong" and rules["blur"].clauses[0].feature == "sharpness"
    frame, summary = report(table, selector)
    assert summary["gain"] > 0.2 and summary["conditions_better"] == 2 and summary["conditions_worse"] == 0
    assert summary["clean_f1_selector"] == pytest.approx(summary["clean_f1_none"]) and summary["clean_trigger_rate"] == 0.0
    assert log["objective"] == pytest.approx(summary["objective_selector"])
    # Khi hai ngưỡng cho cùng mục tiêu, bộ chọn lấy ngưỡng kích hoạt ít ảnh hơn, nên có thể bỏ sót vài ảnh mà hành động không cải thiện.
    assert frame["trigger_rate"].iloc[0] == 0.0 and (frame["trigger_rate"].iloc[1:] >= 0.95).all()
    assert summary["image_level_upper_bound"] >= summary["objective_selector"] - 1e-9


def test_fit_reports_a_negative_result_when_no_action_helps():
    selector, log = fit(_synthetic_table(helpful=False), CONFIG["rules"], SETTINGS)
    assert selector.rules == () and log["objective"] == pytest.approx(log["none_objective"])
    assert all(not entry["accepted"] for entry in log["rules"])


def test_fit_respects_the_clean_constraints_when_features_cannot_separate_clean_images():
    table = _synthetic_table(separable=False)
    selector, _ = fit(table, CONFIG["rules"], SETTINGS)
    _, summary = report(table, selector)
    assert summary["clean_trigger_rate"] <= SETTINGS.max_clean_trigger_rate
    assert summary["clean_f1_selector"] >= summary["clean_f1_none"] - SETTINGS.max_clean_f1_drop - 1e-12


def test_dropping_a_feature_group_disables_the_rules_that_depend_on_it():
    table = _synthetic_table()
    full, _ = fit(table, CONFIG["rules"], SETTINGS)
    without_sharpness, log = fit(table, CONFIG["rules"], SETTINGS, dropped_groups=["sharpness", "naturalness"])
    assert {rule.name for rule in full.rules} == {"noise", "blur"}
    assert "blur" not in {rule.name for rule in without_sharpness.rules}
    assert report(table, without_sharpness)[1]["gain"] < report(table, full)[1]["gain"]
    assert any("không còn đặc trưng ứng viên" in entry.get("reason", "") for entry in log["rules"])


def test_min_gain_drops_a_rule_whose_improvement_is_too_small():
    table = _synthetic_table()
    strict = Settings(**{**SETTINGS.__dict__, "min_gain": 0.9})
    selector, log = fit(table, CONFIG["rules"], strict)
    assert selector.rules == () and all("min_gain" in entry["reason"] for entry in log["rules"])


def test_assign_selector_and_score_agree_with_manual_lookup():
    table = _synthetic_table()
    selector = RuleSelector([Rule("blur", (Clause("sharpness", "<=", 0.2),), "unsharp_strong")])
    assignment = assign_selector(table, selector)
    assert assignment.shape == (3, 100) and set(np.unique(assignment)) <= {0, ACTIONS.index("unsharp_strong")}
    assert (assignment[1] == ACTIONS.index("unsharp_strong")).all() and (assignment[0] == 0).all()
    manual = np.where(assignment == 0, table.predictions[0], table.predictions[ACTIONS.index("unsharp_strong")])
    np.testing.assert_allclose(score(table, assignment), macro_f1_rows(table.labels, manual, 5))
    with pytest.raises(ValueError):
        assign_selector(table, RuleSelector([Rule("x", (Clause("sharpness", "<=", 1.0),), "không_có")]))


def test_cross_validation_reports_heldout_gain_per_fold():
    stability = cross_validate(_synthetic_table(images=150), CONFIG["rules"], SETTINGS, folds=3, seed=42)
    assert len(stability["folds"]) == 3 and stability["folds_with_positive_gain"] == 3
    assert stability["heldout_gain_mean"] > 0.2 and stability["heldout_gain_std"] >= 0.0


def test_table_round_trips_through_npz_and_subset_keeps_alignment(tmp_path):
    table = _synthetic_table()
    table.probabilities = np.random.default_rng(0).random((len(ACTIONS), 3, 100, 5)).astype(np.float16)
    table.save(tmp_path / "table.npz")
    loaded = CalibrationTable.load(tmp_path / "table.npz")
    np.testing.assert_array_equal(loaded.features, table.features)
    np.testing.assert_array_equal(loaded.predictions, table.predictions)
    assert loaded.conditions == table.conditions and loaded.actions == table.actions and loaded.paths == table.paths
    part = loaded.subset(np.array([3, 1]))
    assert part.paths == ["ảnh_3.jpg", "ảnh_1.jpg"] and part.labels.tolist() == [3, 1]
    np.testing.assert_array_equal(part.features[:, 0], table.features[:, 3])
    with pytest.raises(ValueError):
        CalibrationTable([CLEAN], [CLEAN], ["không_phải_keep"], table.features[:1], table.predictions[:1, :1], None, table.labels, table.paths, table.classes)


class _Cache:
    """Giả lập ImageCache: ảnh uint8 tổng hợp, điều kiện suy giảm tạo theo yêu cầu."""

    def __init__(self, count: int) -> None:
        self.images = torch.stack([scene(seed) for seed in range(count)])
        self.labels = np.arange(count) % 5
        self.paths = [f"ảnh_{i}.jpg" for i in range(count)]
        self.classes = [f"lớp_{i}" for i in range(5)]
        self.released: list[str] = []

    def get(self, condition: str) -> torch.Tensor:
        source = self.images if condition == CLEAN else torch.stack([apply_corruption(image, {"type": "blur", "value": 2.0}) for image in self.images])
        return torch.round(source * 255).to(torch.uint8)

    def release(self, condition: str) -> None:
        self.released.append(condition)


def test_collect_builds_a_consistent_table_and_saves_artifacts(tmp_path):
    cache = _Cache(10)
    conditions = {"blur_s1": {"family": "blur", "type": "blur", "value": 2.0}}
    seen = []
    table = collect(tiny_model(), cache, conditions, CONFIG["actions"], {"mean": [0.5] * 3, "std": [0.25] * 3}, torch.device("cpu"),
                    workers=2, progress=lambda _: None, on_condition_done=lambda partial: seen.append(partial.conditions))
    assert table.conditions == [CLEAN, "blur_s1"] and table.families == [CLEAN, "blur"] and table.actions == ACTIONS
    assert table.features.shape == (2, 10, len(FEATURES)) and table.predictions.shape == (5, 2, 10) and table.probabilities.shape == (5, 2, 10, 5)
    assert seen == [[CLEAN]] and cache.released == [CLEAN, "blur_s1"]
    # Đặc trưng tính trên ảnh chưa xử lý: ảnh mờ có độ nét thấp hơn ảnh sạch trên cùng ảnh gốc.
    assert (table.feature("sharpness")[1] < table.feature("sharpness")[0]).all()
    # Hành động `keep` cho đúng dự đoán của model trên ảnh gốc.
    assert (table.predictions[0, 0] == table.probabilities[0, 0].argmax(axis=1)).all()

    selector, log = fit(table, CONFIG["rules"], Settings(min_gain=0.0))
    frame, summary = report(table, selector)
    stability = cross_validate(table, CONFIG["rules"], Settings(min_gain=0.0), folds=2, seed=1)
    path = save_artifacts(tmp_path / "out", selector, log, frame, summary, stability, CONFIG, {"note": "kiểm thử"})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["locked"] is False and record["split"] == "val" and RuleSelector.from_dict(record["selector"]).to_dict() == selector.to_dict()
    assert (tmp_path / "out/calibration_log.json").is_file() and (tmp_path / "out/calibration_conditions_val.csv").is_file()
    plot_by_family(frame, tmp_path / "by_family.png")
    assert (tmp_path / "by_family.png").stat().st_size > 0


def _table_with_low_contrast_lookalikes() -> CalibrationTable:
    """Trong điều kiện `blur_s1`, 40 ảnh đầu không mờ thật mà chỉ có tương phản thấp. Độ nét thấp giống ảnh mờ, nhưng làm nét sẽ làm hỏng chúng."""
    table = _synthetic_table(images=100, seed=3)
    index = {name: FEATURES.index(name) for name in FEATURES}
    lookalike = np.arange(100) < 40
    table.features[1, lookalike, index["luma_std"]] = 0.02
    table.features[1, ~lookalike, index["luma_std"]] = 0.2
    table.features[[0, 2], :, index["luma_std"]] = 0.2
    strong = ACTIONS.index("unsharp_strong")
    table.predictions[0, 1, lookalike] = table.labels[lookalike]
    table.predictions[strong, 1, lookalike] = (table.labels[lookalike] + 1) % 5
    return table


def test_extra_guard_is_added_only_when_it_stops_a_harmful_trigger():
    table = _table_with_low_contrast_lookalikes()
    blur_rule = next(spec for spec in CONFIG["rules"] if spec["name"] == "blur")
    plain = {key: value for key, value in blur_rule.items() if key != "extra_guard"}
    without, _ = fit(table, [plain], SETTINGS)
    with_guard, log = fit(table, [blur_rule], SETTINGS)
    features = [clause.feature for clause in with_guard.rules[0].clauses]
    assert "luma_std" in features
    assert report(table, with_guard)[1]["gain"] > report(table, without)[1]["gain"]
    assert log["rules"][0]["extra_guard"]["feature"] == "luma_std" and log["rules"][0]["extra_guard_gain"] >= SETTINGS.min_gain


def test_extra_guard_is_dropped_when_it_does_not_help():
    table = _synthetic_table()
    blur_rule = next(spec for spec in CONFIG["rules"] if spec["name"] == "blur")
    selector, log = fit(table, [blur_rule], SETTINGS)
    assert log["rules"][0]["extra_guard"] is None
    assert all(clause.feature not in {"luma_std", "dark_fraction"} for clause in selector.rules[0].clauses)

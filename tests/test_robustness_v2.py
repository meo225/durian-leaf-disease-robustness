import math

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation import robustness_v2 as R
from src.evaluation.control_compare import bootstrap_indices
from src.selector.calibrate import macro_f1_rows
from src.selector.rules import Clause, Rule, RuleSelector
from tests.selector_helpers import synthetic_table

CLASSES = ["a", "b", "c"]
NAMES = ["x_s1", "x_s2", "x_s3", "y_s1", "y_s2", "y_s3"]


def _labels(images=90):
    return np.arange(images) % 3


def _matrix(labels, accuracy_by_row, seed=0):
    """(1 + C) × N dự đoán: hàng i đúng với tỉ lệ `accuracy_by_row[i]`, sai thì lệch một lớp."""
    rng = np.random.default_rng(seed)
    rows = []
    for accuracy in accuracy_by_row:
        wrong = (labels + 1) % 3
        rows.append(np.where(rng.random(len(labels)) < accuracy, labels, wrong))
    return np.stack(rows).astype(np.int64)


# --- cấu hình ------------------------------------------------------------------------------------------------------------


def test_default_settings_are_valid_and_exclude_test_role_families():
    settings = R.load_settings()
    assert settings["split"] == "val" and tuple(settings["roles"]) == ("train", "val")
    conditions = R.evaluation_conditions(settings)
    roles = {spec["role"] for spec in conditions.values()}
    assert roles == {"train", "val"}
    assert len(conditions) == 3 * len({spec["family"] for spec in conditions.values()})
    cfg = yaml.safe_load(open("configs/unseen_families_v2.yaml", encoding="utf-8"))
    test_families = {name for name, spec in cfg["families"].items() if spec["role"] == "test"}
    assert test_families and not test_families & {spec["family"] for spec in conditions.values()}


@pytest.mark.parametrize("override", [{"split": "test"}, {"roles": ["train", "val", "test"]}, {"roles": ["test"]}, {"roles": []}])
def test_settings_refuse_the_test_split_and_test_role_families(tmp_path, override):
    settings = yaml.safe_load(open(R.SETTINGS_PATH, encoding="utf-8"))
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({**settings, **override}), encoding="utf-8")
    with pytest.raises(PermissionError):
        R.load_settings(path)


# --- điểm theo họ -------------------------------------------------------------------------------------------------------


def test_family_scores_point_is_the_mean_over_levels_of_macro_f1():
    labels = _labels()
    matrix = _matrix(labels, [0.95, 0.9, 0.8, 0.6, 0.7, 0.5, 0.3])
    scores = R.family_scores(labels, matrix, NAMES, 3, bootstrap_indices(len(labels), 50, 0))
    f1 = macro_f1_rows(labels, matrix[1:], 3)
    assert scores.families == ("x", "y")
    assert scores.point == pytest.approx([f1[:3].mean(), f1[3:].mean()])
    assert scores.boot.shape == (50, 2)


def test_bootstrap_interval_contains_point_and_is_reproducible():
    labels = _labels()
    matrix = _matrix(labels, [0.95, 0.9, 0.8, 0.6, 0.7, 0.5, 0.3])
    indices = bootstrap_indices(len(labels), 400, 7)
    first, second = (R.family_scores(labels, matrix, NAMES, 3, indices) for _ in range(2))
    assert np.array_equal(first.boot, second.boot)
    low, high = first.interval()
    assert np.all(low <= first.point) and np.all(first.point <= high)


def test_family_scores_rejects_wrong_matrix_shape():
    labels = _labels()
    with pytest.raises(ValueError, match="dạng"):
        R.family_scores(labels, _matrix(labels, [0.9] * 6), NAMES, 3, bootstrap_indices(len(labels), 10, 0))


def test_minus_pairs_resamples_so_identical_systems_have_zero_width_difference():
    labels = _labels()
    matrix = _matrix(labels, [0.95, 0.9, 0.8, 0.6, 0.7, 0.5, 0.3])
    indices = bootstrap_indices(len(labels), 100, 1)
    scores = R.family_scores(labels, matrix, NAMES, 3, indices)
    table = R.paired_table(scores, scores)
    assert (table["delta"] == 0).all() and (table["low"] == 0).all() and (table["high"] == 0).all()
    assert set(table["kết luận"]) == {"chưa phân biệt được"}
    assert table["family"].iloc[-1] == R.ALL_FAMILIES


def test_paired_table_detects_a_clear_gain_and_reports_overall_row():
    labels = _labels(300)
    good = _matrix(labels, [0.97] * 7, seed=1)
    bad = _matrix(labels, [0.55] * 7, seed=2)
    indices = bootstrap_indices(len(labels), 300, 3)
    table = R.paired_table(R.family_scores(labels, good, NAMES, 3, indices), R.family_scores(labels, bad, NAMES, 3, indices))
    assert (table["kết luận"] == "tốt hơn").all()
    assert (table["delta"] > 0.2).all()


def test_minus_and_mean_require_the_same_resamples():
    labels = _labels()
    matrix = _matrix(labels, [0.9] * 7)
    a = R.family_scores(labels, matrix, NAMES, 3, bootstrap_indices(len(labels), 10, 0))
    b = R.family_scores(labels, matrix, NAMES, 3, bootstrap_indices(len(labels), 20, 0))
    with pytest.raises(ValueError):
        a.minus(b)
    with pytest.raises(ValueError):
        R.mean_scores([a, b])
    with pytest.raises(ValueError):
        R.mean_scores([])


def test_mean_scores_averages_points_and_resamples():
    labels = _labels()
    indices = bootstrap_indices(len(labels), 20, 0)
    a = R.family_scores(labels, _matrix(labels, [0.9] * 7, 1), NAMES, 3, indices)
    b = R.family_scores(labels, _matrix(labels, [0.6] * 7, 2), NAMES, 3, indices)
    mean = R.mean_scores([a, b])
    assert mean.point == pytest.approx((a.point + b.point) / 2)
    assert mean.boot == pytest.approx((a.boot + b.boot) / 2)


# --- phần lấy lại được ---------------------------------------------------------------------------------------------------


def test_recovered_fraction_values_and_negative_when_system_is_worse():
    system = np.array([0.90, 0.60, 0.80, 0.95])
    base = np.array([0.80, 0.70, 0.79, 0.94])
    out = R.recovered_fraction(system, base, 0.95, levels=3, min_drop=0.05)
    assert out[0] == pytest.approx(0.10 / 0.15)  # lấy lại hai phần ba ΔF1
    assert out[1] == pytest.approx(-0.10 / 0.25)  # tệ hơn base
    assert out[2] == pytest.approx(0.01 / 0.16)
    assert math.isnan(out[3])  # base chỉ giảm 0.01 mỗi mức, tổng 0.03 dưới ngưỡng 0.05


def test_recovered_fraction_is_nan_when_base_barely_drops():
    out = R.recovered_fraction(np.array([0.95]), np.array([0.94]), 0.95, levels=3, min_drop=0.05)
    assert math.isnan(out[0])  # 0.01 × 3 = 0.03 < 0.05
    out = R.recovered_fraction(np.array([0.95]), np.array([0.93]), 0.95, levels=3, min_drop=0.05)
    assert out[0] == pytest.approx(1.0)  # 0.02 × 3 = 0.06 đủ ngưỡng


# --- entropy và chẩn đoán ------------------------------------------------------------------------------------------------


def test_prediction_entropy_bounds():
    assert R.prediction_entropy(np.array([0, 1, 2] * 10), 3) == pytest.approx(1.0)
    assert R.prediction_entropy(np.zeros(30, dtype=int), 3) == pytest.approx(0.0)
    assert 0.0 < R.prediction_entropy(np.array([0] * 20 + [1] * 10), 3) < 1.0
    with pytest.raises(ValueError):
        R.prediction_entropy(np.array([], dtype=int), 3)


def test_condition_diagnostics_flags_collapse_and_reports_small_class_interval():
    labels = np.array([0] * 40 + [1] * 40 + [2] * 10)
    perfect = labels.copy()
    collapsed = np.zeros_like(labels)
    matrix = np.stack([perfect, perfect, collapsed])
    frame = R.condition_diagnostics(labels, matrix, ["x_s1", "x_s2"], ["a", "b", "c"], small_class="c")
    assert list(frame["condition"]) == ["x_s1", "x_s2"]
    assert frame.loc[0, "pred_entropy"] == pytest.approx(frame.loc[0, "label_entropy"])
    assert frame.loc[1, "pred_entropy"] == 0.0 and frame.loc[1, "top_class"] == "a" and frame.loc[1, "top_share"] == 1.0
    assert frame.loc[0, "small_support"] == 10 and frame.loc[0, "small_recall"] == 1.0
    assert frame.loc[1, "small_recall"] == 0.0 and frame.loc[1, "small_low"] == 0.0 and 0.0 < frame.loc[1, "small_high"] < 0.5
    with pytest.raises(ValueError, match="lớp"):
        R.condition_diagnostics(labels, matrix, ["x_s1", "x_s2"], ["a", "b", "c"], small_class="zzz")


# --- so sánh nhiều seed --------------------------------------------------------------------------------------------------


def _arm(points, resamples=20):
    """Nhánh nhiều seed dựng thẳng từ điểm theo họ, mẫu bootstrap giả cho đủ dạng."""
    return {seed: R.FamilyScores(("x", "y"), np.array(point), np.tile(point, (resamples, 1))) for seed, point in zip((42, 43, 44), points)}


def test_compare_seed_arms_applies_the_sd_rule_per_family_and_overall():
    a = _arm([[0.90, 0.50], [0.91, 0.60], [0.89, 0.40]])  # y nhiễu seed lớn
    b = _arm([[0.80, 0.52], [0.81, 0.58], [0.79, 0.50]])
    table = R.compare_seed_arms(a, b).set_index("family")
    assert table.loc["x", "delta"] == pytest.approx(0.10)
    assert table.loc["x", "kết luận"] == "vượt"  # hiệu 0.10, SD gộp 0.01
    assert table.loc["y", "kết luận"] == "chưa kết luận"  # hiệu 0.0, SD lớn
    assert list(table.index)[-1] == R.ALL_FAMILIES


def test_compare_seed_arms_needs_two_seeds_for_a_verdict():
    one = {42: R.FamilyScores(("x", "y"), np.array([0.9, 0.9]), np.zeros((5, 2)))}
    assert set(R.compare_seed_arms(one, one)["kết luận"]) == {"chưa đủ seed"}


def test_compare_single_uses_reference_sd():
    reference = _arm([[0.80, 0.50], [0.81, 0.52], [0.79, 0.48]])
    single = R.FamilyScores(("x", "y"), np.array([0.90, 0.51]), np.zeros((5, 2)))
    table = R.compare_single_to_seed_arm(single, reference).set_index("family")
    assert table.loc["x", "kết luận"] == "vượt"
    assert table.loc["y", "kết luận"] == "chưa kết luận"
    assert table.loc["x", "reference_sd"] == pytest.approx(0.01)


def test_seed_spread_and_family_mismatch_errors():
    spread = R.seed_spread(_arm([[0.9, 0.5], [0.8, 0.6], [0.7, 0.7]])).set_index("family")
    assert spread.loc["x", "mean"] == pytest.approx(0.8) and spread.loc["x", "sd"] == pytest.approx(0.1)
    other = {42: R.FamilyScores(("x", "z"), np.array([0.9, 0.9]), np.zeros((5, 2)))}
    with pytest.raises(ValueError, match="họ"):
        R.compare_single_to_seed_arm(other[42], _arm([[0.9, 0.9]] * 3))


# --- bốn hệ trên bảng tổng hợp -------------------------------------------------------------------------------------------


def _selector_for_blur():
    """Chỉ chọn unsharp_strong khi độ nét rất thấp, như luật mờ của bộ chọn đã khóa."""
    return RuleSelector([Rule("blur", (Clause("sharpness", "<=", 0.3),), "unsharp_strong")], default="keep")


def test_system_predictions_orders_none_selector_and_oracles():
    table = synthetic_table()
    predictions = R.system_predictions(table, _selector_for_blur())
    assert set(predictions) == set(R.SYSTEMS)
    shapes = {matrix.shape for matrix in predictions.values()}
    assert shapes == {(len(table.conditions), len(table.labels))}
    f1 = {name: macro_f1_rows(table.labels, matrix, table.num_classes) for name, matrix in predictions.items()}
    # Bộ chọn chữa họ blur (cột 1) mà không đụng ảnh sạch (cột 0) và nhiễu (cột 2).
    assert f1["selector"][1] > f1["none"][1]
    assert f1["selector"][0] == pytest.approx(f1["none"][0]) and f1["selector"][2] == pytest.approx(f1["none"][2])
    # Cận trên không thấp hơn bộ chọn và không xử lý ở bất kỳ điều kiện nào: oracle theo ảnh ≥ theo điều kiện ≥ theo họ.
    for name in ("none", "selector"):
        assert np.all(f1["oracle_condition"] >= f1[name] - 1e-12)
        assert np.all(f1["oracle_image"] >= f1[name] - 1e-12)
    assert np.all(f1["oracle_image"] >= f1["oracle_condition"] - 1e-12)
    assert np.all(f1["oracle_condition"] >= f1["oracle_family"] - 1e-12)


def test_action_rates_sum_to_one_and_show_where_the_selector_fires():
    table = synthetic_table()
    rates = R.action_rates(table, _selector_for_blur()).set_index("condition")
    assert rates[list(table.actions)].sum(axis=1).tolist() == pytest.approx([1.0] * len(rates))
    assert rates.loc["blur_s1", "processed_rate"] > 0.9
    assert rates.loc["clean", "processed_rate"] == 0.0 and rates.loc["noise_s1", "processed_rate"] == 0.0


# --- MSD ----------------------------------------------------------------------------------------------------------------------


def test_msd_correlation_sign_interval_and_minimum_families():
    msd = {f"f{i}": float(i) for i in range(8)}
    effect = {name: -value + (0.1 if value == 3 else 0.0) for name, value in msd.items()}
    result = R.msd_correlation(msd, effect)
    assert result["rho"] < -0.9 and result["n"] == 8 and result["low"] <= result["rho"] <= result["high"] + 1e-9
    with pytest.raises(ValueError, match="ít nhất"):
        R.msd_correlation({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 2.0, "c": 3.0})


def test_msd_correlation_ignores_missing_and_nan_values():
    msd = {f"f{i}": float(i) for i in range(7)}
    effect = {f"f{i}": float(i) for i in range(6)} | {"f6": float("nan")}
    assert R.msd_correlation(msd, effect, min_families=5)["n"] == 6.0


# --- 14 họ đã thấy và MSD ------------------------------------------------------------------------------------------------


def _frame(values):
    rows = [{"condition": "clean", "family": "clean", "macro_f1": 0.97}]
    rows += [{"condition": f"{family}_s{level}", "family": family, "macro_f1": value} for family, scores in values.items() for level, value in enumerate(scores, start=1)]
    return pd.DataFrame(rows)


def test_family_means_skips_clean_and_keeps_order():
    means = R.family_means(_frame({"blur": [0.9, 0.8, 0.7], "noise": [0.6, 0.5, 0.4]}))
    assert list(means.index) == ["blur", "noise"]
    assert means["blur"] == pytest.approx(0.8) and means["noise"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        R.family_means(pd.DataFrame([{"condition": "clean", "family": "clean", "macro_f1": 0.9}]))


def test_seen_effect_averages_seeds_before_subtracting():
    arm = [_frame({"blur": [0.9] * 3, "noise": [0.7] * 3}), _frame({"blur": [0.8] * 3, "noise": [0.7] * 3})]
    base = [_frame({"blur": [0.6] * 3, "noise": [0.6] * 3})]
    effect = R.seen_effect(arm, base)
    assert effect == pytest.approx({"blur": 0.25, "noise": 0.10})
    with pytest.raises(ValueError, match="họ"):
        R.seen_effect(arm, [_frame({"blur": [0.6] * 3})])
    with pytest.raises(ValueError):
        R.seen_effect([], base)


def test_msd_vectors_cover_fourteen_seen_and_nine_new_families():
    import json

    record = json.load(open("reports/unseen_v2/msd.json", encoding="utf-8"))
    vectors = R.msd_vectors(record)
    assert len(vectors["ta_seen"]) == 14 and len(vectors["ta_new"]) == 9 and len(vectors["aug_new"]) == 9
    assert set(vectors["ta_new"]) == set(vectors["aug_new"])
    assert all(value > 0 for table in vectors.values() for value in table.values())


# --- chạy model trên ảnh dựng sẵn -----------------------------------------------------------------------------------------


class _FakeCache:
    def __init__(self, count=12):
        from tests.selector_helpers import uint8_images

        self.images = uint8_images(count)
        self.calls = []

    def __len__(self):
        return len(self.images)

    def get(self, name):
        self.calls.append(name)
        return self.images


def test_model_matrix_has_clean_row_first_and_one_row_per_condition():
    import torch

    from tests.selector_helpers import tiny_model

    cache = _FakeCache()
    cfg = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    matrix = R.model_matrix(tiny_model(), cache, ["x_s1", "x_s2"], cfg, torch.device("cpu"), batch_size=5)
    assert matrix.shape == (3, 12) and matrix.dtype == np.int64
    assert cache.calls == ["clean", "x_s1", "x_s2"]
    assert matrix.min() >= 0 and matrix.max() < 5


def test_retained_cache_never_releases_images():
    cache = object.__new__(R.RetainedCache)
    cache._cache = {"clean": object()}
    cache.release("clean")
    assert "clean" in cache._cache


# --- ghép bảng ----------------------------------------------------------------------------------------------------------


SETTINGS = {"statistics": {"bootstrap_resamples": 100, "bootstrap_seed": 1, "k": 2.5, "min_reference_drop": 0.05, "small_class": "c"}}
FIVE = ["x_s1", "x_s2", "x_s3", "y_s1", "y_s2", "y_s3"]


def _system_set(labels, accuracy, seed):
    """Năm hệ giả trên reference: bộ chọn và oracle hơn không xử lý ở họ x (ba hàng đầu), bằng nhau ở họ y."""
    none = _matrix(labels, [0.97, *accuracy], seed)
    better = none.copy()
    better[1:4] = _matrix(labels, [0.9, 0.9, 0.9], seed + 10)[:3]
    return {"none": none, "selector": better, "oracle_family": better, "oracle_condition": better, "oracle_image": better}


def _tables(side=True):
    labels = _labels(150)
    reference = _system_set(labels, [0.5, 0.5, 0.5, 0.7, 0.7, 0.7], 1)
    retrained = {
        "baseline": {seed: _matrix(labels, [0.97, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7], seed) for seed in (42, 43, 44)},
        "degradation": {seed: _matrix(labels, [0.97, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], seed + 5) for seed in (42, 43, 44)},
        "trivialaugment": {seed: _matrix(labels, [0.97, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8], seed + 9) for seed in (42, 43, 44)},
        "trivialaugment_geometric": {42: _matrix(labels, [0.97, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7], 77)},
        "degradation_leave_group": {42: _matrix(labels, [0.97, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9], 78)},
    }
    side_check = _system_set(labels, [0.5, 0.5, 0.5, 0.7, 0.7, 0.7], 3) if side else None
    return R.build_tables(labels, FIVE, CLASSES, reference, retrained, SETTINGS, side_check)


def test_build_tables_returns_every_expected_table_with_consistent_families():
    tables = _tables()
    assert set(tables) == {"condition_scores", "systems_vs_none", "retrained_comparison", "retrained_vs_baseline_ci", "seed_spread", "recovered_fraction", "diagnostics", "side_check_new_baseline"}
    assert "side_check_new_baseline" not in _tables(side=False)
    for name in ("systems_vs_none", "retrained_vs_baseline_ci", "side_check_new_baseline"):
        assert set(tables[name]["family"]) == {"x", "y", R.ALL_FAMILIES}, name
    scores = tables["condition_scores"]
    assert len(scores) == (1 + len(FIVE)) * (len(R.SYSTEMS) + 3 * 3 + 2)
    assert set(scores["model"]) == {"reference_v1", "retrained"}


def test_build_tables_systems_vs_none_finds_gain_only_where_built():
    table = _tables()["systems_vs_none"]
    selector = table[table["system"] == "selector"].set_index("family")
    assert selector.loc["x", "kết luận"] == "tốt hơn" and selector.loc["x", "delta"] > 0.2
    assert selector.loc["y", "delta"] == 0 and selector.loc["y", "kết luận"] == "chưa phân biệt được"


def test_build_tables_retrained_comparison_uses_the_sd_rule_and_labels_pairs():
    table = _tables()["retrained_comparison"]
    pairs = set(table["pair"])
    assert {"trivialaugment − baseline", "degradation − baseline", "degradation − trivialaugment", "trivialaugment_geometric − baseline", "degradation_leave_group − degradation"} <= pairs
    degradation = table[table["pair"] == "degradation − baseline"].set_index("family")
    assert degradation.loc["x", "kết luận"] == "vượt" and degradation.loc["x", "delta"] > 0.2
    geometric = table[table["pair"] == "trivialaugment_geometric − baseline"].set_index("family")
    assert geometric.loc["x", "kết luận"] == "chưa kết luận"


def test_build_tables_recovered_fraction_uses_the_right_base_and_nan_for_small_drops():
    table = _tables()["recovered_fraction"]
    assert set(table["base"]) == {"reference_v1", "baseline_5.9"}
    assert set(table.loc[table["base"] == "reference_v1", "system"]) == {"selector", *R.ORACLES}
    assert set(table.loc[table["base"] == "baseline_5.9", "system"]) == {"degradation", "trivialaugment", "trivialaugment_geometric", "degradation_leave_group"}
    degradation_x = table[(table["system"] == "degradation") & (table["family"] == "x")].iloc[0]
    assert degradation_x["base_drop"] > 0.2 and 0.5 < degradation_x["recovered"] <= 1.2
    selector_y = table[(table["system"] == "selector") & (table["family"] == "y")].iloc[0]
    assert selector_y["recovered"] == pytest.approx(0.0)


def test_build_tables_diagnostics_cover_reference_systems_and_seed_42_arms():
    table = _tables()["diagnostics"]
    assert set(table["system"]) == {"none", "selector", "oracle_condition", "baseline", "degradation", "trivialaugment"}
    assert len(table) == 6 * len(FIVE)


def _drop_extra_baseline_seeds(reference, retrained):
    del retrained["baseline"][43], retrained["baseline"][44]


@pytest.mark.parametrize("break_it, message", [
    (lambda ref, rt: ref.pop("oracle_image"), "reference"),
    (lambda ref, rt: rt.pop("degradation"), "2 seed"),
    (_drop_extra_baseline_seeds, "2 seed"),
    (lambda ref, rt: rt.update({"mystery": {42: rt["baseline"][42]}}), "không nhận ra"),
])
def test_build_tables_rejects_incomplete_inputs(break_it, message):
    labels = _labels(60)
    reference = _system_set(labels, [0.5] * 6, 1)
    retrained = {arm: {seed: _matrix(labels, [0.9] * 7, seed) for seed in (42, 43, 44)} for arm in R.THREE_SEED_ARMS}
    break_it(reference, retrained)
    with pytest.raises(ValueError, match=message):
        R.build_tables(labels, FIVE, CLASSES, reference, retrained, SETTINGS)


def test_build_tables_rejects_uneven_levels_per_family():
    labels = _labels(60)
    names = ["x_s1", "x_s2", "x_s3", "y_s1"]
    reference = _system_set(labels, [0.5] * 6, 1)
    reference = {key: value[:5] for key, value in reference.items()}
    retrained = {arm: {seed: _matrix(labels, [0.9] * 5, seed) for seed in (42, 43, 44)} for arm in R.THREE_SEED_ARMS}
    with pytest.raises(ValueError, match="số mức"):
        R.build_tables(labels, names, CLASSES, reference, retrained, SETTINGS)


def test_correlation_table_pools_seen_families_for_trivialaugment_only():
    vectors = {
        "ta_seen": {f"s{i}": float(i) for i in range(1, 15)},
        "ta_new": {"n1": 3.0, "n2": 5.0, "n3": 7.0, "outside": 1.0},
        "aug_new": {"n1": 1.0, "n2": 2.0, "n3": 3.0, "n4": 4.0, "n5": 5.0, "outside": 9.0},
    }
    seen_effect = {f"s{i}": -0.01 * i for i in range(1, 15)}
    new = {"trivialaugment": {"n1": -0.03, "n2": -0.05, "n3": -0.07}, "degradation": {f"n{i}": -0.1 * i for i in range(1, 6)}}
    table = R.correlation_table(vectors, seen_effect, new, scope_families=["n1", "n2", "n3", "n4", "n5"]).set_index("arm")
    assert table.loc["trivialaugment", "n"] == 17 and table.loc["degradation", "n"] == 5
    assert table.loc["trivialaugment", "rho"] < -0.9 and table.loc["degradation", "rho"] == pytest.approx(-1.0)
    assert list(table["msd"]) == ["trivialaugment", "degradation_aug"]


def test_correlation_table_refuses_name_collisions_and_too_few_families():
    vectors = {"ta_seen": {"blur": 1.0}, "ta_new": {"blur": 2.0}, "aug_new": {"blur": 2.0}}
    with pytest.raises(ValueError, match="trùng"):
        R.correlation_table(vectors, {"blur": 0.1}, {"trivialaugment": {"blur": 0.1}, "degradation": {"blur": 0.1}}, ["blur"])
    vectors = {"ta_seen": {f"s{i}": float(i) for i in range(6)}, "ta_new": {"n": 1.0}, "aug_new": {"n": 1.0}}
    with pytest.raises(ValueError, match="ít nhất"):
        R.correlation_table(vectors, {f"s{i}": float(i) for i in range(6)}, {"trivialaugment": {"n": 0.0}, "degradation": {"n": 0.0}}, ["n"])

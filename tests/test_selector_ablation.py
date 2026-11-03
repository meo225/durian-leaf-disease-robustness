import json

import numpy as np
import pandas as pd
import pytest
from selector_helpers import synthetic_table

from src.selector.ablation import (
    add_cross_validation,
    bootstrap_indices,
    build_variants,
    condition_oracle_predictions,
    condition_table,
    family_gain,
    objective_of,
    oracle_predictions,
    restrict_actions,
    save_results,
    strip_guards,
    summarize,
)
from src.selector.ablation_plots import (
    plot_contribution,
    plot_family_heatmap,
    plot_tradeoff,
    plot_variants,
)
from src.selector.calibrate import Settings, macro_f1_rows
from src.selector.pipeline import load_selector_config
from src.selector.rules import Clause, Rule, RuleSelector

CONFIG = load_selector_config("configs/selector.yaml")
SETTINGS = Settings.from_config(CONFIG["calibration"])
TECHNIQUES = {name: spec["technique"] for name, spec in CONFIG["actions"].items() if spec}


@pytest.fixture(scope="module")
def built():
    table = synthetic_table(images=100)
    variants = build_variants(table, CONFIG["rules"], SETTINGS, TECHNIQUES)
    return table, variants


def _by_name(variants):
    return {variant.name: variant for variant in variants}


def test_restrict_actions_filters_actions_and_drops_empty_rules():
    rules = [{"name": "a", "actions": ["x", "y"]}, {"name": "b", "actions": ["z"]}]
    assert restrict_actions(rules, ["y"]) == [{"name": "a", "actions": ["x"]}, {"name": "b", "actions": ["z"]}]
    assert [rule["name"] for rule in restrict_actions(rules, ["z"])] == ["a"]
    assert restrict_actions(rules, ["x", "y", "z"]) == []


def test_strip_guards_keeps_only_the_primary_clause():
    rule = Rule("r", (Clause("sharpness", "<=", 0.1), Clause("luma_mean", ">=", 0.3)), "unsharp_strong")
    stripped = strip_guards(RuleSelector([rule]))
    assert stripped.rules[0].clauses == (Clause("sharpness", "<=", 0.1),)


def test_every_variant_has_aligned_predictions_and_unique_names(built):
    table, variants = built
    names = [variant.name for variant in variants]
    assert len(set(names)) == len(names)
    assert {"none", "full", "oracle"} <= set(names)
    for variant in variants:
        assert variant.predictions.shape == table.predictions[0].shape


def test_reference_variants_match_direct_lookup(built):
    table, variants = built
    by_name = _by_name(variants)
    np.testing.assert_array_equal(by_name["none"].predictions, table.predictions[0])
    for index, action in enumerate(table.actions[1:], start=1):
        np.testing.assert_array_equal(by_name[f"fixed:{action}"].predictions, table.predictions[index])


def test_oracle_is_an_upper_bound_of_every_variant(built):
    table, variants = built
    summary = summarize(table, variants, resamples=20)
    oracle = summary.loc[summary["variant"] == "oracle", "objective"].iloc[0]
    assert (summary["objective"] <= oracle + 1e-9).all()
    oracle = oracle_predictions(table)
    labels = np.broadcast_to(table.labels, oracle.shape)
    any_correct = (table.predictions == table.labels[None, None, :]).any(axis=0)
    assert (oracle[any_correct] == labels[any_correct]).all()
    np.testing.assert_array_equal(oracle[~any_correct], table.predictions[0][~any_correct])


def test_oracle_bounds_are_ordered_from_family_to_image(built):
    table, variants = built
    summary = summarize(table, variants, resamples=10).set_index("variant")
    family, condition, image = (summary.loc[n, "objective"] for n in ("oracle_family", "oracle_condition", "oracle"))
    assert family <= condition + 1e-9 <= image + 2e-9
    assert condition >= summary.loc[[v for v in summary.index if v.startswith("fixed:")], "objective"].max() - 1e-9
    chosen = condition_oracle_predictions(table)
    assert any(np.array_equal(chosen[c], table.predictions[a, c]) for a in range(len(table.actions)) for c in range(1, len(table.conditions)))


def test_full_selector_improves_on_no_processing_when_actions_help(built):
    table, variants = built
    summary = summarize(table, variants, resamples=20).set_index("variant")
    assert summary.loc["full", "gain_vs_none"] > 0.05
    assert summary.loc["none", "gain_vs_none"] == 0
    assert summary.loc["full", "delta_vs_full"] == 0


def test_dropping_the_feature_group_a_rule_needs_loses_that_gain(built):
    table, variants = built
    summary = summarize(table, variants, resamples=20).set_index("variant")
    assert summary.loc["drop_group:sharpness", "objective"] < summary.loc["full", "objective"]
    assert summary.loc["drop_group:color", "objective"] == pytest.approx(summary.loc["full", "objective"])


def test_dropping_a_technique_removes_its_actions_from_the_selector(built):
    _, variants = built
    by_name = _by_name(variants)
    used = by_name["drop_technique:unsharp"].selector.actions_used()
    assert not {name for name, tech in TECHNIQUES.items() if tech == "unsharp"} & used
    assert by_name["drop_action:bilateral_wide"].selector is not None


def test_single_rule_variants_and_no_guards_only_remove_clauses(built):
    table, variants = built
    by_name = _by_name(variants)
    full_rules = {rule.name for rule in by_name["full"].selector.rules}
    for name in full_rules:
        assert [rule.name for rule in by_name[f"only_rule:{name}"].selector.rules] == [name]
    if "no_guards" in by_name:
        assert all(len(rule.clauses) == 1 for rule in by_name["no_guards"].selector.rules)


def test_locked_selector_is_reported_as_given():
    table = synthetic_table(images=100)
    locked = RuleSelector([Rule("blur", (Clause("sharpness", "<=", 0.3),), "unsharp_strong")])
    variants = build_variants(table, CONFIG["rules"], SETTINGS, TECHNIQUES, locked=locked)
    assert _by_name(variants)["locked"].selector is locked


def test_bootstrap_is_paired_and_reproducible(built):
    table, variants = built
    assert np.array_equal(bootstrap_indices(50, 7, seed=1), bootstrap_indices(50, 7, seed=1))
    first = summarize(table, variants, resamples=30, seed=3)
    second = summarize(table, variants, resamples=30, seed=3)
    pd.testing.assert_frame_equal(first, second)
    row = first.set_index("variant").loc["full"]
    assert row["delta_ci_low"] == 0 == row["delta_ci_high"]
    assert row["objective_ci_low"] <= row["objective"] <= row["objective_ci_high"]


def test_summary_objective_matches_macro_f1_of_predictions(built):
    table, variants = built
    summary = summarize(table, variants, resamples=10).set_index("variant")
    for variant in variants:
        expected = objective_of(macro_f1_rows(table.labels, variant.predictions, table.num_classes))
        assert summary.loc[variant.name, "objective"] == pytest.approx(expected)


def test_cross_validation_is_added_only_to_refit_variants(built):
    table, variants = built
    summary = summarize(table, variants, resamples=10)
    result = add_cross_validation(summary, table, CONFIG["rules"], SETTINGS, TECHNIQUES, folds=3, seed=0).set_index("variant")
    assert result.loc["full", "cv_gain_mean"] > 0
    assert np.isnan(result.loc["none", "cv_gain_mean"])
    assert np.isnan(result.loc["oracle", "cv_gain_mean"])
    assert not np.isnan(result.loc["drop_group:noise", "cv_gain_mean"])


def test_family_gain_is_relative_to_no_processing(built):
    table, variants = built
    gains = family_gain(condition_table(table, variants))
    assert (gains.loc["none"] == 0).all()
    assert set(gains.columns) == {"blur", "noise"}
    assert gains.loc["full", "blur"] > 0


def test_save_results_and_plots_write_files(built, tmp_path):
    table, variants = built
    summary = add_cross_validation(summarize(table, variants, resamples=10), table, CONFIG["rules"], SETTINGS, TECHNIQUES, folds=3, seed=0)
    conditions = condition_table(table, variants)
    path = save_results(tmp_path, summary, conditions, variants, {"images": 100})
    assert path.exists() and (tmp_path / "ablation_conditions_val.csv").exists()
    record = json.loads((tmp_path / "ablation_selectors.json").read_text(encoding="utf-8"))
    assert "full" in record["selectors"] and "none" not in record["selectors"]

    plot_variants(summary, tmp_path / "variants.png")
    plot_contribution(summary, ["feature_group"], "Nhóm đặc trưng", tmp_path / "groups.png")
    plot_contribution(summary, ["action", "technique", "rule", "guard"], "Hành động và luật", tmp_path / "actions.png")
    plot_family_heatmap(family_gain(conditions), summary, tmp_path / "heatmap.png")
    plot_tradeoff(summary, tmp_path / "tradeoff.png")
    for name in ("variants", "groups", "actions", "heatmap", "tradeoff"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000

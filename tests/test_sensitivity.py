import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from src.evaluation.sensitivity import (
    bootstrap_indices,
    compare_splits,
    condition_table,
    family_summary,
    fragile_samples,
    flip_destinations,
    holm_adjust,
    image_quality,
    macro_f1_batch,
    mcnemar_exact,
    priority_for_next_task,
    quality_summary,
    recorded_calibration,
)

CLASSES = ["A", "B", "C"]
CFG = {
    "bootstrap": {"resamples": 300, "seed": 1, "confidence": 0.95},
    "alpha": 0.05,
    "thresholds": {"severe": 0.15, "moderate": 0.05, "mild": 0.02},
    "early_drop": 0.05,
    "handlers": {"fam_a": ["4.3"], "fam_b": []},
}


def one_hot(pred, num=3):
    out = np.zeros((len(pred), num))
    out[np.arange(len(pred)), pred] = 1.0
    return out


def make_case():
    labels = np.tile(np.arange(3), 40)
    clean = labels.copy()
    clean[:6] = (clean[:6] + 1) % 3
    heavy = clean.copy()
    heavy[6:66] = (heavy[6:66] + 1) % 3
    light = clean.copy()
    light[6:9] = (light[6:9] + 1) % 3
    probs = {"clean": one_hot(clean), "fam_a_s1": one_hot(light), "fam_a_s2": one_hot(heavy), "fam_b_s1": one_hot(clean)}
    conditions = {
        name: {"family": name.rsplit("_s", 1)[0], "severity": int(name[-1]), "group": "plan", "value": 1.0}
        for name in ["fam_a_s1", "fam_a_s2", "fam_b_s1"]
    }
    return labels, probs, conditions


def test_macro_f1_batch_matches_sklearn():
    labels, probs, _ = make_case()
    pred = probs["fam_a_s2"].argmax(axis=1)
    value = macro_f1_batch(labels, pred, np.arange(len(labels))[None, :], 3)[0]
    assert value == pytest.approx(f1_score(labels, pred, average="macro"))


def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5**10)


def test_holm_is_monotone_and_bounded():
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[2] == pytest.approx(0.09)
    assert adjusted[1] == pytest.approx(0.09)
    assert adjusted.max() <= 1.0


def test_bootstrap_indices_reproducible():
    assert np.array_equal(bootstrap_indices(50, 10, 3), bootstrap_indices(50, 10, 3))


def test_condition_table_counts_and_intervals():
    labels, probs, conditions = make_case()
    table = condition_table(labels, probs, CLASSES, conditions, CFG).set_index("condition")
    assert table.loc["fam_a_s2", "right_to_wrong"] > table.loc["fam_a_s1", "right_to_wrong"]
    assert table.loc["fam_b_s1", "right_to_wrong"] == 0
    assert table.loc["fam_b_s1", "degradation"] == pytest.approx(0.0)
    for name in table.index:
        row = table.loc[name]
        assert row["f1_low"] <= row["macro_f1"] <= row["f1_high"]
        assert row["degradation_low"] <= row["degradation"] <= row["degradation_high"]
    assert bool(table.loc["fam_a_s2", "significant"])
    assert not bool(table.loc["fam_b_s1", "significant"])
    assert table.attrs["clean"]["clean_macro_f1"] == pytest.approx(f1_score(labels, probs["clean"].argmax(axis=1), average="macro"))


def test_family_summary_and_priority():
    labels, probs, conditions = make_case()
    table = condition_table(labels, probs, CLASSES, conditions, CFG)
    summary = family_summary(table, CFG)
    assert summary.iloc[0]["family"] == "fam_a"
    assert summary.iloc[0]["category"] in {"severe", "moderate"}
    assert summary.set_index("family").loc["fam_b", "category"] == "negligible"
    plan = priority_for_next_task(summary, table, CFG)
    assert plan["ranking"][0]["family"] == "fam_a"
    assert "fam_b" in plan["negligible"]
    assert any(item["condition"] == "fam_a_s2" for item in plan["damaging_conditions"])


def test_flip_destinations_and_fragile():
    labels, probs, conditions = make_case()
    names = list(conditions)
    flips = flip_destinations(labels, probs, CLASSES, names).set_index("condition")
    assert flips.loc["fam_a_s2", "lost"] > 0
    assert sum(flips.loc["fam_a_s2", f"to_{c}"] for c in CLASSES) == pytest.approx(1.0)
    assert flips.loc["fam_b_s1", "lost"] == 0
    fragile = fragile_samples(labels, [f"img{i}" for i in range(len(labels))], probs, CLASSES, names)
    assert (fragile["flips"].diff().dropna() <= 0).all()
    assert fragile["flips"].iloc[0] >= 1


def test_compare_splits_identical():
    labels, probs, conditions = make_case()
    table = condition_table(labels, probs, CLASSES, conditions, CFG)
    result = compare_splits(table, table)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["mean_abs_f1_gap"] == 0.0


def test_image_quality_constant_and_noisy():
    flat = torch.full((2, 3, 64, 64), 128, dtype=torch.uint8)
    stats = image_quality(flat)
    assert stats["brightness"].iloc[0] == pytest.approx(128 / 255, abs=1e-3)
    assert stats["contrast"].iloc[0] == pytest.approx(0.0, abs=1e-4)
    assert stats["sharpness"].iloc[0] == pytest.approx(0.0, abs=1e-3)
    assert stats["noise"].iloc[0] == pytest.approx(0.0, abs=1e-3)
    generator = torch.Generator().manual_seed(0)
    noisy = (flat.float() + torch.randn(flat.shape, generator=generator) * 10).clamp(0, 255).round().to(torch.uint8)
    estimate = image_quality(noisy)["noise"].mean()
    assert 5 < estimate < 12
    bright = torch.full((1, 3, 16, 16), 255, dtype=torch.uint8)
    assert image_quality(bright)["saturated"].iloc[0] == pytest.approx(1.0)


def test_quality_summary_groups_by_condition():
    frame = image_quality(torch.randint(0, 255, (6, 3, 32, 32), dtype=torch.uint8))
    frame["condition"] = ["clean"] * 3 + ["blur_s1"] * 3
    summary = quality_summary(frame)
    assert list(summary["condition"]) == ["clean", "blur_s1"]
    assert "noise_p90" in summary.columns


def test_recorded_calibration_reads_all_conditions():
    recorded = recorded_calibration()
    assert len(recorded) == 42
    assert recorded["brightness_dark_s3"] == pytest.approx(0.604)
    assert recorded["color_cast_s1"] == pytest.approx(0.944)

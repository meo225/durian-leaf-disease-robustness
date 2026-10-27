import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation.recovery_stats import PairedBootstrap, compare, holm, mcnemar_exact, normalized_recovery


def _data(n=600, seed=0, accuracy=0.8):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, n)
    wrong = rng.random(n) > accuracy
    predictions = np.where(wrong, rng.integers(0, 5, n), labels)
    return labels, predictions


def test_macro_f1_matches_sklearn_including_missing_class():
    labels, predictions = _data()
    boot = PairedBootstrap(labels, 5, replicates=50)
    assert boot.macro_f1(predictions) == pytest.approx(f1_score(labels, predictions, average="macro", labels=list(range(5)), zero_division=0))
    collapsed = np.where(predictions == 4, 3, predictions)  # lớp 4 không bao giờ được dự đoán
    assert boot.macro_f1(collapsed) == pytest.approx(f1_score(labels, collapsed, average="macro", labels=list(range(5)), zero_division=0))


def test_bootstrap_samples_center_on_point_estimate_and_are_reproducible():
    labels, predictions = _data()
    first = PairedBootstrap(labels, 5, replicates=300, seed=1).macro_f1_samples(predictions)
    second = PairedBootstrap(labels, 5, replicates=300, seed=1).macro_f1_samples(predictions)
    assert np.array_equal(first, second)
    point = PairedBootstrap(labels, 5, replicates=10).macro_f1(predictions)
    low, high = PairedBootstrap.interval(first)
    assert low < point < high and high - low < 0.12
    assert abs(first.mean() - point) < 0.01


def test_mcnemar_counts_and_pvalue():
    base = np.array([True] * 10 + [False] * 10 + [True] * 5)
    other = np.array([True] * 10 + [True] * 10 + [False] * 5)
    result = mcnemar_exact(base, other)
    assert result["gains"] == 10 and result["harms"] == 5
    assert result["p"] == pytest.approx(0.3018, abs=1e-3)
    assert mcnemar_exact(base, base)["p"] == 1.0


def test_holm_known_values_and_dict_form():
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm({"a": 0.5, "b": 0.001})["b"] == pytest.approx(0.002) and holm({"a": 0.5, "b": 0.001})["a"] == pytest.approx(0.5)


def test_compare_detects_real_gain_and_no_change():
    labels, base = _data(n=800, seed=2, accuracy=0.6)
    better = np.where(np.random.default_rng(3).random(800) < 0.5, labels, base)
    boot = PairedBootstrap(labels, 5, replicates=400)
    gain = compare(boot, base, better)
    assert gain["delta"] > 0.05 and gain["delta_low"] > 0 and gain["p"] < 0.001 and gain["gains"] > gain["harms"]
    same = compare(boot, base, base)
    assert same["delta"] == 0 and same["delta_low"] == 0 and same["delta_high"] == 0 and same["p"] == 1.0


def test_normalized_recovery_full_none_and_small_degradation():
    labels, clean = _data(n=800, seed=4, accuracy=0.95)
    corrupted = np.where(np.random.default_rng(5).random(800) < 0.6, clean, (clean + 1) % 5)
    boot = PairedBootstrap(labels, 5, replicates=400)
    full = normalized_recovery(boot, clean, corrupted, clean)
    assert full["normalized_recovery"] == pytest.approx(1.0) and full["recovery"] == pytest.approx(full["degradation"])
    assert full["normalized_low"] <= 1.0 <= full["normalized_high"]
    none = normalized_recovery(boot, clean, corrupted, corrupted)
    assert none["recovery"] == 0 and none["normalized_recovery"] == 0
    tiny = normalized_recovery(boot, clean, clean, clean)
    assert np.isnan(tiny["normalized_recovery"]) and np.isnan(tiny["normalized_low"])

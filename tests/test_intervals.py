import math

import numpy as np
import pytest

from src.evaluation.intervals import bootstrap_ci, per_class_recall_wilson, spearman_with_ci, wilson_interval


def test_wilson_matches_known_values():
    low, high = wilson_interval(50, 100)
    assert (low, high) == (pytest.approx(0.4038, abs=1e-4), pytest.approx(0.5962, abs=1e-4))
    assert wilson_interval(0, 10) == (0.0, pytest.approx(0.2775, abs=1e-4))
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-4) and high == pytest.approx(1.0)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_allocaridara_example_from_the_review():
    # recall khoảng 0.25 trên 46 ảnh: khoảng 95% rộng từ khoảng 0.16 đến 0.40
    low, high = wilson_interval(12, 46)
    assert 0.15 < low < 0.17 and 0.39 < high < 0.41


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(-1, 4)


def test_wilson_interval_contains_estimate_and_narrows_with_n():
    for k, n in [(1, 5), (30, 50), (99, 100)]:
        low, high = wilson_interval(k, n)
        assert low <= k / n <= high
    assert (lambda a, b: b - a)(*wilson_interval(50, 100)) < (lambda a, b: b - a)(*wilson_interval(5, 10))


def test_per_class_recall_wilson():
    labels = [0, 0, 0, 0, 1, 1, 2]
    predictions = [0, 0, 1, 0, 1, 0, 0]
    table = per_class_recall_wilson(labels, predictions, ["a", "b", "c", "d"])
    assert table["a"]["support"] == 4 and table["a"]["recall"] == pytest.approx(0.75)
    assert table["b"]["recall"] == pytest.approx(0.5) and table["c"]["recall"] == 0.0
    assert math.isnan(table["d"]["recall"]) and (table["d"]["low"], table["d"]["high"]) == (0.0, 1.0)
    assert table["a"]["low"] <= 0.75 <= table["a"]["high"]
    with pytest.raises(ValueError):
        per_class_recall_wilson([0, 1], [0], ["a", "b"])


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    data = np.random.RandomState(0).normal(5.0, 1.0, 200)
    low, high = bootstrap_ci(data, seed=1)
    assert low < data.mean() < high and high - low < 0.5
    assert bootstrap_ci(data, seed=1) == (low, high)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_spearman_with_ci_detects_perfect_and_weak_association():
    x = np.arange(15, dtype=float)
    perfect = spearman_with_ci(x, x**2, seed=0)
    assert perfect["rho"] == pytest.approx(1.0) and perfect["n"] == 15
    noise = np.random.RandomState(1).rand(15)
    weak = spearman_with_ci(x, noise, seed=0)
    # n nhỏ nên khoảng rộng: đúng điều cần báo khi có 14 đến 20 họ
    assert weak["high"] - weak["low"] > 0.5 and weak["low"] < weak["rho"] < weak["high"]
    with pytest.raises(ValueError):
        spearman_with_ci([1, 2], [1, 2])
    with pytest.raises(ValueError):
        spearman_with_ci([1, 2, 3], [1, 2])

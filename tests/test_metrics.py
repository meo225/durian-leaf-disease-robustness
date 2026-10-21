import pytest

from src.evaluation.metrics import compute_metrics

CLASSES = ["a", "b", "c"]


def test_perfect_prediction() -> None:
    result = compute_metrics([0, 1, 2, 2], [0, 1, 2, 2], CLASSES)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_known_values() -> None:
    result = compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], CLASSES)
    assert result["per_class_recall"] == pytest.approx({"a": 0.5, "b": 1.0, "c": 0.5})
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_class_never_predicted_keeps_full_shape() -> None:
    result = compute_metrics([0, 1, 2], [0, 1, 1], CLASSES)
    assert len(result["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in result["confusion_matrix"])
    assert result["per_class_recall"]["c"] == 0.0

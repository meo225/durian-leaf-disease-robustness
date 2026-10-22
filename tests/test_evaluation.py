import json

import pytest

from src.evaluation import plot_confusion_matrix, plot_history
from src.evaluation.metrics import classification_scores

CLASSES = ["a", "b", "c"]


def test_scores_known_values():
    scores = classification_scores([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], CLASSES)
    assert scores["per_class_recall"] == pytest.approx({"a": 0.5, "b": 1.0, "c": 0.5})
    assert scores["balanced_accuracy"] == pytest.approx(2 / 3)
    assert scores["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert scores["accuracy"] == pytest.approx(4 / 6)
    assert scores["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_class_never_predicted_keeps_full_shape():
    scores = classification_scores([0, 1, 2], [0, 1, 1], CLASSES)
    assert all(len(row) == 3 for row in scores["confusion_matrix"])
    assert scores["per_class_recall"]["c"] == 0.0


def test_plots_write_png(tmp_path):
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "val_macro_f1": 0.5},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6, "val_macro_f1": 0.8},
    ]
    plot_history(history, tmp_path / "h.png", best_epoch=2)
    plot_confusion_matrix([[1, 0], [1, 2]], ["A_B", "C"], tmp_path / "c.png")
    assert (tmp_path / "h.png").stat().st_size > 0
    assert (tmp_path / "c.png").stat().st_size > 0


def test_result_records_share_one_schema():
    for model in ("resnet50", "mobilenetv3_small"):
        path = f"reports/baselines/{model}/{model}_seed42.json"
        record = json.loads(open(path, encoding="utf-8").read())
        assert record["model_name"] == model
        assert {"library", "backbone", "elapsed_seconds", "trainable_parameters", "val_macro_f1", "history"} <= set(record)

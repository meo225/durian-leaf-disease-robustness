import numpy as np
import pytest
from PIL import Image

from src.training.hog_svm import build_classifier, build_record, hog_features

HOG_CFG = {"cell_size": 16, "block_cells": 2, "bins": 9}
SVM_CFG = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None}


def test_hog_dimension_for_224_image():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    # 13×13 khối, mỗi khối 2×2 ô × 9 hướng
    assert hog_features(image, HOG_CFG).shape == (13 * 13 * 4 * 9,)


def test_classifier_separates_two_easy_classes():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (20, 8)), rng.normal(5, 1, (20, 8))])
    y = np.array([0] * 20 + [1] * 20)
    assert (build_classifier(SVM_CFG, 42).fit(x, y).predict(x) == y).all()


def test_record_matches_baseline_keys_and_has_no_test_scores():
    scores = {
        "macro_f1": 0.4, "balanced_accuracy": 0.4, "accuracy": 0.4,
        "per_class_recall": {"a": 0.4}, "confusion_matrix": [[1]],
    }
    train_cfg = {
        "model_name": "hog_svm", "library": "x", "backbone": "y", "train_mode": "classical",
        "seed": 42, "selection_metric": "macro_f1", "hog": HOG_CFG, "svm": SVM_CFG,
    }
    record = build_record(scores, train_cfg, {"image_size": 224, "classes": ["a"]}, 6084, 1.0)
    assert record["val_macro_f1"] == 0.4 and record["resource"] == "cpu"
    assert not any(key.startswith("test_") for key in record)

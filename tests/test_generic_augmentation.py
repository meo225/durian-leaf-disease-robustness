import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torchvision import transforms

from src.evaluation.control_compare import (
    check_same_images,
    condition_comparison,
    family_scores,
    retrained_matrix,
    summarize_methods,
    verdict,
)
from src.evaluation.control_plots import plot_delta_heatmap, plot_family_bars, plot_levels, plot_objective, plot_scatter_methods
from src.selector.calibrate import macro_f1_rows
from src.training.run import ROOT, load_configs
from src.training.run_generic_aug import (
    build_generic_augmentation,
    build_train_steps,
    selection_config,
)

CONFIG = "configs/train_swin_tiny_trivialaugment.yaml"
NAMES = [f"{family}_s{level}" for family in ("blur", "jpeg") for level in (1, 2, 3)]
CLASSES = 5


def _matrices(images: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, size=images)

    def noisy(rate_by_row):
        rows = []
        for rate in rate_by_row:
            preds = labels.copy()
            wrong = rng.random(images) < rate
            preds[wrong] = (preds[wrong] + 1) % CLASSES
            rows.append(preds)
        return np.stack(rows)

    weak = noisy([0.05, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4])
    strong = noisy([0.05, 0.05, 0.08, 0.1, 0.05, 0.08, 0.1])
    return labels, {"weak": weak, "strong": strong}


def test_config_matches_the_degradation_control_except_for_the_augmentation():
    ta, _ = load_configs(CONFIG)
    aug, _ = load_configs("configs/train_swin_tiny_aug.yaml")
    shared = ("pipeline_config", "library", "backbone", "pretrained", "epochs", "patience", "min_delta", "learning_rate",
              "weight_decay", "optimizer", "scheduler", "selection_metric", "num_workers")
    assert {key: ta[key] for key in shared} == {key: aug[key] for key in shared}
    assert ta["validation"]["severity"] == aug["validation"]["severity"]
    assert ta["validation"]["corruption_config"] == aug["augmentation"]["corruption_config"]
    assert ta["model_name"] != aug["model_name"]


def test_train_pipeline_adds_trivialaugment_right_before_totensor_and_no_corruption():
    train_cfg, pipeline_cfg = load_configs(CONFIG)
    steps = build_train_steps(pipeline_cfg, build_generic_augmentation(train_cfg))
    kinds = [type(step).__name__ for step in steps]
    assert kinds.index("TrivialAugmentWide") + 1 == kinds.index("ToTensor")
    assert "RandomCorruption" not in kinds
    baseline = [type(step).__name__ for step in transforms.Compose(steps).transforms if type(step).__name__ != "TrivialAugmentWide"]
    assert baseline == ["ConditionalResize", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation", "ToTensor", "Normalize"]


def test_unknown_method_is_rejected():
    train_cfg, _ = load_configs(CONFIG)
    train_cfg["augmentation"]["method"] = "randaugment"
    with pytest.raises(ValueError):
        build_generic_augmentation(train_cfg)


def test_selection_rule_is_the_one_of_task_4_5():
    train_cfg, _ = load_configs(CONFIG)
    aug, _ = load_configs("configs/train_swin_tiny_aug.yaml")
    assert selection_config(train_cfg) == {
        "augmentation": {"corruption_config": aug["augmentation"]["corruption_config"]},
        "validation": {"severity": aug["validation"]["severity"]},
    }


def test_trivialaugment_changes_pil_images_and_keeps_size():
    from PIL import Image

    train_cfg, _ = load_configs(CONFIG)
    augmentation = build_generic_augmentation(train_cfg)
    image = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8))
    torch.manual_seed(0)
    outputs = [np.asarray(augmentation(image)) for _ in range(20)]
    assert all(output.shape == (64, 64, 3) for output in outputs)
    assert any(not np.array_equal(output, np.asarray(image)) for output in outputs)


def test_summary_is_paired_and_reports_levels():
    labels, matrices = _matrices()
    summary = summarize_methods(labels, matrices, NAMES, CLASSES, base="weak", resamples=50, seed=1).set_index("method")
    assert summary.loc["weak", "delta_vs_base"] == 0 and summary.loc["weak", "delta_low"] == 0 == summary.loc["weak", "delta_high"]
    assert summary.loc["strong", "delta_vs_base"] > 0 and summary.loc["strong", "delta_low"] > 0
    expected = macro_f1_rows(labels, matrices["strong"], CLASSES)[1:].mean()
    assert summary.loc["strong", "objective"] == pytest.approx(expected)
    assert {"level_1", "level_2", "level_3"} <= set(summary.columns)
    again = summarize_methods(labels, matrices, NAMES, CLASSES, base="weak", resamples=50, seed=1)
    pd.testing.assert_frame_equal(summary.reset_index(), again)


def test_family_scores_and_condition_comparison():
    labels, matrices = _matrices()
    scores = family_scores(labels, matrices, NAMES, CLASSES)
    assert list(scores.index) == ["blur", "jpeg"] and set(scores.columns) == {"weak", "strong"}
    level3 = family_scores(labels, matrices, NAMES, CLASSES, level=3)
    assert (level3["strong"] > level3["weak"]).all()
    comparison = condition_comparison(labels, matrices["weak"], matrices["strong"], NAMES, CLASSES)
    assert list(comparison["condition"]) == NAMES
    assert (comparison["p_holm"] >= comparison["p"] - 1e-12).all()
    assert (comparison["gains"] + comparison["harms"] > 0).all()
    assert comparison["delta"].gt(0).all()


def test_check_same_images_rejects_different_labels():
    labels = np.array([0, 1, 2])
    check_same_images({"a": labels, "b": labels.copy()})
    with pytest.raises(ValueError):
        check_same_images({"a": labels, "b": np.array([0, 1, 1])})


def test_retrained_matrix_reads_saved_keys_and_reports_missing(tmp_path):
    labels, matrices = _matrices(images=30)
    arrays = {f"retrained__m__{name}": matrices["weak"][index].astype(np.int8) for index, name in enumerate(["clean", *NAMES])}
    np.savez(tmp_path / "p.npz", labels=labels, **arrays)
    loaded_labels, matrix = retrained_matrix(tmp_path / "p.npz", "m", NAMES)
    np.testing.assert_array_equal(matrix, matrices["weak"])
    np.testing.assert_array_equal(loaded_labels, labels)
    with pytest.raises(KeyError):
        retrained_matrix(tmp_path / "p.npz", "other", NAMES)


def test_verdict_depends_on_the_interval_not_on_the_point_estimate():
    assert verdict(0.02, 0.01, 0.03) == "tốt hơn"
    assert verdict(-0.02, -0.03, -0.01) == "kém hơn"
    assert verdict(0.01, -0.01, 0.03) == "chưa phân biệt được"


def test_plots_write_files(tmp_path):
    labels, matrices = _matrices()
    summary = summarize_methods(labels, matrices, NAMES, CLASSES, base="weak", resamples=30)
    comparison = condition_comparison(labels, matrices["weak"], matrices["strong"], NAMES, CLASSES)
    plot_objective(summary, tmp_path / "a.png", "t")
    plot_levels(summary, tmp_path / "b.png", "t")
    plot_family_bars(family_scores(labels, matrices, NAMES, CLASSES), tmp_path / "c.png", "t", "F1")
    plot_delta_heatmap(comparison, tmp_path / "d.png", "t")
    plot_scatter_methods(comparison, tmp_path / "e.png", "x", "y", "t")
    for name in "abcde":
        assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_config_file_is_valid_yaml():
    assert yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))["augmentation"]["method"] == "trivialaugment_wide"

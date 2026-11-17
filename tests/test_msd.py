import math

import pytest
import torch
from torch import nn

from src.corruption import load_conditions
from src.corruption.augment import VALUE_BOUNDS, RandomCorruption, jitter_value
from src.evaluation.msd import (
    AugmentationPool,
    Featurizer,
    MSDResult,
    apply_trivial_augment,
    build_degradation_pool,
    build_trivial_augment_pool,
    classify_near_far,
    family_deltas,
    leave_one_out_msd,
    msd,
    near_far_threshold,
    penultimate_features,
    sample_degradation_augmentation,
    sample_trivial_augment,
    trivial_augment_ops,
)

MEAN, STD = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]


class TinyModel(nn.Module):
    """Đặc trưng = trung bình từng kênh. Có đủ forward_features và forward_head như model timm."""

    def forward_features(self, x):
        return x.mean(dim=(2, 3))

    def forward_head(self, x, pre_logits=False):
        return x


@pytest.fixture
def images():
    return torch.rand(6, 3, 64, 64, generator=torch.Generator().manual_seed(0)) * 0.8 + 0.1


@pytest.fixture
def featurizer():
    return Featurizer(TinyModel(), MEAN, STD, torch.device("cpu"), batch_size=4)


def pool_of(means, labels, k=4):
    deltas = torch.tensor(means, dtype=torch.float32)[:, None, :].repeat(1, k, 1)
    return AugmentationPool("test", deltas, list(labels))


def test_penultimate_features_requires_timm_style_model():
    assert penultimate_features(TinyModel(), torch.ones(2, 3, 4, 4)).shape == (2, 3)
    with pytest.raises(ValueError):
        penultimate_features(nn.Linear(2, 2), torch.ones(1, 2))


def test_deltas_are_transformed_minus_clean_features(featurizer, images):
    deltas = featurizer.deltas(images, lambda image: image * 0.5)
    expected = -0.5 * images.mean(dim=(2, 3))
    assert deltas.shape == (6, 3)
    assert torch.allclose(deltas, expected, atol=1e-6)
    assert torch.allclose(featurizer.deltas(images, lambda image: image), torch.zeros(6, 3))


def test_normalization_is_applied_inside_featurizer(images):
    featurizer = Featurizer(TinyModel(), [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], torch.device("cpu"))
    assert torch.allclose(featurizer.features(images), (images.mean(dim=(2, 3)) - 0.5) / 0.25, atol=1e-5)


def test_msd_is_min_euclidean_distance_and_names_nearest():
    pool = pool_of([[0, 0], [3, 4], [10, 0]], ["a", "b", "c"])
    family = torch.tensor([[3.0, 3.0]]).repeat(4, 1)
    result = msd(pool, family, n_bootstrap=0)
    assert result.value == pytest.approx(1.0)
    assert result.nearest == "b" and result.n_samples == 3
    assert math.isnan(result.ci_low) and math.isnan(result.ci_high)


def test_msd_leave_one_out_excludes_labels():
    pool = pool_of([[0, 0], [3, 4], [10, 0]], ["a", "b", "c"])
    family = torch.tensor([[3.0, 3.0]]).repeat(4, 1)
    result = msd(pool, family, exclude_labels=["b"], n_bootstrap=0)
    assert result.nearest == "a" and result.value == pytest.approx(math.sqrt(18))
    with pytest.raises(ValueError):
        msd(pool, family, exclude_labels=["a", "b", "c"], n_bootstrap=0)
    with pytest.raises(ValueError):
        msd(pool, torch.zeros(5, 2), n_bootstrap=0)


def test_bootstrap_interval_brackets_value_and_collapses_when_images_agree():
    pool = pool_of([[0, 0], [3, 4]], ["a", "b"])
    constant = msd(pool, torch.tensor([[3.0, 3.0]]).repeat(4, 1), n_bootstrap=50)
    assert constant.ci_low == pytest.approx(constant.value) and constant.ci_high == pytest.approx(constant.value)
    generator = torch.Generator().manual_seed(1)
    noisy_pool = AugmentationPool("noisy", torch.randn(5, 20, 4, generator=generator), list("abcde"))
    noisy_family = torch.randn(20, 4, generator=generator)
    a = msd(noisy_pool, noisy_family, n_bootstrap=100, seed=3)
    b = msd(noisy_pool, noisy_family, n_bootstrap=100, seed=3)
    assert a.ci_low < a.ci_high and a.ci_low <= a.value * 1.5 and a.ci_high >= a.value * 0.5
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_pool_validates_shapes():
    with pytest.raises(ValueError):
        AugmentationPool("bad", torch.zeros(3, 2), ["a", "b", "c"])
    with pytest.raises(ValueError):
        AugmentationPool("bad", torch.zeros(3, 2, 2), ["a", "b"])


def test_leave_one_out_and_threshold():
    pool = pool_of([[0, 0], [3, 4], [10, 0]], ["a", "b", "c"])
    deltas = {"a": torch.zeros(4, 2), "b": torch.tensor([[3.0, 4.0]]).repeat(4, 1), "c": torch.tensor([[10.0, 0.0]]).repeat(4, 1)}
    loo = leave_one_out_msd(pool, deltas, n_bootstrap=0)
    # bỏ chính nó thì mẫu gần nhất là họ khác: a->b cách 5, b->a cách 5, c->b cách sqrt(65)
    assert loo["a"].value == pytest.approx(5.0) and loo["b"].value == pytest.approx(5.0) and loo["c"].value == pytest.approx(math.sqrt(65))
    assert near_far_threshold(loo) == pytest.approx(math.sqrt(65))
    with pytest.raises(ValueError):
        near_far_threshold({})


def test_classify_near_far_flags_ambiguous_when_interval_contains_threshold():
    assert classify_near_far(MSDResult(1.0, "a", 0.9, 1.1, 10), 2.0) == "near"
    assert classify_near_far(MSDResult(3.0, "a", 2.9, 3.2, 10), 2.0) == "far"
    assert classify_near_far(MSDResult(1.9, "a", 1.7, 2.3, 10), 2.0) == "ambiguous"
    assert classify_near_far(MSDResult(3.0, "a", float("nan"), float("nan"), 10), 2.0) == "far"


def test_trivial_augment_samples_follow_forward_distribution():
    space = trivial_augment_ops()
    assert list(space) == ["Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate", "Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize", "AutoContrast", "Equalize"]
    samples = sample_trivial_augment(3000, seed=1)
    assert samples == sample_trivial_augment(3000, seed=1) and samples != sample_trivial_augment(3000, seed=2)
    assert {op for op, _ in samples} == set(space) - {"Identity"}
    counts = {op: sum(1 for o, _ in samples if o == op) for op in {o for o, _ in samples}}
    assert min(counts.values()) > 3000 / 13 * 0.7
    for op, magnitude in samples:
        grid = space[op][0].reshape(-1)
        assert abs(magnitude) <= float(grid.abs().max()) + 1e-6
        assert any(abs(abs(magnitude) - float(g)) < 1e-6 for g in grid)
        if not space[op][1]:
            assert magnitude >= 0
    assert any(m < 0 for op, m in samples if op == "Brightness") and any(m > 0 for op, m in samples if op == "Brightness")
    assert any(op == "Identity" for op, _ in sample_trivial_augment(500, 0, include_identity=True))


def test_apply_trivial_augment_matches_pil_ops_and_identity(images):
    image = images[0]
    assert torch.equal(apply_trivial_augment(image, "Identity", 0.0), torch.round(image * 255) / 255)
    # Brightness hệ số 1.5: nhân 1.5 rồi cắt về 1
    out = apply_trivial_augment(image, "Brightness", 0.5)
    assert torch.allclose(out, (torch.round(image * 255) * 1.5).clamp(0, 255).floor() / 255, atol=1.5 / 255)
    for op in trivial_augment_ops():
        assert apply_trivial_augment(image, op, 0.0 if op in ("Identity", "AutoContrast", "Equalize") else 0.3).shape == image.shape


def test_degradation_samples_use_training_jitter_and_bounds():
    conditions = load_conditions("configs/corruption.yaml", "val")
    samples = sample_degradation_augmentation(conditions, 400, seed=5, jitter=0.1)
    assert samples == sample_degradation_augmentation(conditions, 400, seed=5, jitter=0.1)
    assert {s["family"] for s in samples} == {spec["family"] for spec in conditions.values()}
    for sample in samples:
        spec = conditions[sample["source_condition"]]
        assert spec["family"] == sample["family"] and spec["type"] == sample["type"]
        assert abs(sample["value"] / spec["value"] - 1.0) <= 0.1 + 1e-9 or sample["type"] in VALUE_BOUNDS
        low, high = VALUE_BOUNDS.get(sample["type"], (None, None))
        assert (low is None or sample["value"] >= low) and (high is None or sample["value"] <= high)
    with pytest.raises(ValueError):
        sample_degradation_augmentation({}, 3, 0)


def test_jitter_value_is_what_random_corruption_uses():
    conditions = load_conditions("configs/corruption.yaml", "val")
    aug = RandomCorruption(conditions, probability=1.0, jitter=0.1)
    spec = {"type": "jpeg", "value": 50.0}
    torch.manual_seed(0)
    u = float(torch.rand(1))
    torch.manual_seed(0)
    assert aug._value(spec) == jitter_value(spec, 0.1, u)
    assert jitter_value({"type": "jpeg", "value": 100.0}, 0.1, 1.0) == 95.0
    assert jitter_value({"type": "blur", "value": 1.0}, 0.1, 0.0) == pytest.approx(0.9)


def test_pools_and_family_deltas_end_to_end_on_tiny_model(featurizer):
    images = torch.rand(4, 3, 64, 64, generator=torch.Generator().manual_seed(2)) * 0.8 + 0.1
    conditions = load_conditions("configs/corruption.yaml", "val")
    pool = build_degradation_pool(featurizer, images, conditions, n_samples=8, seed=3)
    assert pool.deltas.shape == (8, 4, 3) and len(pool.labels) == 8
    again = build_degradation_pool(featurizer, images, conditions, n_samples=8, seed=3)
    assert torch.equal(pool.deltas, again.deltas)
    ta = build_trivial_augment_pool(featurizer, images, n_samples=8, seed=3)
    assert ta.deltas.shape == (8, 4, 3) and "Identity" not in ta.labels
    family = family_deltas(featurizer, images, conditions, "brightness_dark", seed=3)
    assert family.shape == (4, 3)
    # brightness_dark làm tối: đặc trưng trung bình kênh giảm ở mọi ảnh
    assert (family < 0).all()
    with pytest.raises(KeyError):
        family_deltas(featurizer, images, conditions, "no_such_family", seed=3)
    result = msd(pool, family, exclude_labels=["brightness_dark"], n_bootstrap=20, seed=1)
    assert result.value >= 0 and result.ci_low <= result.ci_high


def test_pairwise_distances_are_symmetric_with_zero_diagonal():
    from src.evaluation.msd import pairwise_distances

    deltas = {"a": torch.zeros(3, 2), "b": torch.tensor([[3.0, 4.0]]).repeat(3, 1), "c": torch.tensor([[0.0, 1.0]]).repeat(3, 1)}
    matrix = pairwise_distances(deltas)
    assert matrix["a"]["a"] == 0 and matrix["a"]["b"] == pytest.approx(5.0) and matrix["b"]["a"] == matrix["a"]["b"]
    assert matrix["b"]["c"] == pytest.approx(math.sqrt(9 + 9))


def test_msd_plots_write_files(tmp_path):
    from src.evaluation.msd import pairwise_distances
    from src.evaluation.msd_plots import plot_distance_matrix, plot_msd_bars

    rows = [
        {"family": "x", "msd": 1.0, "ci_low": 0.8, "ci_high": 1.2, "group": "near"},
        {"family": "y", "msd": 2.0, "ci_low": float("nan"), "ci_high": float("nan"), "group": "far"},
    ]
    assert plot_msd_bars(rows[:1], 1.5, tmp_path / "bars.png", "t").stat().st_size > 0
    matrix = pairwise_distances({"a": torch.zeros(2, 2), "b": torch.ones(2, 2)})
    assert plot_distance_matrix(matrix, tmp_path / "matrix.png", "t").stat().st_size > 0


def test_select_images_is_seeded_and_refuses_the_test_split(tmp_path, monkeypatch):
    import pandas as pd
    from PIL import Image

    from src.evaluation.msd import select_images

    rows = []
    for index in range(9):
        name = f"img_{index}.png"
        Image.fromarray((torch.full((48, 48, 3), index * 20, dtype=torch.uint8)).numpy()).save(tmp_path / name)
        rows.append({"image_path": name, "class_name": "A" if index % 2 else "B"})
    monkeypatch.setattr("src.data.splits.get_split", lambda name: pd.DataFrame(rows))
    cfg = {"image_root": str(tmp_path), "image_size": 32, "classes": ["A", "B"]}
    first = select_images(cfg, "val", 4, seed=1)
    assert first.shape == (4, 3, 32, 32) and 0.0 <= first.min() and first.max() <= 1.0
    assert torch.equal(first, select_images(cfg, "val", 4, seed=1))
    assert not torch.equal(first, select_images(cfg, "val", 4, seed=2))
    with pytest.raises(ValueError):
        select_images(cfg, "val", 10, seed=1)
    with pytest.raises(PermissionError):
        select_images(cfg, "test", 4, seed=1)

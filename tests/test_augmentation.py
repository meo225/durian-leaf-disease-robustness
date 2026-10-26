from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

import src.corruption.benchmark as corruption_benchmark
import src.training.run_aug as run_aug
from src.corruption import load_conditions
from src.corruption.augment import VALUE_BOUNDS, RandomCorruption
from src.evaluation.condition_predictions import evaluate_with_predictions, save_predictions

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = load_conditions("configs/corruption.yaml", "train")
PIPELINE = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
TRAIN_CFG = yaml.safe_load((ROOT / "configs/train_convnext_tiny_aug.yaml").read_text(encoding="utf-8"))


def _image(seed: int = 0, size: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, size, size, generator=generator) * 0.8 + 0.1


def test_zero_probability_keeps_image_and_full_probability_changes_it():
    image = _image()
    assert torch.equal(RandomCorruption(CONDITIONS, probability=0.0)(image), image)
    for name, spec in CONDITIONS.items():
        out = RandomCorruption({name: spec}, probability=1.0, jitter=0.1)(image)
        assert out.shape == image.shape
        assert out.min() >= 0 and out.max() <= 1, name
        assert torch.allclose(out * 255, torch.round(out * 255), atol=1e-4), name
        assert not torch.equal(out, image), name


def test_randomness_follows_global_seed_and_differs_between_calls():
    noise = {name: spec for name, spec in CONDITIONS.items() if spec["family"] == "noise"}
    augment = RandomCorruption(noise, probability=1.0)
    image = _image()
    torch.manual_seed(1)
    first, second = augment(image), augment(image)
    assert not torch.equal(first, second)
    torch.manual_seed(1)
    assert torch.equal(augment(image), first)


def test_exclude_families_and_empty_set():
    augment = RandomCorruption(CONDITIONS, exclude_families=["blur", "noise"])
    assert "blur" not in augment.families() and "noise" not in augment.families()
    assert len(augment.families()) == 12
    assert len(RandomCorruption(CONDITIONS).families()) == 14
    with pytest.raises(ValueError):
        RandomCorruption(CONDITIONS, exclude_families=list({spec["family"] for spec in CONDITIONS.values()}))
    with pytest.raises(ValueError):
        RandomCorruption(CONDITIONS, probability=1.5)


def test_jitter_keeps_values_valid():
    for name, spec in CONDITIONS.items():
        augment = RandomCorruption({name: spec}, jitter=0.5)
        low, high = VALUE_BOUNDS.get(spec["type"], (None, None))
        for _ in range(40):
            value = augment._value(spec)
            assert value > 0
            assert low is None or value >= low
            assert high is None or value <= high


def test_validation_names_cover_every_family_at_fixed_severity():
    names = run_aug.validation_condition_names(TRAIN_CFG)
    assert len(names) == 14 and all(name.endswith("_s2") for name in names)
    assert set(names) <= set(CONDITIONS)


def test_mixed_validation_has_clean_plus_each_family(monkeypatch):
    original = run_aug.get_split
    monkeypatch.setattr(run_aug, "get_split", lambda name: original(name).head(4))
    dataset = run_aug.build_mixed_validation(TRAIN_CFG, PIPELINE)
    assert len(dataset) == 4 * 15
    image, label = dataset[0]
    assert image.shape == (3, 224, 224) and image.dtype == torch.float32
    assert dataset.images.dtype == torch.uint8
    assert [dataset[index][1] for index in range(4)] == [dataset[index + 4][1] for index in range(4)]
    assert not torch.equal(dataset.images[0:4], dataset.images[4:8])


def test_train_loader_applies_augmentation_after_totensor():
    augmentation = run_aug.build_augmentation(TRAIN_CFG)
    assert len(augmentation.specs) == 42 and augmentation.probability == 0.5
    loader = run_aug.build_train_loader({**TRAIN_CFG, "num_workers": 0}, PIPELINE, augmentation)
    steps = [type(step).__name__ for step in loader.dataset.transform.transforms]
    assert steps.index("RandomCorruption") == steps.index("ToTensor") + 1
    assert steps.index("RandomCorruption") < steps.index("Normalize")
    images, labels, _ = next(iter(loader))
    assert images.shape == (PIPELINE["batch_size"], 3, 224, 224)


def test_evaluate_with_predictions_keeps_keys_and_saves_probabilities(monkeypatch, tmp_path):
    original = corruption_benchmark.get_split
    monkeypatch.setattr(corruption_benchmark, "get_split", lambda name: original(name).head(8))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 5, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))
    conditions = {name: CONDITIONS[name] for name in ["noise_s1", "blur_s3"]}
    result = evaluate_with_predictions(model, PIPELINE, "val", conditions, torch.device("cpu"), 42, "retrained", "demo", progress=lambda _: None)
    frame = result["frame"]
    assert list(frame["condition"]) == ["clean", "noise_s1", "blur_s3"]
    for column in ["role", "technique", "config_id", "condition", "family", "severity", "macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "degradation", "recall_HEALTHY_LEAF"]:
        assert column in frame.columns
    assert frame.loc[0, "degradation"] == 0
    probabilities = result["predictions"]["retrained__demo__noise_s1__probs"]
    assert probabilities.shape == (8, 5) and np.allclose(probabilities.astype(np.float32).sum(axis=1), 1.0, atol=1e-2)
    assert result["predictions"]["retrained__demo__blur_s3"].dtype == np.int8
    path = tmp_path / "p.npz"
    save_predictions(result, path)
    with np.load(path) as saved:
        assert saved["labels"].shape == (8,) and len(saved["paths"]) == 8
        assert "retrained__demo__clean__probs" in saved.files

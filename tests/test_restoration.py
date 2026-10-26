import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.processing_benchmark import PROCESSING_MODULES, apply_processing as dispatch, load_grid, predict
from src.processing.restoration import TECHNIQUES, apply_processing, bilateral, gaussian, median, nlm, unsharp

CONFIG = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}


def _flat(value: float = 0.5, size: int = 64) -> torch.Tensor:
    return torch.full((3, size, size), value)


def _noisy(seed: int = 0, std: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    clean = _flat()
    generator = torch.Generator().manual_seed(seed)
    noisy = torch.round((clean + torch.randn(clean.shape, generator=generator) * std).clamp(0, 1) * 255) / 255
    return clean, noisy


def _edge(size: int = 64) -> torch.Tensor:
    image = torch.zeros(3, size, size)
    image[:, :, size // 2 :] = 1.0
    return image


def _edge_steepness(image: torch.Tensor) -> float:
    """Độ dốc lớn nhất giữa hai pixel kề nhau. Cạnh càng sắc thì càng lớn."""
    return float((image[:, :, 1:] - image[:, :, :-1]).abs().max())


PARAMS = [
    ("gaussian", {"sigma": 1.0}),
    ("median", {"ksize": 3}),
    ("bilateral", {"d": 5, "sigma_color": 30, "sigma_space": 3}),
    ("nlm", {"h": 10}),
    ("unsharp", {"sigma": 1.0, "amount": 1.0}),
]


def test_output_is_valid_eight_bit_image():
    _, image = _noisy()
    for name, params in PARAMS:
        out = apply_processing(image, {"technique": name, "params": params})
        assert out.shape == image.shape and out.dtype == torch.float32
        assert out.min() >= 0 and out.max() <= 1
        assert torch.allclose(out * 255, torch.round(out * 255), atol=1e-4)


def test_registry_covers_config_techniques():
    grid = load_grid("configs/restoration.yaml")
    assert set(grid["techniques"]) == set(TECHNIQUES)
    assert "jpeg" in grid["scope_families"]
    assert not any("jpeg" in spec["families"] for spec in grid["techniques"].values())
    for spec in grid["techniques"].values():
        assert set(spec["families"]) <= set(grid["scope_families"])
    with pytest.raises(ValueError):
        apply_processing(_flat(), {"technique": "khong-co"})


def test_every_grid_config_runs():
    _, image = _noisy()
    for technique, spec in load_grid("configs/restoration.yaml")["techniques"].items():
        for params in spec["grid"].values():
            assert apply_processing(image, {"technique": technique, "params": params}).shape == image.shape


def test_denoisers_reduce_error_on_flat_noisy_image():
    clean, noisy = _noisy()
    baseline = float(((noisy - clean) ** 2).mean())
    for out in (gaussian(noisy, 1.0), median(noisy, 3), bilateral(noisy, 5, 60, 3), nlm(noisy, 15)):
        assert float(((out - clean) ** 2).mean()) < baseline


def test_bilateral_keeps_edge_better_than_gaussian():
    edge = _edge()
    assert _edge_steepness(bilateral(edge, 5, 30, 3)) > _edge_steepness(gaussian(edge, 1.5))


def test_unsharp_sharpens_blurred_edge_and_zero_amount_is_identity():
    blurred = gaussian(_edge(), 2.0)
    assert _edge_steepness(unsharp(blurred, 1.0, 2.0)) > _edge_steepness(blurred)
    assert torch.equal(unsharp(blurred, 1.0, 0.0), blurred)


def test_dispatch_finds_techniques_in_both_modules():
    image = _flat(0.3)
    assert len(PROCESSING_MODULES) == 2
    assert torch.equal(dispatch(image, {"technique": "gamma", "params": {"gamma": 0.6}}), dispatch(image, {"technique": "gamma", "params": {"gamma": 0.6}}))
    assert dispatch(image, {"technique": "median", "params": {"ksize": 3}}).shape == image.shape
    with pytest.raises(ValueError):
        dispatch(image, {"technique": "khong-co"})


def test_predict_with_threads_matches_serial():
    generator = torch.Generator().manual_seed(0)
    images = (torch.rand(10, 3, 32, 32, generator=generator) * 255).to(torch.uint8)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 4))
    model.eval()
    torch.manual_seed(0)
    nn.init.normal_(model[1].weight)
    processor = lambda image: dispatch(image, {"technique": "nlm", "params": {"h": 10}})  # noqa: E731
    serial = predict(model, images, processor, CONFIG, torch.device("cpu"), batch_size=4, workers=1)
    threaded = predict(model, images, processor, CONFIG, torch.device("cpu"), batch_size=4, workers=4)
    assert np.array_equal(serial, threaded)

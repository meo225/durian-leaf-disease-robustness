import pandas as pd
import pytest
import torch

from src.evaluation.processing_benchmark import (
    best_per_condition,
    build_matrix,
    clean_cost_table,
    load_grid,
    select_configs,
)
from src.processing.illumination import (
    TECHNIQUES,
    apply_processing,
    clahe,
    gamma,
    gamma_auto,
    gray_world,
    mean_luminance,
    retinex,
)


def _image(seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(3, 64, 64, generator=generator) * scale).clamp(0, 1)


def test_output_is_valid_eight_bit_image():
    image = _image()
    for name, params in [
        ("gamma", {"gamma": 0.6}),
        ("gamma_auto", {"target": 0.5}),
        ("clahe", {"clip": 2.0, "tiles": 4}),
        ("retinex", {"sigmas": [5, 20], "alpha": 1.0}),
        ("gray_world", {"strength": 1.0}),
    ]:
        out = apply_processing(image, {"technique": name, "params": params})
        assert out.shape == image.shape
        assert out.min() >= 0 and out.max() <= 1
        assert torch.allclose(out * 255, torch.round(out * 255), atol=1e-4)


def test_registry_covers_config_techniques():
    grid = load_grid("configs/illumination.yaml")
    assert set(grid["techniques"]) == set(TECHNIQUES)
    with pytest.raises(ValueError):
        apply_processing(_image(), {"technique": "khong-co"})


def test_gamma_identity_and_direction():
    image = _image(scale=0.3)
    assert torch.allclose(gamma(image, 1.0), torch.round(image * 255) / 255, atol=1e-6)
    assert mean_luminance(gamma(image, 0.5)) > mean_luminance(image)
    assert mean_luminance(gamma(image, 2.0)) < mean_luminance(image)


def test_gamma_auto_moves_dark_and_bright_toward_target():
    dark, bright = _image(scale=0.25), 1 - _image(seed=1, scale=0.25)
    for image in (dark, bright):
        before = abs(mean_luminance(image) - 0.5)
        after = abs(mean_luminance(gamma_auto(image, 0.5)) - 0.5)
        assert after < before


def test_clahe_raises_contrast_of_flat_image():
    flat = 0.5 + (_image(seed=2) - 0.5) * 0.1
    assert clahe(flat, 4.0, 4).std() > flat.std()


def test_gray_world_equalizes_channel_means():
    cast = _image(seed=3) * torch.tensor([1.0, 0.7, 0.4]).view(3, 1, 1)
    means = gray_world(cast, 1.0).mean(dim=(1, 2))
    assert (means.max() - means.min()) < 0.02
    half = gray_world(cast, 0.5).mean(dim=(1, 2))
    assert (half.max() - half.min()) > (means.max() - means.min())


def test_retinex_handles_uniform_image():
    out = retinex(torch.full((3, 32, 32), 0.4), [5, 20], 1.0)
    assert torch.isfinite(out).all()


def _frame() -> pd.DataFrame:
    rows = []

    def add(role, technique, config_id, condition, f1):
        family, severity = ("clean", 0) if condition == "clean" else (condition.rsplit("_s", 1)[0], int(condition.rsplit("_s", 1)[1]))
        rows.append({"role": role, "technique": technique, "config_id": config_id, "condition": condition, "family": family, "severity": severity, "macro_f1": f1, "balanced_accuracy": f1, "accuracy": f1, "recall_A": f1})

    add("raw", "none", "none", "clean", 0.95)
    for family, raw in (("dark", 0.60), ("bright", 0.70)):
        for severity in (1, 2, 3):
            add("raw", "none", "none", f"{family}_s{severity}", raw)
    for severity in (1, 2, 3):
        add("tune", "gamma", "g_low", f"dark_s{severity}", 0.90)
        add("tune", "gamma", "g_high", f"dark_s{severity}", 0.62)
        add("tune", "gamma", "g_low", f"bright_s{severity}", 0.65)
        add("tune", "gamma", "g_high", f"bright_s{severity}", 0.85)
    add("clean_cost", "gamma", "g_low", "clean", 0.94)
    add("clean_cost", "gamma", "g_high", "clean", 0.90)
    return pd.DataFrame(rows)


GRID = {
    "selection": {"tie_margin": 0.003, "min_degradation": 0.02},
    "techniques": {"gamma": {"families": ["dark", "bright"], "grid": {"g_low": {"gamma": 0.5}, "g_high": {"gamma": 2.0}}}},
}


def test_select_configs_per_family_and_general():
    selected = select_configs(_frame(), GRID)
    assert selected["gamma"]["per_family"]["dark"]["config_id"] == "g_low"
    assert selected["gamma"]["per_family"]["bright"]["config_id"] == "g_high"
    assert selected["gamma"]["general"]["config_id"] == "g_low"
    assert selected["gamma"]["general"]["clean_macro_f1"] == pytest.approx(0.94)


def test_tie_prefers_higher_clean_f1():
    frame = _frame()
    frame.loc[(frame["config_id"] == "g_high") & (frame["family"] == "dark"), "macro_f1"] = 0.899
    frame.loc[(frame["config_id"] == "g_high") & (frame["technique"] == "gamma") & (frame["role"] == "clean_cost"), "macro_f1"] = 0.99
    selected = select_configs(frame, GRID)
    assert selected["gamma"]["per_family"]["dark"]["config_id"] == "g_high"


def test_matrix_recovery_and_clean_cost():
    frame = _frame()
    selected = select_configs(frame, GRID)
    matrix = build_matrix(frame, selected, 0.02)
    dark = matrix[matrix["condition"] == "dark_s1"].iloc[0]
    assert dark["degradation"] == pytest.approx(0.35)
    assert dark["recovery"] == pytest.approx(0.30)
    assert dark["normalized_recovery"] == pytest.approx(0.30 / 0.35)
    assert bool(dark["suitable"])
    best = best_per_condition(matrix)
    assert best.loc["bright_s2", "technique"] == "gamma"
    cost = clean_cost_table(frame, selected)
    assert cost.iloc[0]["change_macro_f1"] == pytest.approx(-0.01)
    assert cost.iloc[0]["d_recall_A"] == pytest.approx(-0.01)

import pytest
import torch
import yaml

from src.corruption import CORRUPTIONS, UNSEEN_CORRUPTIONS, apply_unseen, corrupt_tensor, load_conditions
from src.corruption.augment import RandomCorruption
from src.corruption.unseen import LUMA_WEIGHTS, UNSEEN_RANDOM_TYPES, chroma_noise, defocus_blur, fog, rain_streak, saturate, spatter

CONFIG = "configs/unseen_corruption.yaml"
VALUES = {"defocus_blur": 3.0, "chroma_noise": 0.08, "fog": 0.30, "spatter": 0.25, "rain_streak": 0.005, "saturate": 3.0}


@pytest.fixture
def image():
    return torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(0))


def test_registry_matches_config_and_is_disjoint_from_seen_families():
    families = yaml.safe_load(open(CONFIG, encoding="utf-8"))["families"]
    assert {spec["type"] for spec in families.values()} == set(UNSEEN_CORRUPTIONS) == set(VALUES)
    assert not set(UNSEEN_CORRUPTIONS) & set(CORRUPTIONS)
    seen = yaml.safe_load(open("configs/corruption.yaml", encoding="utf-8"))["families"]
    assert not set(families) & set(seen)


def test_every_family_keeps_shape_and_range(image):
    for kind, value in VALUES.items():
        out = apply_unseen(image, {"type": kind, "value": value}, torch.Generator().manual_seed(1))
        assert out.shape == image.shape
        assert 0.0 <= out.min() and out.max() <= 1.0
        assert not torch.equal(out, image)


def test_generator_makes_random_families_repeatable(image):
    for kind in UNSEEN_RANDOM_TYPES:
        spec = {"type": kind, "value": VALUES[kind]}
        first = apply_unseen(image, spec, torch.Generator().manual_seed(3))
        again = apply_unseen(image, spec, torch.Generator().manual_seed(3))
        other = apply_unseen(image, spec, torch.Generator().manual_seed(4))
        assert torch.equal(first, again)
        assert not torch.equal(first, other)


def test_defocus_blur_preserves_mean_and_smooths(image):
    out = defocus_blur(image, 3.0)
    assert out.mean() == pytest.approx(image.mean(), abs=0.01)
    assert out.std() < image.std()


def test_chroma_noise_leaves_luma_untouched():
    image = 0.4 + 0.2 * torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(0))  # xa 0 và 1 nên không bị cắt
    out = chroma_noise(image, 0.04, torch.Generator().manual_seed(0))
    weights = torch.tensor(LUMA_WEIGHTS).view(3, 1, 1)
    luma = lambda x: (x * weights).sum(dim=0)
    assert torch.allclose(luma(out), luma(image), atol=1e-5)
    chroma = lambda x: (x[0] - luma(x)).std()
    assert (out - image).abs().mean() > 0.01
    assert chroma(out) > chroma(image)


def test_fog_follows_scattering_model(image):
    out = fog(image, 0.30, torch.Generator().manual_seed(0))
    assert out.std() < image.std()  # mất tương phản
    assert out.mean() > image.mean()  # kéo về ánh sáng khí quyển
    denser = fog(image, 0.85, torch.Generator().manual_seed(0))
    assert denser.std() < out.std()


def test_spatter_covers_roughly_the_requested_area():
    flat = torch.full((3, 128, 128), 0.5)
    out = spatter(flat, 0.25, torch.Generator().manual_seed(0))
    covered = float(((out - flat).abs().mean(dim=0) > 0.02).float().mean())  # giọt làm sáng nhẹ vùng phẳng
    assert 0.1 < covered < 0.4


def test_rain_streak_adds_thin_bright_streaks_and_scales_with_density():
    flat = torch.full((3, 128, 128), 0.3)
    light = rain_streak(flat, 0.004, torch.Generator().manual_seed(0))
    heavy = rain_streak(flat, 0.02, torch.Generator().manual_seed(0))
    assert light.min() >= 0.3 - 1e-6  # chỉ làm sáng, không làm tối
    covered = lambda x: float(((x - flat).abs().mean(dim=0) > 0.02).float().mean())
    assert 0.01 < covered(light) < covered(heavy) < 0.7


def test_saturate_identity_gray_and_boost(image):
    assert torch.allclose(saturate(image, 1.0), image, atol=1e-6)
    assert saturate(image, 0.0).std(dim=0).max() < 1e-5  # mọi kênh bằng nhau
    chroma = lambda x: (x.max(dim=0).values - x.min(dim=0).values).mean()
    assert chroma(saturate(image, 2.0)) > chroma(image)


def test_corrupt_tensor_is_deterministic_and_8bit(image):
    spec = {"type": "fog", "value": 0.30}
    first = corrupt_tensor(image, "fog_s2", spec, 42)
    assert torch.equal(first, corrupt_tensor(image, "fog_s2", spec, 42))
    assert not torch.equal(first, corrupt_tensor(image, "fog_s2", spec, 43))
    assert torch.allclose(first * 255, torch.round(first * 255))


def test_unknown_type_is_rejected(image):
    with pytest.raises(ValueError):
        apply_unseen(image, {"type": "blur", "value": 1.0})


def test_augmentation_cannot_use_unseen_families(image):
    conditions = load_conditions(CONFIG, "val")
    augment = RandomCorruption(conditions, probability=1.0)
    with pytest.raises(ValueError):
        augment(image)


def test_unlocked_config_refuses_test_split():
    with pytest.raises(PermissionError):
        load_conditions(CONFIG, "test")
    assert len(load_conditions(CONFIG, "val")) == 18

import hashlib
import json

import numpy as np
import pytest
import torch
import yaml

from src.corruption import corrupt_tensor, load_conditions
from src.corruption.external import (
    EXTERNAL_TYPE,
    IMAGENET_C_BAR_GRID,
    IMAGENET_C_CORRUPTIONS,
    SOURCE_IMAGENET_C,
    SOURCE_IMAGENET_C_BAR,
    apply_external,
    apply_external_array,
    candidate_values,
    corruptions_of,
    family_name,
    imagenet_c_bar_reference_levels,
    probe_families,
    split_family_name,
)
from src.corruption.third_party.imagenet_c_bar.corrupt import IMAGENET_C_BAR_FAMILIES

# Mã băm đầu ra của mã gốc facebookresearch/augmentation-corruption commit ba4d5a5 (ảnh ngẫu nhiên RandomState(123) 64×64, np.random.seed(7), mức 5.0).
# Được tính từ chính repo gốc trước khi đưa vào, nên bản vendored khớp từng byte nghĩa là các sửa đổi trong NOTICE.md không đổi kết quả.
UPSTREAM_GOLDEN = {
    "single_frequency_greyscale": "8d6905b4155727bbfcc32ea01204518ac6e2295c9b9ebb31ba5eddc1497e8e37",
    "cocentric_sine_waves": "cb4865ec0b0615be208fe85d74b47ff2c07f1e407a2365f62263089f1dbc4c8b",
    "plasma_noise": "729e92c6be265c5c5c690ab9e1df267e867ca76eed0d2629ff1e9bb7bab03da2",
    "perlin_noise": "e557ec33ed23b69e600f3a2090e542f4860f6ab2d52ef25337c811fa8fdafaa3",
    "blue_noise_sample": "10188c570a1f33ee8e000795e0c24cedb5300749358fa26cb48787b38bfeaa5b",
    "brownish_noise": "9bb0dafa01657db5c4e11a45d83444d3f32e187c124d001dc50a6bc582c8875a",
    "checkerboard_cutout": "06eea3346bd459b1c3fead28c01bb0f3118ec7fcc6bb7f7b54955c33ee646bed",
    "sparkles": "d31a69e4861f0f44019068b33f74d3b8597b30d414a1d3b1a5884b909da31671",
    "inverse_sparkles": "f0963e00365b24f9d5839331136d7325c571e7363b1bf4daa9d2a19fe2eca85c",
    "caustic_refraction": "66f371686498dad47c335fe722e3f5781d570c2f4016534df382868a10fc6bab",
}
FAST_BAR = ["single_frequency_greyscale", "cocentric_sine_waves", "blue_noise_sample", "checkerboard_cutout"]


@pytest.fixture
def image():
    return torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(0))


def has_imagecorruptions() -> bool:
    try:
        from src.corruption.external import _imagenet_c_module

        _imagenet_c_module()
        return True
    except ImportError:
        return False


needs_imagecorruptions = pytest.mark.skipif(not has_imagecorruptions(), reason="chưa cài imagecorruptions")


def test_vendored_imagenet_c_bar_matches_upstream_byte_for_byte():
    image = (np.random.RandomState(123).rand(64, 64, 3) * 255).astype(np.uint8)
    assert set(IMAGENET_C_BAR_FAMILIES) == set(UPSTREAM_GOLDEN)
    for name, family in IMAGENET_C_BAR_FAMILIES.items():
        np.random.seed(7)
        assert hashlib.sha256(family(5.0, 64)(image).tobytes()).hexdigest() == UPSTREAM_GOLDEN[name], name


def test_vendored_families_are_exactly_the_imagenet_c_bar_tagged_ones():
    reference = imagenet_c_bar_reference_levels()
    assert set(reference) == set(IMAGENET_C_BAR_FAMILIES)
    assert all(len(levels) == 5 and all(0 < v <= 10 for v in levels) for levels in reference.values())
    # mọi mức tham chiếu của bản gốc nằm trong khoảng lưới ứng viên (0, 10]
    assert max(IMAGENET_C_BAR_GRID) == 10.0 and min(IMAGENET_C_BAR_GRID) > 0


def test_names_and_candidates():
    assert family_name(SOURCE_IMAGENET_C, "gaussian_noise") == "inc_gaussian_noise"
    assert split_family_name("cbar_sparkles") == (SOURCE_IMAGENET_C_BAR, "sparkles")
    with pytest.raises(ValueError):
        split_family_name("fog")
    assert candidate_values(SOURCE_IMAGENET_C) == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert candidate_values(SOURCE_IMAGENET_C_BAR) == IMAGENET_C_BAR_GRID
    assert corruptions_of(SOURCE_IMAGENET_C) == IMAGENET_C_CORRUPTIONS


@pytest.mark.parametrize("name", FAST_BAR)
def test_bar_families_keep_shape_and_are_repeatable(name):
    image = (np.random.RandomState(1).rand(48, 48, 3) * 255).astype(np.uint8)
    first = apply_external_array(image, SOURCE_IMAGENET_C_BAR, name, 6.0, seed=11)
    again = apply_external_array(image, SOURCE_IMAGENET_C_BAR, name, 6.0, seed=11)
    other = apply_external_array(image, SOURCE_IMAGENET_C_BAR, name, 6.0, seed=12)
    assert first.dtype == np.uint8 and first.shape == image.shape
    assert np.array_equal(first, again)
    assert not np.array_equal(first, image)
    # họ có tham số ngẫu nhiên đổi theo seed (trừ khi toàn bộ tham số cố định)
    assert not np.array_equal(first, other)


def test_seeding_does_not_touch_global_numpy_state():
    image = (np.random.RandomState(1).rand(32, 32, 3) * 255).astype(np.uint8)
    np.random.seed(99)
    expected = np.random.rand(3)
    np.random.seed(99)
    apply_external_array(image, SOURCE_IMAGENET_C_BAR, "single_frequency_greyscale", 5.0, seed=5)
    assert np.array_equal(np.random.rand(3), expected)


@needs_imagecorruptions
def test_imagenet_c_families_repeatable_and_severity_is_integer(image):
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    for corruption in ("gaussian_noise", "brightness", "jpeg_compression"):
        a = apply_external_array(array, SOURCE_IMAGENET_C, corruption, 3.0, seed=3)
        b = apply_external_array(array, SOURCE_IMAGENET_C, corruption, 3.0, seed=3)
        assert np.array_equal(a, b) and a.shape == array.shape
    with pytest.raises(ValueError):
        apply_external_array(array, SOURCE_IMAGENET_C, "gaussian_noise", 2.5, seed=0)
    with pytest.raises(ValueError):
        apply_external_array(array, SOURCE_IMAGENET_C, "gaussian_noise", 6.0, seed=0)


def test_invalid_inputs_are_rejected():
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_external_array(array.astype(np.float32), SOURCE_IMAGENET_C_BAR, "sparkles", 5.0, 0)
    with pytest.raises(ValueError):
        apply_external_array(array, SOURCE_IMAGENET_C_BAR, "not_a_family", 5.0, 0)
    with pytest.raises(ValueError):
        apply_external_array(np.zeros((32, 48, 3), dtype=np.uint8), SOURCE_IMAGENET_C_BAR, "single_frequency_greyscale", 5.0, 0)
    with pytest.raises(ValueError):
        apply_external_array(array, SOURCE_IMAGENET_C_BAR, "single_frequency_greyscale", 11.0, 0)


def test_corrupt_tensor_routes_external_and_depends_on_condition_name(image):
    spec = {"type": EXTERNAL_TYPE, "source": SOURCE_IMAGENET_C_BAR, "corruption": "blue_noise_sample", "value": 5.0}
    first = corrupt_tensor(image, "cbar_blue_noise_sample_s1", spec, 42)
    again = corrupt_tensor(image, "cbar_blue_noise_sample_s1", spec, 42)
    other_name = corrupt_tensor(image, "cbar_blue_noise_sample_s2", spec, 42)
    assert first.shape == image.shape and 0.0 <= first.min() and first.max() <= 1.0
    assert torch.equal(first, again)
    assert not torch.equal(first, other_name)
    assert torch.equal(first, torch.round(first * 255) / 255)
    assert torch.equal(apply_external(image, spec, 1), apply_external(image, spec, 1))


def test_probe_reports_every_family_with_status_and_time():
    status = probe_families(size=32, repeats=1)
    expected = {family_name(SOURCE_IMAGENET_C_BAR, c) for c in IMAGENET_C_BAR_FAMILIES} | {family_name(SOURCE_IMAGENET_C, c) for c in IMAGENET_C_CORRUPTIONS}
    assert set(status) == expected
    assert all(set(entry) == {"ok", "error", "seconds_per_image"} for entry in status.values())
    assert status["cbar_single_frequency_greyscale"]["ok"]
    assert all(entry["error"] for entry in status.values() if not entry["ok"])
    json.dumps(status, default=float)


def test_load_conditions_carries_external_fields(tmp_path):
    cfg = {
        "protocol": "other",
        "locked": False,
        "seed": 1,
        "families": {
            "cbar_sparkles": {"type": EXTERNAL_TYPE, "source": SOURCE_IMAGENET_C_BAR, "corruption": "sparkles", "group": "near", "role": "val", "levels": [1.0, 2.0, 3.0]},
            "cbar_perlin_noise": {"type": EXTERNAL_TYPE, "source": SOURCE_IMAGENET_C_BAR, "corruption": "perlin_noise", "group": "far", "role": "train", "levels": [2.0, 4.0, 6.0]},
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    conditions = load_conditions(path, "val")
    assert list(conditions) == [f"cbar_sparkles_s{k}" for k in (1, 2, 3)] + [f"cbar_perlin_noise_s{k}" for k in (1, 2, 3)]
    assert conditions["cbar_sparkles_s2"] == {
        "type": EXTERNAL_TYPE, "value": 2.0, "family": "cbar_sparkles", "severity": 2, "group": "near", "source": SOURCE_IMAGENET_C_BAR, "corruption": "sparkles", "role": "val",
    }
    assert set(load_conditions(path, "val", roles=["train"])) == {f"cbar_perlin_noise_s{k}" for k in (1, 2, 3)}
    with pytest.raises(PermissionError):
        load_conditions(path, "test")

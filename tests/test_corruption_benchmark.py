import json

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from src.corruption import benchmark
from src.corruption.benchmark import build_loader, corrupt_tensor, example_rows, load_conditions
from src.evaluation import robustness
from src.evaluation.plots import plot_corruption_examples
from src.models.build import build_model

CLASSES = ["ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK", "HEALTHY_LEAF", "LEAF_BLIGHT", "PHOMOPSIS_LEAF_SPOT"]


def _write_config(path, locked):
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "locked": locked,
                "seed": 7,
                "families": {"brightness_dark": {"type": "brightness", "levels": [0.7, 0.3]}, "noise": {"type": "noise", "levels": [0.05, 0.1]}},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def val_data(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {"image_path": [f"img_{i}.jpg" for i in range(6)], "class_name": [CLASSES[i % 5] for i in range(6)], "split": "val"}
    )
    for name in frame["image_path"]:
        Image.fromarray((torch.rand(224, 224, 3) * 255).byte().numpy()).save(tmp_path / name)
    monkeypatch.setattr(benchmark, "get_split", lambda _name: frame)
    pipeline = yaml.safe_load((benchmark.ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    pipeline.update(image_root=str(tmp_path), batch_size=3)
    return pipeline


def test_conditions_are_named_by_family_and_severity(tmp_path):
    conditions = load_conditions(_write_config(tmp_path / "c.yaml", locked=False), "val")
    assert list(conditions) == ["brightness_dark_s1", "brightness_dark_s2", "noise_s1", "noise_s2"]
    assert conditions["noise_s2"] == {"type": "noise", "value": 0.1, "family": "noise", "severity": 2}


def test_test_split_needs_locked_config(tmp_path):
    with pytest.raises(PermissionError):
        load_conditions(_write_config(tmp_path / "c.yaml", locked=False), "test")
    assert load_conditions(_write_config(tmp_path / "c.yaml", locked=True), "test")


def test_project_config_is_valid_and_not_yet_locked_for_test():
    conditions = load_conditions("configs/corruption.yaml", "val")
    assert len(conditions) == 15
    assert {spec["severity"] for spec in conditions.values()} == {1, 2, 3}


def test_noise_is_repeatable_per_image_and_differs_between_images():
    spec = {"type": "noise", "value": 0.1}
    image = torch.rand(3, 16, 16)
    first = corrupt_tensor(image, "noise_s2", spec, seed=1)
    assert torch.equal(first, corrupt_tensor(image, "noise_s2", spec, seed=1))
    assert not torch.equal(first, corrupt_tensor(image, "noise_s2", spec, seed=2))
    assert not torch.equal(first, corrupt_tensor(torch.rand(3, 16, 16), "noise_s2", spec, seed=1))


def test_corrupted_values_are_eight_bit():
    out = corrupt_tensor(torch.rand(3, 16, 16), "brightness_dark_s1", {"type": "brightness", "value": 0.7}, seed=0)
    assert torch.allclose(out * 255, torch.round(out * 255))


def test_loader_keeps_samples_and_labels_across_conditions(val_data):
    def ids(loader):
        labels, paths = [], []
        for _, batch_labels, batch_paths in loader:
            labels += batch_labels.tolist()
            paths += list(batch_paths)
        return labels, paths

    clean = ids(build_loader(val_data, "val"))
    dark = ids(build_loader(val_data, "val", "dark", {"type": "brightness", "value": 0.5}))
    assert dark == clean
    assert len(clean[0]) == 6


def test_loader_is_deterministic_with_noise(val_data):
    spec = {"type": "noise", "value": 0.1}
    one = next(iter(build_loader(val_data, "val", "noise_s2", spec, seed=3)))[0]
    two = next(iter(build_loader(val_data, "val", "noise_s2", spec, seed=3)))[0]
    assert torch.equal(one, two)


def test_examples_and_plot(val_data, tmp_path):
    conditions = load_conditions(_write_config(tmp_path / "c.yaml", locked=False), "val")
    rows = example_rows(val_data, "val", conditions, seed=7, index=0)
    assert [family for family, _ in rows] == ["brightness_dark", "noise"]
    assert all(len(cells) == 3 for _, cells in rows)
    plot_corruption_examples(rows, tmp_path / "grid.png")
    assert (tmp_path / "grid.png").stat().st_size > 0


def test_calibration_end_to_end(val_data, tmp_path, monkeypatch):
    config = _write_config(tmp_path / "c.yaml", locked=False)
    train_cfg = yaml.safe_load((benchmark.ROOT / "configs/train_mobilenetv3_small.yaml").read_text(encoding="utf-8"))
    model = build_model({**train_cfg, "pretrained": False}, num_classes=5)
    torch.save({"state_dict": model.state_dict()}, tmp_path / "mobilenetv3_small_seed42.pt")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection": {"reference": "mobilenetv3_small"}}), encoding="utf-8")
    monkeypatch.setattr(robustness, "SELECTION_PATH", selection)
    monkeypatch.setattr(robustness, "CALIBRATION_PATH", tmp_path / "out" / "calibration.json")
    monkeypatch.setattr(robustness.yaml, "safe_load", lambda text, _real=yaml.safe_load: val_data if "image_size" in text else _real(text))

    record = robustness.run_calibration(config, tmp_path)
    saved = json.loads((tmp_path / "out" / "calibration.json").read_text(encoding="utf-8"))
    assert set(saved["conditions"]) == {"brightness_dark_s1", "brightness_dark_s2", "noise_s1", "noise_s2"}
    assert record["clean_macro_f1"] == saved["clean_macro_f1"]

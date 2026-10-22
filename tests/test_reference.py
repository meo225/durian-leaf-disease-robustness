import json

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from src.corruption import apply_corruption
from src.evaluation import reference
from src.evaluation.benchmark import measure_latency, model_size
from src.models.build import build_model


def _entry(clean, degradation, ms):
    return {
        "clean": {"macro_f1": clean},
        "corrupted": {"degradation": degradation},
        "latency": [{"batch_size": 1, "ms_per_image": ms}, {"batch_size": 32, "ms_per_image": ms / 2}],
    }


@pytest.mark.parametrize(
    "spec, check",
    [
        ({"type": "brightness", "value": 0.5}, lambda x, y: torch.allclose(y, x * 0.5)),
        ({"type": "contrast", "value": 0.0}, lambda x, y: torch.allclose(y, torch.full_like(x, x.mean().item()))),
        ({"type": "blur", "value": 2.0}, lambda x, y: y.std() < x.std()),
        ({"type": "noise", "value": 0.1}, lambda x, y: not torch.equal(x, y)),
    ],
)
def test_corruptions_keep_shape_and_range(spec, check):
    image = torch.rand(3, 32, 32)
    out = apply_corruption(image, spec)
    assert out.shape == image.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert check(image, out)


def test_unknown_corruption_is_rejected():
    with pytest.raises(ValueError):
        apply_corruption(torch.rand(3, 8, 8), {"type": "x", "value": 1})


def test_choose_reference_prefers_clear_winner():
    results = {"a": _entry(0.97, 0.10, 5), "b": _entry(0.93, 0.01, 1)}
    assert reference.choose_reference(results, 0.01)["reference"] == "a"


def test_choose_reference_uses_degradation_then_latency():
    close = {"a": _entry(0.975, 0.10, 5), "b": _entry(0.970, 0.03, 7), "c": _entry(0.90, 0.0, 1)}
    assert reference.choose_reference(close, 0.01)["reference"] == "b"
    tied = {"a": _entry(0.975, 0.030, 9), "b": _entry(0.970, 0.035, 4)}
    assert reference.choose_reference(tied, 0.01)["reference"] == "b"


def test_benchmark_reports_size_and_latency():
    model = torch.nn.Linear(3 * 8 * 8, 5)
    assert model_size(model)["parameters"] == 3 * 8 * 8 * 5 + 5
    flat = torch.nn.Sequential(torch.nn.Flatten(), model)
    result = measure_latency(flat, torch.device("cpu"), batch_size=4, image_size=8, warmup=1, runs=2)
    assert result["ms_per_image"] > 0 and result["images_per_second"] > 0


def test_main_end_to_end_on_small_validation(tmp_path, monkeypatch):
    classes = ["ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK", "HEALTHY_LEAF", "LEAF_BLIGHT", "PHOMOPSIS_LEAF_SPOT"]
    frame = pd.DataFrame(
        {"image_path": [f"img_{i}.jpg" for i in range(10)], "class_name": [classes[i % 5] for i in range(10)], "split": "val"}
    )
    for name in frame["image_path"]:
        Image.fromarray((torch.rand(224, 224, 3) * 255).byte().numpy()).save(tmp_path / name)
    monkeypatch.setattr(reference, "get_split", lambda _name: frame)

    pipeline = yaml.safe_load((reference.ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    pipeline["image_root"] = str(tmp_path)
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    cfg = yaml.safe_load((reference.ROOT / "configs/reference_selection.yaml").read_text(encoding="utf-8"))
    cfg.update(
        pipeline_config=str(tmp_path / "pipeline.yaml"),
        models=["configs/train_mobilenetv3_small.yaml"],
        output=str(tmp_path / "out.json"),
        latency={"warmup": 1, "runs": 2, "batch_sizes": [1, 2]},
    )
    cfg["corruptions"] = {"dark": cfg["corruptions"]["brightness_dark"]}
    (tmp_path / "selection.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    train_cfg = yaml.safe_load((reference.ROOT / "configs/train_mobilenetv3_small.yaml").read_text(encoding="utf-8"))
    model = build_model({**train_cfg, "pretrained": False}, num_classes=5)
    torch.save({"state_dict": model.state_dict()}, tmp_path / "mobilenetv3_small_seed42.pt")

    record = reference.main(tmp_path / "selection.yaml", tmp_path)
    saved = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert saved["selection"]["reference"] == "mobilenetv3_small"
    assert saved["models"]["mobilenetv3_small"]["model"]["parameters"] == 1522981
    assert record["models"]["mobilenetv3_small"]["clean"]["matches_recorded"] is False


def test_collect_checkpoint_rebuilds_unzipped_directory(tmp_path):
    import zipfile

    source = torch.nn.Linear(4, 2)
    original = tmp_path / "m_seed42.pt"
    torch.save({"state_dict": source.state_dict()}, original)
    unpacked = tmp_path / "input" / "m_seed42" / "m_seed42"
    with zipfile.ZipFile(original) as archive:
        for info in archive.infolist():
            archive.extract(info, unpacked.parent)
    assert (unpacked / "data.pkl").is_file()

    path = reference.collect_checkpoint(tmp_path / "input", "m_seed42.pt", tmp_path / "out")
    restored = torch.nn.Linear(4, 2)
    restored.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    assert torch.equal(restored.weight, source.weight)


def test_collect_checkpoint_links_plain_file(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "m_seed42.pt").write_bytes(b"x")
    path = reference.collect_checkpoint(tmp_path / "in", "m_seed42.pt", tmp_path / "out")
    assert path.read_bytes() == b"x"
    with pytest.raises(FileNotFoundError):
        reference.collect_checkpoint(tmp_path / "in", "missing.pt", tmp_path / "out")

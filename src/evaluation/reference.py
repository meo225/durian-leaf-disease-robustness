"""Chọn reference model từ các checkpoint đã train (task 3.3).

Đọc tập validation, không train lại, không mở tập test.
Chạy từ dòng lệnh: python -m src.evaluation.reference configs/reference_selection.yaml <thư mục checkpoint>
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.corruption import apply_corruption
from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import ConditionalResize
from src.evaluation.benchmark import measure_latency, model_size
from src.models.build import build_model
from src.training.loop import evaluate_loader, resolve_device, set_seed

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _val_loader(pipeline_cfg: Mapping[str, Any], spec: Mapping[str, Any] | None) -> DataLoader:
    """Loader validation. Suy giảm được áp trên ảnh 0 đến 1, trước bước chuẩn hóa."""
    steps: list[Any] = [ConditionalResize(int(pipeline_cfg["image_size"])), transforms.ToTensor()]
    if spec is not None:
        steps.append(transforms.Lambda(lambda image: apply_corruption(image, spec)))
    steps.append(transforms.Normalize(pipeline_cfg["mean"], pipeline_cfg["std"]))
    root = Path(pipeline_cfg["image_root"])
    dataset = DurianDataset(
        split_df=get_split("val"),
        image_root=root if root.is_absolute() else ROOT / root,
        transform=transforms.Compose(steps),
        classes=pipeline_cfg["classes"],
    )
    return DataLoader(dataset, batch_size=int(pipeline_cfg["batch_size"]), shuffle=False, num_workers=0)


def collect_checkpoint(search_root: str | Path, file_name: str, target_dir: str | Path) -> Path:
    """Đưa checkpoint về một file `.pt` trong target_dir.

    Kaggle có thể giải nén file `.pt` thành thư mục cùng tên. Khi đó ghép lại thành file zip như torch.save tạo ra.
    """
    search_root, target_dir = Path(search_root), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_name
    if target.exists() or target.is_symlink():
        target.unlink()
    found = sorted(search_root.rglob(file_name))
    if found and found[0].is_file():
        target.symlink_to(found[0])
        return target
    stem = Path(file_name).stem
    for pickle in sorted(search_root.rglob("data.pkl")):
        if pickle.parent.name == stem:
            root = pickle.parent
            members = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: (path != pickle, str(path)))
            with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
                for path in members:
                    archive.write(path, f"{stem}/{path.relative_to(root).as_posix()}")
            return target
    raise FileNotFoundError(f"Không thấy checkpoint {file_name} trong {search_root}.")


def load_trained_model(train_cfg: Mapping[str, Any], checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Dựng đúng kiến trúc rồi nạp trọng số của checkpoint."""
    model = build_model({**train_cfg, "pretrained": False}, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def _single_image_ms(entry: Mapping[str, Any]) -> float:
    return next(item["ms_per_image"] for item in entry["latency"] if item["batch_size"] == 1)


def choose_reference(results: Mapping[str, Mapping[str, Any]], margin: float) -> dict[str, Any]:
    """Áp quy tắc đã khóa trong config. Trả về model được chọn kèm lý do."""
    best_clean = max(entry["clean"]["macro_f1"] for entry in results.values())
    close = [name for name, entry in results.items() if entry["clean"]["macro_f1"] >= best_clean - margin]
    if len(close) == 1:
        return {"reference": close[0], "candidates": close, "reason": "macro F1 sạch cao hơn các model còn lại quá khoảng cho phép"}
    drops = {name: results[name]["corrupted"]["degradation"] for name in close}
    lowest = min(drops.values())
    steady = [name for name in close if drops[name] - lowest < margin]
    if len(steady) == 1:
        return {"reference": steady[0], "candidates": close, "reason": "macro F1 sạch ngang nhau, mức giảm dưới suy giảm thấp nhất"}
    pick = min(steady, key=lambda name: _single_image_ms(results[name]))
    return {"reference": pick, "candidates": close, "reason": "macro F1 sạch và mức giảm ngang nhau, độ trễ batch 1 thấp hơn"}


def main(config_path: str | Path, checkpoint_dir: str | Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    pipeline_cfg = _load_yaml(cfg["pipeline_config"])
    device = resolve_device()
    classes = list(pipeline_cfg["classes"])
    criterion = nn.CrossEntropyLoss()
    results: dict[str, Any] = {}

    for train_path in cfg["models"]:
        train_cfg = _load_yaml(train_path)
        name = str(train_cfg["model_name"])
        checkpoint = Path(checkpoint_dir) / cfg["checkpoint_name"].format(model_name=name)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Không thấy checkpoint: {checkpoint}")
        recorded = json.loads((ROOT / cfg["recorded_result"].format(model_name=name)).read_text(encoding="utf-8"))
        model = load_trained_model(train_cfg, checkpoint, len(classes), device)

        def score(spec: Mapping[str, Any] | None) -> dict[str, Any]:
            set_seed(int(cfg["seed"]))
            return evaluate_loader(model, _val_loader(pipeline_cfg, spec), criterion, device, classes)

        clean = score(None)
        corrupted = {key: score(spec)["macro_f1"] for key, spec in cfg["corruptions"].items()}
        mean_corrupted = sum(corrupted.values()) / len(corrupted)
        latency = [
            measure_latency(model, device, int(batch), int(pipeline_cfg["image_size"]), int(cfg["latency"]["warmup"]), int(cfg["latency"]["runs"]))
            for batch in cfg["latency"]["batch_sizes"]
        ]
        results[name] = {
            "model": {**model_size(model), "library": train_cfg["library"], "backbone": train_cfg["backbone"]},
            "clean": {
                "macro_f1": clean["macro_f1"],
                "balanced_accuracy": clean["balanced_accuracy"],
                "per_class_recall": clean["per_class_recall"],
                "recorded_macro_f1": recorded["val_macro_f1"],
                "matches_recorded": abs(clean["macro_f1"] - recorded["val_macro_f1"]) < 0.005,
            },
            "corrupted": {
                "macro_f1": corrupted,
                "mean_macro_f1": mean_corrupted,
                "degradation": clean["macro_f1"] - mean_corrupted,
            },
            "latency": latency,
        }
        print(f"{name}: clean {clean['macro_f1']:.4f}  corrupted {mean_corrupted:.4f}  {_single_image_ms({'latency': latency})} ms/ảnh", flush=True)

    record = {
        "device": str(torch.cuda.get_device_name(device)) if device.type == "cuda" else "cpu",
        "close_margin": cfg["close_margin"],
        "corruptions": cfg["corruptions"],
        "models": results,
        "selection": choose_reference(results, float(cfg["close_margin"])),
    }
    output = ROOT / cfg["output"]
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reference model: {record['selection']['reference']} ({record['selection']['reason']})", flush=True)
    return record


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Dùng: python -m src.evaluation.reference configs/reference_selection.yaml <thư mục checkpoint>")
    main(sys.argv[1], sys.argv[2])

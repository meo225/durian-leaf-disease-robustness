"""Đánh giá một model trên ảnh sạch và từng điều kiện suy giảm của benchmark.

Cùng một hàm dùng cho hiệu chỉnh mức trên validation (task 4.1) và cho đánh giá độ nhạy trên test (task 4.2).
Chạy hiệu chỉnh: python -m src.evaluation.robustness configs/corruption.yaml <thư mục checkpoint>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

from src.corruption import build_loader, load_conditions, load_seed
from src.evaluation.reference import load_trained_model
from src.training.loop import evaluate_loader, resolve_device

ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = ROOT / "reports" / "baselines" / "reference_selection.json"
CALIBRATION_PATH = ROOT / "reports" / "corruptions" / "validation_calibration.json"


@torch.no_grad()
def evaluate_conditions(
    model: nn.Module,
    pipeline_cfg: Mapping[str, Any],
    split: str,
    conditions: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Điểm trên ảnh sạch và trên từng điều kiện, cùng các mẫu và nhãn. Degradation là F1 sạch trừ F1 suy giảm."""
    classes = list(pipeline_cfg["classes"])
    criterion = nn.CrossEntropyLoss()
    clean = evaluate_loader(model, build_loader(pipeline_cfg, split), criterion, device, classes)
    results = {}
    for name, spec in conditions.items():
        scores = evaluate_loader(model, build_loader(pipeline_cfg, split, name, spec, seed), criterion, device, classes)
        results[name] = {**scores, "degradation": clean["macro_f1"] - scores["macro_f1"]}
    return {"clean": clean, "conditions": results}


def run_calibration(config_path: str | Path, checkpoint_dir: str | Path) -> dict[str, Any]:
    """Đo reference model trên validation qua mọi điều kiện, để kiểm tra các mức trước khi khóa."""
    reference = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selection"]["reference"]
    pipeline_cfg = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / f"train_{reference}.yaml").read_text(encoding="utf-8"))
    device = resolve_device()
    checkpoint = Path(checkpoint_dir) / f"{reference}_seed42.pt"
    model = load_trained_model(train_cfg, checkpoint, len(pipeline_cfg["classes"]), device)
    conditions = load_conditions(config_path, "val")
    result = evaluate_conditions(model, pipeline_cfg, "val", conditions, device, load_seed(config_path))
    record = {
        "split": "val",
        "reference_model": reference,
        "clean_macro_f1": result["clean"]["macro_f1"],
        "conditions": {
            name: {
                "type": conditions[name]["type"],
                "value": conditions[name]["value"],
                "macro_f1": entry["macro_f1"],
                "balanced_accuracy": entry["balanced_accuracy"],
                "degradation": entry["degradation"],
            }
            for name, entry in result["conditions"].items()
        },
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, entry in record["conditions"].items():
        print(f"{name:20s} giá trị {entry['value']:<5}  macro F1 {entry['macro_f1']:.4f}  giảm {entry['degradation']:.4f}", flush=True)
    return record


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Dùng: python -m src.evaluation.robustness configs/corruption.yaml <thư mục checkpoint>")
    run_calibration(sys.argv[1], sys.argv[2])

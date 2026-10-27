"""Giai đoạn A của bản đồ phục hồi (task 4.5): dự đoán của reference model đóng băng trên ảnh test, không xử lý và đã xử lý.

Với mỗi điều kiện suy giảm của benchmark 4.1, ảnh suy giảm được tạo một lần, rồi mỗi kỹ thuật xử lý ảnh của task 4.3 và 4.4 chạy lên chính
các ảnh đó bằng cấu hình đã khóa trong `reports/processing/*_selected.json`. Cấu hình của một họ là cấu hình tốt nhất của họ nếu họ phù hợp với
kỹ thuật, không thì cấu hình chung. Ảnh sạch cũng chạy qua mọi cấu hình đã dùng để đo chi phí. Không chỉnh gì theo kết quả test.

Kết quả là nhãn dự đoán và xác suất từng ảnh, để giai đoạn B (recovery_report.py) tính khoảng tin cậy, McNemar, oracle trên CPU.
Test chỉ mở được khi FINAL_EVAL=1 và config suy giảm đã khóa.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torchvision import transforms

from src.corruption import corrupt_tensor
from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import ConditionalResize
from src.evaluation.processing_benchmark import make_processor, selected_config

ROOT = Path(__file__).resolve().parents[2]
NONE = ("none", "none")


def key_of(technique: str, config_id: str, condition: str) -> str:
    """Khóa của một lượt chạy trong file .npz. Xác suất có thêm hậu tố __probs."""
    return f"{technique}__{config_id}__{condition}"


def load_selected(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Gộp cấu hình đã chọn của các task xử lý ảnh. Tên kỹ thuật không được trùng nhau giữa các file."""
    merged: dict[str, Any] = {}
    for path in paths:
        full = Path(path)
        record = json.loads((full if full.is_absolute() else ROOT / full).read_text(encoding="utf-8"))
        for technique, entry in record["selected"].items():
            if technique in merged:
                raise ValueError(f"Kỹ thuật {technique} xuất hiện ở hai file cấu hình.")
            merged[technique] = entry
    return merged


def task_scopes(config_paths: Sequence[str | Path]) -> dict[str, list[str]]:
    """Họ nằm trong scope của mỗi kỹ thuật theo config của task (scope_families), để chạy chế độ `task`."""
    scopes: dict[str, list[str]] = {}
    for path in config_paths:
        full = Path(path)
        cfg = yaml.safe_load((full if full.is_absolute() else ROOT / full).read_text(encoding="utf-8"))
        for technique in cfg["techniques"]:
            scopes[technique] = list(cfg["scope_families"])
    return scopes


def jobs_for_family(
    selected: Mapping[str, Any], family: str, scopes: Mapping[str, Sequence[str]] | None = None
) -> list[dict[str, Any]]:
    """Các lượt xử lý cần chạy trên một họ. scopes=None là chạy mọi kỹ thuật trên mọi họ.

    Mỗi kỹ thuật chạy cấu hình dành cho họ này (role family). Nếu kỹ thuật có cấu hình riêng khác cấu hình chung thì chạy thêm cấu hình chung
    (role general), để đo "kỹ thuật cố định" áp một cấu hình lên mọi ảnh.
    """
    jobs = []
    for technique, entry in selected.items():
        if scopes is not None and family not in scopes.get(technique, []):
            continue
        config_id, params = selected_config(selected, technique, family)
        suitable = family in entry["per_family"]
        jobs.append({"technique": technique, "config_id": config_id, "params": dict(params), "suitable": suitable, "role": "family"})
        general = entry["general"]
        if general["config_id"] != config_id:
            jobs.append({"technique": technique, "config_id": general["config_id"], "params": dict(general["params"]), "suitable": suitable, "role": "general"})
    return jobs


def clean_jobs(selected: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Mọi cấu hình đã dùng ở bất kỳ họ nào (cấu hình từng họ và cấu hình chung), mỗi cấu hình một lần, để đo chi phí trên ảnh sạch."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for technique, entry in selected.items():
        for chosen in [*entry["per_family"].values(), entry["general"]]:
            seen.setdefault((technique, chosen["config_id"]), {"technique": technique, "config_id": chosen["config_id"], "params": dict(chosen["params"]), "suitable": None})
    return list(seen.values())


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    processor: Callable[[torch.Tensor], torch.Tensor] | None,
    pipeline_cfg: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 64,
    workers: int = 1,
) -> np.ndarray:
    """Xác suất softmax của model trên ảnh uint8 N×3×H×W, sau khi xử lý (nếu có) và chuẩn hóa như pipeline chuẩn."""
    model.eval()
    mean = torch.tensor(pipeline_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(pipeline_cfg["std"], device=device).view(1, 3, 1, 1)
    pool = ThreadPoolExecutor(max_workers=workers) if processor is not None and workers > 1 else None
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].float() / 255.0
        if processor is not None:
            batch = torch.stack(list(pool.map(processor, batch))) if pool is not None else torch.stack([processor(image) for image in batch])
        logits = model((batch.to(device) - mean) / std)
        outputs.append(torch.softmax(logits.float(), dim=1).cpu())
    if pool is not None:
        pool.shutdown()
    return torch.cat(outputs).numpy()


def load_clean_images(pipeline_cfg: Mapping[str, Any], split: str) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Ảnh sạch của split (uint8 N×3×H×W), nhãn và đường dẫn theo đúng thứ tự manifest."""
    root = Path(pipeline_cfg["image_root"])
    dataset = DurianDataset(
        split_df=get_split(split),
        image_root=root if root.is_absolute() else ROOT / root,
        transform=transforms.Compose([ConditionalResize(int(pipeline_cfg["image_size"])), transforms.ToTensor()]),
        classes=pipeline_cfg["classes"],
    )
    items = [dataset[index] for index in range(len(dataset))]
    images = torch.stack([torch.round(item[0] * 255).to(torch.uint8) for item in items])
    return images, np.array([item[1] for item in items], dtype=np.int64), [item[2] for item in items]


def run_map(
    model: nn.Module,
    pipeline_cfg: Mapping[str, Any],
    split: str,
    conditions: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, Any],
    device: torch.device,
    seed: int,
    scopes: Mapping[str, Sequence[str]] | None = None,
    workers: int = 1,
    progress: Callable[[str], None] = print,
    on_condition_done: Callable[[dict[str, Any]], None] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Chạy toàn bộ bản đồ. Trả về nhãn, đường dẫn ảnh, dự đoán và xác suất từng lượt chạy, và bảng mô tả các lượt chạy."""
    clean, labels, paths = load_clean_images(pipeline_cfg, split)
    if limit is not None:
        clean, labels, paths = clean[:limit], labels[:limit], paths[:limit]
    result: dict[str, Any] = {"labels": labels, "paths": paths, "predictions": {}, "probabilities": {}, "rows": [], "split": split}

    def record(technique: str, config_id: str, condition: str, family: str, suitable: bool | None, images: torch.Tensor, params: Mapping[str, Any] | None, role: str = "family") -> None:
        processor = None if params is None else make_processor(technique, params)
        probabilities = predict_probabilities(model, images, processor, pipeline_cfg, device, workers=workers)
        key = key_of(technique, config_id, condition)
        result["predictions"][key] = probabilities.argmax(axis=1).astype(np.int8)
        result["probabilities"][key] = probabilities.astype(np.float16)
        result["rows"].append({"technique": technique, "config_id": config_id, "condition": condition, "family": family, "suitable": suitable, "role": role})

    record(*NONE, "clean", "clean", None, clean, None, "none")
    for job in clean_jobs(selected):
        record(job["technique"], job["config_id"], "clean", "clean", None, clean, job["params"], "clean")
    progress(f"clean: {len(clean)} ảnh, {len(clean_jobs(selected))} cấu hình")
    if on_condition_done is not None:
        on_condition_done(result)

    for name, spec in conditions.items():
        degraded = torch.stack([torch.round(corrupt_tensor(image.float() / 255.0, name, spec, seed) * 255).to(torch.uint8) for image in clean])
        family = str(spec["family"])
        record(*NONE, name, family, None, degraded, None, "none")
        jobs = jobs_for_family(selected, family, scopes)
        for job in jobs:
            record(job["technique"], job["config_id"], name, family, job["suitable"], degraded, job["params"], job["role"])
        progress(f"{name}: {len(jobs)} kỹ thuật")
        if on_condition_done is not None:
            on_condition_done(result)
    return result


def save_run(result: Mapping[str, Any], path: str | Path) -> None:
    """Ghi nhãn, đường dẫn, dự đoán, xác suất và bảng mô tả vào một file .npz."""
    arrays: dict[str, Any] = {"labels": result["labels"], "paths": np.array(result["paths"]), "rows_json": np.array(json.dumps(result["rows"], ensure_ascii=False))}
    arrays.update(result["predictions"])
    arrays.update({f"{key}__probs": value for key, value in result["probabilities"].items()})
    np.savez_compressed(path, **arrays)

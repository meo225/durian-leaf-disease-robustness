"""Đánh giá một model trên ảnh sạch và từng điều kiện suy giảm, lưu điểm, nhãn dự đoán và xác suất từng ảnh (task 4.5).

Cùng đường đi ảnh với benchmark 4.1 (build_loader), nên mẫu, nhãn và suy giảm giống hệt phép đo của task 4.2.
Dòng kết quả cùng khóa với bảng của task 4.3 và 4.4 (role, technique, config_id, condition, family, severity, macro_f1, ...),
nên bản đồ phục hồi ghép được. Thêm macro_precision, và xác suất từng ảnh để tính PR-AUC ở các bước sau.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score
from torch import nn

from src.corruption import build_loader
from src.evaluation.metrics import classification_scores


@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Nhãn thật và xác suất softmax của mọi ảnh trong loader, theo đúng thứ tự loader."""
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        images, targets = batch[0], batch[1]
        logits = model(images.to(device, non_blocking=device.type == "cuda"))
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(targets.cpu())
    return torch.cat(labels).numpy().astype(np.int64), torch.cat(probabilities).numpy()


def _split_condition(condition: str) -> tuple[str, int]:
    family, severity = condition.rsplit("_s", 1)
    return family, int(severity)


def evaluate_with_predictions(
    model: nn.Module,
    pipeline_cfg: Mapping[str, Any],
    split: str,
    conditions: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    seed: int,
    technique: str,
    config_id: str,
    role: str = "retrained",
    progress: Any = print,
) -> dict[str, Any]:
    """Chạy model trên ảnh sạch rồi từng điều kiện. Trả về bảng điểm, nhãn thật, đường dẫn ảnh và dự đoán từng ảnh.

    split là val hoặc test. Test chỉ mở được khi FINAL_EVAL=1 (get_split) và khi config suy giảm đã khóa (load_conditions).
    """
    classes = list(pipeline_cfg["classes"])
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    labels_ref: np.ndarray | None = None
    paths: list[str] = []
    clean_f1 = float("nan")

    for condition, spec in [("clean", None), *conditions.items()]:
        loader = build_loader(pipeline_cfg, split, None if spec is None else condition, spec, seed)
        labels, probabilities = predict_probabilities(model, loader, device)
        if labels_ref is None:
            labels_ref = labels
            paths = [str(path) for path in loader.dataset.split_df["image_path"]]
        elif not np.array_equal(labels, labels_ref):
            raise RuntimeError(f"Thứ tự mẫu của {condition} khác ảnh sạch.")
        preds = probabilities.argmax(axis=1)
        scores = classification_scores(labels, preds, classes)
        family, severity = ("clean", 0) if condition == "clean" else _split_condition(condition)
        if condition == "clean":
            clean_f1 = scores["macro_f1"]
        row = {
            "role": role,
            "technique": technique,
            "config_id": config_id,
            "condition": condition,
            "family": family,
            "severity": severity,
            "macro_f1": scores["macro_f1"],
            "balanced_accuracy": scores["balanced_accuracy"],
            "accuracy": scores["accuracy"],
            "macro_precision": float(precision_score(labels, preds, labels=list(range(len(classes))), average="macro", zero_division=0)),
            "degradation": clean_f1 - scores["macro_f1"],
        }
        row.update({f"recall_{name}": value for name, value in scores["per_class_recall"].items()})
        rows.append(row)
        key = f"{technique}__{config_id}__{condition}"
        predictions[key] = preds.astype(np.int8)
        predictions[f"{key}__probs"] = probabilities.astype(np.float16)
        progress(f"{condition:20s} macro F1 {scores['macro_f1']:.4f}")

    assert labels_ref is not None
    return {"frame": pd.DataFrame(rows), "labels": labels_ref, "paths": paths, "predictions": predictions}


def save_predictions(result: Mapping[str, Any], path: str | Path) -> None:
    """Ghi nhãn, đường dẫn ảnh, dự đoán và xác suất từng ảnh vào một file .npz."""
    np.savez_compressed(path, labels=result["labels"], paths=np.array(result["paths"]), **result["predictions"])

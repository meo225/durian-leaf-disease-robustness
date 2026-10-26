"""Baseline truyền thống HOG + SVM, chạy trên CPU và chọn theo validation.

Dùng cùng split khóa, cùng ảnh 224 và cùng chỉ số với các baseline học sâu. Không augmentation,
không tối ưu siêu tham số. Tập test không được mở. Kết quả ghi vào outputs/.
Chạy từ dòng lệnh: python -m src.training.hog_svm configs/train_hog_svm.yaml
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import ConditionalResize
from src.evaluation import plot_confusion_matrix
from src.evaluation.metrics import classification_scores
from src.training.loop import set_seed
from src.training.run import ROOT, load_configs


def hog_features(image: Image.Image, hog_cfg: dict[str, Any]) -> np.ndarray:
    """Đặc trưng HOG của một ảnh RGB (PIL), tính trên ảnh xám."""
    cell = int(hog_cfg["cell_size"])
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=int(hog_cfg["bins"]),
        pixels_per_cell=(cell, cell),
        cells_per_block=(int(hog_cfg["block_cells"]),) * 2,
        block_norm="L2-Hys",
    ).astype(np.float32)


def extract_split(
    split: str, pipeline_cfg: dict[str, Any], hog_cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Đặc trưng HOG và nhãn của một split. Chỉ train và val."""
    if split not in ("train", "val"):
        raise ValueError("Baseline HOG + SVM chỉ dùng train và val.")
    size = int(pipeline_cfg["image_size"])
    resize = ConditionalResize(size=size)
    root = Path(pipeline_cfg.get("image_root") or pipeline_cfg["dataset_root"])
    dataset = DurianDataset(
        split_df=get_split(split),
        image_root=root if root.is_absolute() else ROOT / root,
        transform=lambda img: hog_features(resize(img), hog_cfg),
        classes=pipeline_cfg["classes"],
    )
    features, labels = [], []
    for index in range(len(dataset)):
        # Transform trả về mảng HOG nên DurianDataset giữ nguyên mảng numpy.
        feature, label, _ = dataset[index]
        features.append(feature)
        labels.append(label)
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def build_classifier(svm_cfg: dict[str, Any], seed: int):
    """Chuẩn hóa từng chiều rồi SVM."""
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=float(svm_cfg["C"]),
            gamma=svm_cfg["gamma"],
            class_weight=svm_cfg.get("class_weight"),
            random_state=seed,
        ),
    )


def build_record(
    scores: dict[str, Any],
    train_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    feature_dim: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Record JSON cùng khóa với baseline học sâu. Khóa không áp dụng (loss, history) bỏ trống."""
    return {
        "model_name": train_cfg["model_name"],
        "library": train_cfg["library"],
        "backbone": train_cfg["backbone"],
        "train_mode": train_cfg["train_mode"],
        "seed": int(train_cfg["seed"]),
        "best_epoch": None,
        "epochs_run": None,
        "selection_metric": train_cfg["selection_metric"],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "resource": "cpu",
        "trainable_parameters": None,
        "feature_dimension": feature_dim,
        "hyperparameters": {
            "hog": train_cfg["hog"],
            "svm": train_cfg["svm"],
            "image_size": pipeline_cfg["image_size"],
            "grayscale": True,
            "augmentation": False,
        },
        "classes": list(pipeline_cfg["classes"]),
        "val_macro_f1": scores["macro_f1"],
        "val_balanced_accuracy": scores["balanced_accuracy"],
        "val_accuracy": scores["accuracy"],
        "val_loss": None,
        "per_class_recall": scores["per_class_recall"],
        "confusion_matrix": scores["confusion_matrix"],
        "history": [],
    }


def main(config_path: str | Path) -> dict[str, str]:
    train_cfg, pipeline_cfg = load_configs(config_path)
    seed = int(train_cfg["seed"])
    set_seed(seed)
    classes = list(pipeline_cfg["classes"])

    start = time.perf_counter()
    x_train, y_train = extract_split("train", pipeline_cfg, train_cfg["hog"])
    x_val, y_val = extract_split("val", pipeline_cfg, train_cfg["hog"])
    print(f"Ảnh train: {len(y_train)}, validation: {len(y_val)}, số chiều HOG: {x_train.shape[1]}", flush=True)
    print("Tập test: không mở", flush=True)

    classifier = build_classifier(train_cfg["svm"], seed)
    classifier.fit(x_train, y_train)
    scores = classification_scores(y_val, classifier.predict(x_val), classes)
    elapsed = time.perf_counter() - start

    record = build_record(scores, train_cfg, pipeline_cfg, int(x_train.shape[1]), elapsed)
    stem = f"{record['model_name']}_seed{record['seed']}"
    output_dir = ROOT / "outputs"
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    paths = {
        "checkpoint": output_dir / "checkpoints" / f"{stem}.joblib",
        "metrics": output_dir / "results" / f"{stem}.json",
        "confusion_matrix": output_dir / "results" / f"{stem}_confusion_matrix.png",
    }
    joblib.dump({"classifier": classifier, "classes": classes, "hog": train_cfg["hog"]}, paths["checkpoint"])
    paths["metrics"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    plot_confusion_matrix(record["confusion_matrix"], classes, paths["confusion_matrix"])

    print("Thiết bị: cpu", flush=True)
    print(f"Thời gian trích đặc trưng và train (giây): {elapsed:.1f}", flush=True)
    print(f"Val macro F1: {scores['macro_f1']:.4f}", flush=True)
    print(f"Val balanced accuracy: {scores['balanced_accuracy']:.4f}", flush=True)
    print("Recall từng lớp:", flush=True)
    for name, value in scores["per_class_recall"].items():
        print(f"  {name}: {value:.4f}", flush=True)
    for label, key in (("Checkpoint", "checkpoint"), ("Metric", "metrics"), ("Confusion matrix", "confusion_matrix")):
        print(f"{label}: {paths[key]}", flush=True)
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Dùng: python -m src.training.hog_svm configs/<file>.yaml")
    main(sys.argv[1])

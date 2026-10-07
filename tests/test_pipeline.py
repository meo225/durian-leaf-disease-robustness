from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option("mode.string_storage", "python")
from PIL import Image
import pytest
import torch
from torchvision import transforms

from src.data.dataloaders import build_dataloaders
from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import (
    ConditionalResize,
    build_eval_transform,
    build_train_transform,
)

FORBIDDEN_TRANSFORM_NAMES = {
    "colorjitter",
    "gamma",
    "clahe",
    "retinex",
    "denoise",
    "gaussianblur",
    "medianblur",
    "sharpen",
}


@pytest.fixture
def dummy_pipeline_data(tmp_path: Path):
    img_dir = tmp_path / "dummy_images"
    img_dir.mkdir()

    img_224_path = img_dir / "leaf_224.jpg"
    arr_224 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr_224).save(img_224_path)

    img_300_path = img_dir / "leaf_300.jpg"
    arr_300 = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    Image.fromarray(arr_300).save(img_300_path)

    classes = [
        "ALGAL_LEAF_SPOT",
        "ALLOCARIDARA_ATTACK",
        "HEALTHY_LEAF",
        "LEAF_BLIGHT",
        "PHOMOPSIS_LEAF_SPOT",
    ]

    df = pd.DataFrame([
        {"image_path": "dummy_images/leaf_224.jpg", "class_name": "HEALTHY_LEAF"},
        {"image_path": "dummy_images/leaf_300.jpg", "class_name": "ALGAL_LEAF_SPOT"},
    ])

    cfg = {
        "seed": 42,
        "image_root": str(tmp_path),
        "dataset_root": str(tmp_path),
        "image_size": 224,
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "hflip_prob": 0.5,
        "vflip_prob": 0.5,
        "rotation_degrees": 15,
        "classes": classes,
        "num_classes": len(classes),
    }

    return {
        "root": tmp_path,
        "img_224": img_224_path,
        "img_300": img_300_path,
        "df": df,
        "cfg": cfg,
        "classes": classes,
    }


def test_transform_shape_and_resize(dummy_pipeline_data) -> None:
    cfg = dummy_pipeline_data["cfg"]
    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    with Image.open(dummy_pipeline_data["img_224"]) as img:
        t_eval = eval_tf(img)
        t_train = train_tf(img)
        assert t_eval.shape == (3, 224, 224)
        assert t_train.shape == (3, 224, 224)

    with Image.open(dummy_pipeline_data["img_300"]) as img:
        t_eval_resized = eval_tf(img)
        t_train_resized = train_tf(img)
        assert t_eval_resized.shape == (3, 224, 224)
        assert t_train_resized.shape == (3, 224, 224)


def test_eval_transform_is_deterministic(dummy_pipeline_data) -> None:
    cfg = dummy_pipeline_data["cfg"]
    eval_tf = build_eval_transform(cfg)

    with Image.open(dummy_pipeline_data["img_224"]) as img:
        t1 = eval_tf(img)
        t2 = eval_tf(img)
        assert torch.equal(t1, t2), "eval_transform phải có tính xác định (deterministic)"


def test_train_transform_random_and_reproducible_with_seed(dummy_pipeline_data) -> None:
    cfg = dummy_pipeline_data["cfg"]
    train_tf = build_train_transform(cfg)

    with Image.open(dummy_pipeline_data["img_224"]) as img:
        torch.manual_seed(1234)
        t_seed1_run1 = train_tf(img)

        torch.manual_seed(1234)
        t_seed1_run2 = train_tf(img)

        assert torch.equal(
            t_seed1_run1, t_seed1_run2
        ), "train_transform phải cho kết quả giống nhau khi cùng seed"


def test_no_forbidden_operations_in_transforms(dummy_pipeline_data) -> None:
    cfg = dummy_pipeline_data["cfg"]
    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    all_transforms = list(train_tf.transforms) + list(eval_tf.transforms)
    for tf in all_transforms:
        cls_name = tf.__class__.__name__.lower()
        for forbidden in FORBIDDEN_TRANSFORM_NAMES:
            assert forbidden not in cls_name, f"Phát hiện thao tác cấm trong pipeline chuẩn: {tf}"


def test_durian_dataset_dummy(dummy_pipeline_data) -> None:
    df = dummy_pipeline_data["df"]
    root = dummy_pipeline_data["root"]
    classes = dummy_pipeline_data["classes"]
    cfg = dummy_pipeline_data["cfg"]

    eval_tf = build_eval_transform(cfg)
    dataset = DurianDataset(df, image_root=root, transform=eval_tf, classes=classes)

    assert len(dataset) == 2

    tensor0, label0, path0 = dataset[0]
    assert tensor0.shape == (3, 224, 224)
    assert label0 == classes.index("HEALTHY_LEAF")
    assert path0 == "dummy_images/leaf_224.jpg"

    tensor1, label1, path1 = dataset[1]
    assert tensor1.shape == (3, 224, 224)
    assert label1 == classes.index("ALGAL_LEAF_SPOT")
    assert path1 == "dummy_images/leaf_300.jpg"


def test_build_dataloaders_default_does_not_touch_test(dummy_pipeline_data) -> None:
    cfg = dummy_pipeline_data["cfg"]
    loaders = build_dataloaders(cfg, include_test=False)

    assert "train" in loaders
    assert "val" in loaders
    assert "test" not in loaders


def test_get_split_test_blocked_without_final_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINAL_EVAL", raising=False)
    with pytest.raises(PermissionError, match="Tập test đã bị khóa"):
        get_split("test")


def test_train_and_val_files_exist_on_disk() -> None:
    dataset_root = Path("datasets/durian-ldd")
    if not dataset_root.is_dir():
        pytest.skip("Thư mục datasets/durian-ldd không tồn tại, bỏ qua kiểm tra ảnh thật.")

    train_df = get_split("train")
    val_df = get_split("val")

    for p in train_df["image_path"]:
        assert (dataset_root / p).is_file(), f"Thiếu ảnh train: {p}"

    for p in val_df["image_path"]:
        assert (dataset_root / p).is_file(), f"Thiếu ảnh val: {p}"

from PIL import Image
import torch
from torchvision import transforms

from src.data.transforms import (
    ConditionalResize,
    build_eval_transform,
    build_train_transform,
)


def test_conditional_resize_skips_when_size_matches() -> None:
    img = Image.new("RGB", (224, 224), color="red")
    resizer = ConditionalResize(size=224)
    out = resizer(img)
    assert out is img
    assert out.size == (224, 224)


def test_conditional_resize_resizes_when_size_differs() -> None:
    img = Image.new("RGB", (300, 400), color="blue")
    resizer = ConditionalResize(size=224)
    out = resizer(img)
    assert out.size == (224, 224)


def test_build_train_transform_structure_and_output() -> None:
    cfg = {
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "hflip_prob": 0.5,
        "vflip_prob": 0.5,
        "rotation_degrees": 15,
    }
    train_tf = build_train_transform(cfg)
    assert isinstance(train_tf, transforms.Compose)
    assert len(train_tf.transforms) == 6

    assert isinstance(train_tf.transforms[0], ConditionalResize)
    assert isinstance(train_tf.transforms[1], transforms.RandomHorizontalFlip)
    assert isinstance(train_tf.transforms[2], transforms.RandomVerticalFlip)
    assert isinstance(train_tf.transforms[3], transforms.RandomRotation)
    assert isinstance(train_tf.transforms[4], transforms.ToTensor)
    assert isinstance(train_tf.transforms[5], transforms.Normalize)

    img = Image.new("RGB", (256, 256), color="green")
    tensor = train_tf(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_build_eval_transform_structure_and_output() -> None:
    cfg = {
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    eval_tf = build_eval_transform(cfg)
    assert isinstance(eval_tf, transforms.Compose)
    assert len(eval_tf.transforms) == 3

    assert isinstance(eval_tf.transforms[0], ConditionalResize)
    assert isinstance(eval_tf.transforms[1], transforms.ToTensor)
    assert isinstance(eval_tf.transforms[2], transforms.Normalize)

    img = Image.new("RGB", (224, 224), color="yellow")
    tensor = eval_tf(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

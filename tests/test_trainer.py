import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.mobilenet import build_mobilenet_v3_small
from src.training import trainer
from src.training.trainer import evaluate, fit, load_checkpoint

CLASSES = ["a", "b", "c"]


class TinyDataset(Dataset):
    def __init__(self, size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(size, 3, 64, 64, generator=generator)
        self.labels = torch.arange(size) % len(CLASSES)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index]), f"img_{index}"


@pytest.fixture
def loaders():
    return {
        "train": DataLoader(TinyDataset(24, seed=0), batch_size=8, shuffle=True),
        "val": DataLoader(TinyDataset(12, seed=1), batch_size=6),
    }


@pytest.fixture
def cfg():
    return {"lr": 1e-3, "weight_decay": 1e-2, "epochs": 3, "patience": 5}


def new_model():
    return build_mobilenet_v3_small(num_classes=len(CLASSES), pretrained=False)


def test_fit_saves_best_checkpoint(loaders, cfg, tmp_path) -> None:
    checkpoint = tmp_path / "ckpt" / "best.pt"
    result = fit(new_model(), loaders, cfg, torch.device("cpu"), CLASSES, checkpoint)

    assert checkpoint.is_file()
    assert result["epochs_run"] == 3
    assert len(result["history"]) == 3
    assert 1 <= result["best_epoch"] <= 3
    assert result["best_macro_f1"] == max(row["val_macro_f1"] for row in result["history"])
    assert result["optimizer"] == "AdamW"
    assert result["scheduler"] == "CosineAnnealingLR"
    assert result["train_seconds"] > 0


def test_checkpoint_reproduces_best_val_score(loaders, cfg, tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    device = torch.device("cpu")
    result = fit(new_model(), loaders, cfg, device, CLASSES, checkpoint)

    restored = new_model()
    info = load_checkpoint(restored, checkpoint, device)
    scores = evaluate(restored, loaders["val"], device, CLASSES)

    assert info["epoch"] == result["best_epoch"]
    assert scores["macro_f1"] == pytest.approx(result["best_macro_f1"])


def test_early_stopping(loaders, cfg, tmp_path, monkeypatch) -> None:
    constant = {
        "loss": 1.0,
        "macro_f1": 0.5,
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "per_class_recall": {},
        "confusion_matrix": [],
    }
    monkeypatch.setattr(trainer, "evaluate", lambda *args, **kwargs: dict(constant))
    cfg = {**cfg, "epochs": 20, "patience": 3}

    result = fit(new_model(), loaders, cfg, torch.device("cpu"), CLASSES, tmp_path / "best.pt")

    assert result["best_epoch"] == 1
    assert result["epochs_run"] == 4


def test_evaluate_without_criterion_has_no_loss(loaders) -> None:
    scores = evaluate(new_model(), loaders["val"], torch.device("cpu"), CLASSES)
    assert scores["loss"] is None
    assert len(scores["confusion_matrix"]) == len(CLASSES)

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.loop import fit_model
from src.training.metrics import classification_scores


class _NamedTensorDataset(TensorDataset):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, f"sample-{index}"


def test_classification_scores_known_predictions():
    scores = classification_scores(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        class_names=["a", "b"],
    )

    assert scores["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert scores["balanced_accuracy"] == pytest.approx(0.75)
    assert scores["per_class_recall"] == {"a": pytest.approx(0.5), "b": pytest.approx(1.0)}
    assert scores["confusion_matrix"] == [[1, 1], [0, 2]]


def test_fit_model_keeps_best_validation_epoch_and_ignores_paths():
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(8, 4, generator=generator)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = _NamedTensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = nn.Linear(4, 2)
    result = fit_model(
        model=model,
        train_loader=loader,
        val_loader=loader,
        class_names=["a", "b"],
        train_cfg={
            "optimizer": "adamw",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "epochs": 2,
            "patience": 2,
            "min_delta": 0.0,
            "selection_metric": "macro_f1",
        },
        device=torch.device("cpu"),
    )

    assert result["best_epoch"] in {1, 2}
    assert 0.0 <= result["best_val"]["macro_f1"] <= 1.0
    assert len(result["history"]) == 2
    assert set(result["state_dict"]) == set(model.state_dict())

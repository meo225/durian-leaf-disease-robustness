import torch

from src.models.mobilenet import build_mobilenet_v3_small, count_parameters


def test_head_matches_num_classes() -> None:
    model = build_mobilenet_v3_small(num_classes=5, pretrained=False)
    assert model.classifier[-1].out_features == 5

    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, 5)


def test_dropout_is_configurable() -> None:
    model = build_mobilenet_v3_small(num_classes=5, pretrained=False, dropout=0.35)
    assert model.classifier[2].p == 0.35


def test_count_parameters() -> None:
    model = build_mobilenet_v3_small(num_classes=5, pretrained=False)
    counts = count_parameters(model)
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["trainable"] == counts["total"]

    for param in model.features.parameters():
        param.requires_grad = False
    frozen = count_parameters(model)
    assert frozen["total"] == counts["total"]
    assert frozen["trainable"] < frozen["total"]

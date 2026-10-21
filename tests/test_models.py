import pytest
import torch

from src.models.build import build_model


@pytest.mark.parametrize(
    "library, backbone",
    [("timm", "resnet50"), ("torchvision", "mobilenet_v3_small")],
)
def test_build_model_outputs_one_logit_per_class(library, backbone):
    cfg = {"library": library, "backbone": backbone, "pretrained": False}
    model = build_model(cfg, num_classes=5).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 5)


def test_build_model_rejects_unknown_library():
    with pytest.raises(ValueError):
        build_model({"library": "x", "backbone": "y", "pretrained": False}, num_classes=5)

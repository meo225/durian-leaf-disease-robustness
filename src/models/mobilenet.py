"""MobileNetV3-Small cho phân loại bệnh lá sầu riêng."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenet_v3_small(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Tạo MobileNetV3-Small, nạp trọng số ImageNet nếu cần và thay lớp phân loại cuối."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights, dropout=dropout)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Đếm tham số tổng và tham số được huấn luyện."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

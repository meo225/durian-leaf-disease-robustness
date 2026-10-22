"""Tạo model phân loại từ cấu hình train."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn


def _build_timm(backbone: str, num_classes: int, pretrained: bool, dropout: float | None) -> nn.Module:
    import timm

    extra = {} if dropout is None else {"drop_rate": float(dropout)}
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, **extra)


def _build_torchvision(backbone: str, num_classes: int, pretrained: bool, dropout: float | None) -> nn.Module:
    from torchvision import models

    if backbone != "mobilenet_v3_small":
        raise ValueError(f"torchvision backbone chưa hỗ trợ: {backbone}.")
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    extra = {} if dropout is None else {"dropout": float(dropout)}
    model = models.mobilenet_v3_small(weights=weights, **extra)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_model(train_cfg: Mapping[str, Any], num_classes: int) -> nn.Module:
    """Tạo model theo `library` và `backbone` trong cấu hình, lớp cuối có `num_classes` đầu ra."""
    library = str(train_cfg["library"])
    backbone = str(train_cfg["backbone"])
    pretrained = bool(train_cfg["pretrained"])
    dropout = train_cfg.get("dropout")
    if library == "timm":
        return _build_timm(backbone, int(num_classes), pretrained, dropout)
    if library == "torchvision":
        return _build_torchvision(backbone, int(num_classes), pretrained, dropout)
    raise ValueError(f"library chưa hỗ trợ: {library}.")

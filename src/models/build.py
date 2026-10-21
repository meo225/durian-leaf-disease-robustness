"""Tạo model phân loại từ tên trong timm."""

from __future__ import annotations

import torch
from torch import nn


def build_model(timm_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Tạo model pretrained và thay lớp phân loại cuối.

    Args:
        timm_name: Tên model trong timm, ví dụ ``resnet50``.
        num_classes: Số lớp đầu ra.
        pretrained: Tải trọng số ImageNet khi True.
    """
    import timm

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=int(num_classes),
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"timm không trả về nn.Module cho '{timm_name}'.")
    return model

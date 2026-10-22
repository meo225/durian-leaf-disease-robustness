"""Kích thước và tốc độ suy luận của một model."""

from __future__ import annotations

import time

import torch
from torch import nn


def model_size(model: nn.Module) -> dict[str, float]:
    """Số tham số và dung lượng trọng số fp32 tính bằng MB."""
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    return {"parameters": parameters, "size_mb": round(parameters * 4 / 1024**2, 2)}


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    """Đo thời gian một lượt suy luận fp32 trên tensor ngẫu nhiên.

    Trả về mili giây mỗi lượt, mili giây mỗi ảnh và số ảnh mỗi giây. Có đồng bộ GPU trước khi đọc giờ.
    """
    model.eval()
    batch = torch.randn(batch_size, 3, image_size, image_size, device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model(batch)
    sync()
    started = time.perf_counter()
    for _ in range(runs):
        model(batch)
    sync()
    per_batch = (time.perf_counter() - started) / runs
    return {
        "batch_size": batch_size,
        "ms_per_batch": round(per_batch * 1000, 3),
        "ms_per_image": round(per_batch * 1000 / batch_size, 3),
        "images_per_second": round(batch_size / per_batch, 1),
    }

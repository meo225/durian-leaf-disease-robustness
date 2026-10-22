"""Suy giảm ảnh có kiểm soát."""

from src.corruption.basic import CORRUPTIONS, apply_corruption
from src.corruption.benchmark import build_loader, corrupt_tensor, example_rows, load_conditions, load_seed

__all__ = [
    "CORRUPTIONS",
    "apply_corruption",
    "build_loader",
    "corrupt_tensor",
    "example_rows",
    "load_conditions",
    "load_seed",
]

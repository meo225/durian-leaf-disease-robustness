"""Suy giảm ảnh có kiểm soát."""

from src.corruption.basic import CORRUPTIONS, RANDOM_TYPES, apply_corruption
from src.corruption.unseen import UNSEEN_CORRUPTIONS, apply_unseen
from src.corruption.benchmark import build_loader, corrupt_tensor, example_rows, load_conditions, load_seed

__all__ = [
    "CORRUPTIONS",
    "RANDOM_TYPES",
    "UNSEEN_CORRUPTIONS",
    "apply_corruption",
    "apply_unseen",
    "build_loader",
    "corrupt_tensor",
    "example_rows",
    "load_conditions",
    "load_seed",
]

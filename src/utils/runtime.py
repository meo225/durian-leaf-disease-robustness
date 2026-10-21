"""Seed, thiết bị tính toán và thông tin môi trường của một lần chạy."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Cố định seed cho random, numpy và torch, bật chế độ cuDNN xác định."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Chọn GPU nếu có, ngược lại dùng CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> dict[str, str | None]:
    """Mô tả thiết bị và phiên bản thư viện để lưu cùng kết quả."""
    info: dict[str, str | None] = {"device": device.type, "torch": torch.__version__}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def git_commit(root: str | Path) -> str | None:
    """Trả về mã commit rút gọn của repo, hoặc None nếu không đọc được."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

"""Pipeline xử lý ảnh thích ứng: ước lượng chất lượng, chọn hành động, xử lý, rồi đưa vào reference model đóng băng (task 5.2).

Bốn thành phần tách rời để bật hoặc tắt từng phần khi ablation: bộ ước lượng chất lượng (`features.py`), bộ chọn (`rules.py`),
kỹ thuật xử lý ảnh (`src/processing`) và reference model. Cùng một lớp dùng cho đánh giá và cho inference, nên kết quả thực nghiệm
và demo đi qua đúng một đường code.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn

from src.processing import illumination, restoration
from src.selector.features import extract_features
from src.selector.rules import DEFAULT_ACTION, RuleSelector

ROOT = Path(__file__).resolve().parents[2]
# Mỗi nhóm kỹ thuật nằm trong một module riêng, tên kỹ thuật không trùng nhau giữa các module.
PROCESSING_MODULES = (illumination, restoration)


@dataclass(frozen=True)
class Decision:
    """Quyết định cho một ảnh: hành động đã chọn và đặc trưng chất lượng dẫn tới quyết định đó."""

    action: str
    features: Mapping[str, float]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_selector_config(path: str | Path = "configs/selector.yaml") -> dict[str, Any]:
    """Đọc cấu hình bộ chọn và kiểm tra các hành động hợp lệ."""
    config = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    actions = config["actions"]
    if DEFAULT_ACTION not in actions or actions[DEFAULT_ACTION] is not None:
        raise ValueError(f"Cần hành động `{DEFAULT_ACTION}: null` (giữ nguyên ảnh).")
    for name, spec in actions.items():
        if spec is None:
            continue
        if not any(spec["technique"] in module.TECHNIQUES for module in PROCESSING_MODULES):
            raise ValueError(f"Hành động {name}: kỹ thuật chưa hỗ trợ: {spec['technique']}.")
    for rule in config["rules"]:
        missing = [name for name in rule["actions"] if name not in actions]
        if missing:
            raise ValueError(f"Luật {rule['name']} dùng hành động chưa khai báo: {missing}.")
    return config


def apply_action(image: torch.Tensor, spec: Mapping[str, Any] | None) -> torch.Tensor:
    """Áp một hành động lên ảnh 0 đến 1. `None` là giữ nguyên."""
    if spec is None:
        return image
    for module in PROCESSING_MODULES:
        if spec["technique"] in module.TECHNIQUES:
            return module.apply_processing(image, {"technique": spec["technique"], "params": dict(spec["params"])})
    raise ValueError(f"Kỹ thuật chưa hỗ trợ: {spec['technique']}.")


def freeze(model: nn.Module) -> nn.Module:
    """Đưa model về chế độ eval và tắt gradient. Reference model không được đổi khi dùng trong pipeline."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_calibrated(path: str | Path, require_locked: bool = False) -> RuleSelector:
    """Đọc bộ chọn đã hiệu chỉnh (reports/selector/selector_calibrated.json).

    Đánh giá trên tập test chỉ được dùng bộ chọn đã khóa, nên truyền `require_locked=True` ở đó.
    """
    record = json.loads(resolve(path).read_text(encoding="utf-8"))
    if require_locked and not record.get("locked", False):
        raise PermissionError("Bộ chọn chưa khóa (locked: false). Không dùng để đánh giá trên tập test.")
    return RuleSelector.from_dict(record["selector"])


class AdaptivePipeline:
    """Ảnh vào, quyết định và ảnh đã xử lý ra; có reference model thì ra thêm xác suất từng lớp."""

    def __init__(
        self,
        selector: RuleSelector,
        actions: Mapping[str, Mapping[str, Any] | None],
        reference: nn.Module | None = None,
        pipeline_cfg: Mapping[str, Any] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        unknown = selector.actions_used() - set(actions)
        if unknown:
            raise ValueError(f"Bộ chọn dùng hành động chưa khai báo: {sorted(unknown)}.")
        self.selector = selector
        self.actions = dict(actions)
        self.reference = None if reference is None else freeze(reference)
        self.pipeline_cfg = pipeline_cfg
        self.device = torch.device(device)
        if self.reference is not None:
            if pipeline_cfg is None:
                raise ValueError("Cần pipeline_cfg (mean, std) khi có reference model.")
            self.reference.to(self.device)

    def decide(self, image: torch.Tensor) -> Decision:
        """Đặc trưng và hành động cho một ảnh 0 đến 1, chưa xử lý."""
        features = extract_features(image)
        return Decision(action=self.selector.select(features), features=features)

    def process(self, image: torch.Tensor) -> tuple[torch.Tensor, Decision]:
        """Ảnh đã xử lý theo hành động bộ chọn quyết định, kèm quyết định."""
        decision = self.decide(image)
        return apply_action(image, self.actions[decision.action]), decision

    def process_batch(self, images: torch.Tensor, workers: int = 1) -> tuple[torch.Tensor, list[Decision]]:
        """Xử lý lô ảnh uint8 N×3×H×W. Trả về ảnh 0 đến 1 và danh sách quyết định theo đúng thứ tự."""
        if images.dtype != torch.uint8:
            raise TypeError("process_batch nhận ảnh uint8.")
        sources = [image.float() / 255.0 for image in images]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self.process, sources))
        else:
            results = [self.process(image) for image in sources]
        return torch.stack([item[0] for item in results]), [item[1] for item in results]

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor, batch_size: int = 64, workers: int = 1) -> tuple[np.ndarray, list[Decision]]:
        """Xác suất softmax của reference model sau khi bộ chọn xử lý ảnh, và quyết định của từng ảnh."""
        if self.reference is None:
            raise RuntimeError("Pipeline không có reference model.")
        mean = torch.tensor(self.pipeline_cfg["mean"], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.pipeline_cfg["std"], device=self.device).view(1, 3, 1, 1)
        outputs, decisions = [], []
        for start in range(0, len(images), batch_size):
            processed, batch_decisions = self.process_batch(images[start : start + batch_size], workers)
            logits = self.reference((processed.to(self.device) - mean) / std)
            outputs.append(torch.softmax(logits.float(), dim=1).cpu())
            decisions.extend(batch_decisions)
        return torch.cat(outputs).numpy(), decisions

    @classmethod
    def from_files(
        cls,
        config_path: str | Path = "configs/selector.yaml",
        calibrated_path: str | Path = "reports/selector/selector_calibrated.json",
        reference: nn.Module | None = None,
        pipeline_cfg: Mapping[str, Any] | None = None,
        device: torch.device | str = "cpu",
        require_locked: bool = False,
    ) -> "AdaptivePipeline":
        """Dựng pipeline từ cấu hình hành động và bộ chọn đã hiệu chỉnh."""
        config = load_selector_config(config_path)
        return cls(load_calibrated(calibrated_path, require_locked), config["actions"], reference, pipeline_cfg, device)


__all__ = ["AdaptivePipeline", "Decision", "apply_action", "freeze", "load_calibrated", "load_selector_config"]

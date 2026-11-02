"""Bộ chọn dạng luật: đặc trưng chất lượng vào, tên hành động xử lý ảnh ra (task 5.1, 5.2).

Một luật gồm một hoặc nhiều mệnh đề nối bằng AND và một hành động. Luật được xét theo thứ tự, luật khớp đầu tiên thắng,
không luật nào khớp thì dùng hành động mặc định. Mặc định là giữ nguyên ảnh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_ACTION = "keep"
OPERATORS = (">=", "<=")


@dataclass(frozen=True)
class Clause:
    """Một mệnh đề `feature op threshold`, ví dụ `sharpness <= 0.4`."""

    feature: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in OPERATORS:
            raise ValueError(f"Toán tử chưa hỗ trợ: {self.op}. Dùng một trong {OPERATORS}.")
        if not math.isfinite(self.threshold):
            raise ValueError(f"Ngưỡng của {self.feature} không hữu hạn: {self.threshold}.")

    def holds(self, values: np.ndarray) -> np.ndarray:
        """Mặt nạ boolean của các giá trị thỏa mệnh đề."""
        return values >= self.threshold if self.op == ">=" else values <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "threshold": self.threshold}


@dataclass(frozen=True)
class Rule:
    """Nếu mọi mệnh đề đúng thì chọn `action`."""

    name: str
    clauses: tuple[Clause, ...]
    action: str

    def __post_init__(self) -> None:
        if not self.clauses:
            raise ValueError(f"Luật {self.name} cần ít nhất một mệnh đề.")

    def fires(self, table: Mapping[str, np.ndarray]) -> np.ndarray:
        """Mặt nạ boolean ảnh nào khớp luật, trên bảng đặc trưng dạng {tên: mảng}."""
        mask = None
        for clause in self.clauses:
            current = clause.holds(np.asarray(table[clause.feature]))
            mask = current if mask is None else mask & current
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "action": self.action, "clauses": [clause.to_dict() for clause in self.clauses]}


class RuleSelector:
    """Danh sách luật có thứ tự. Luật khớp đầu tiên quyết định hành động."""

    def __init__(self, rules: Sequence[Rule] = (), default: str = DEFAULT_ACTION) -> None:
        names = [rule.name for rule in rules]
        if len(set(names)) != len(names):
            raise ValueError(f"Tên luật bị trùng: {names}.")
        self.rules = tuple(rules)
        self.default = default

    def features_used(self) -> set[str]:
        return {clause.feature for rule in self.rules for clause in rule.clauses}

    def actions_used(self) -> set[str]:
        return {rule.action for rule in self.rules} | {self.default}

    def select(self, features: Mapping[str, float]) -> str:
        """Hành động cho một ảnh, từ từ điển đặc trưng."""
        row = {name: np.array([value]) for name, value in features.items()}
        return str(self.select_batch(row)[0])

    def select_batch(self, table: Mapping[str, np.ndarray]) -> np.ndarray:
        """Hành động cho từng ảnh trong bảng đặc trưng {tên: mảng cùng độ dài}. Trả về mảng tên hành động."""
        length = len(next(iter(table.values())))
        chosen = np.full(length, self.default, dtype=object)
        open_mask = np.ones(length, dtype=bool)
        for rule in self.rules:
            fired = rule.fires(table) & open_mask
            chosen[fired] = rule.action
            open_mask &= ~fired
        return chosen

    def to_dict(self) -> dict[str, Any]:
        return {"default": self.default, "rules": [rule.to_dict() for rule in self.rules]}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RuleSelector":
        rules = [
            Rule(
                name=str(item["name"]),
                action=str(item["action"]),
                clauses=tuple(Clause(str(c["feature"]), str(c["op"]), float(c["threshold"])) for c in item["clauses"]),
            )
            for item in record["rules"]
        ]
        return cls(rules, default=str(record.get("default", DEFAULT_ACTION)))

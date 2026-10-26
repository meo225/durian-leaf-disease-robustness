"""Benchmark kỹ thuật xử lý ảnh trên validation bằng reference model đóng băng (task 4.3).

Ảnh suy giảm của benchmark 4.1 được tạo một lần, lưu dạng uint8, rồi mỗi cấu hình xử lý chạy lên chính các ảnh đó. Nhờ vậy mọi
cấu hình thấy cùng ảnh, cùng nhãn, và điểm "không xử lý" khớp với hiệu chỉnh ở 4.1. Tập test không được mở.

Các bước: ảnh không xử lý, tối ưu tham số, chọn cấu hình, đối chứng không phù hợp, chi phí trên ảnh sạch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torchvision import transforms

from src.corruption import corrupt_tensor
from src.data.dataset import DurianDataset
from src.data.splits import get_split
from src.data.transforms import ConditionalResize
from src.evaluation.metrics import classification_scores
from src.processing.illumination import apply_processing

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "val"
Processor = Callable[[torch.Tensor], torch.Tensor]


class ImageCache:
    """Ảnh validation sạch và ảnh suy giảm, lưu uint8 N×3×H×W, tạo theo yêu cầu rồi giữ lại."""

    def __init__(
        self,
        pipeline_cfg: Mapping[str, Any],
        conditions: Mapping[str, Mapping[str, Any]],
        seed: int,
        limit: int | None = None,
        split: str = SPLIT,
    ) -> None:
        root = Path(pipeline_cfg["image_root"])
        dataset = DurianDataset(
            split_df=get_split(split),
            image_root=root if root.is_absolute() else ROOT / root,
            transform=transforms.Compose([ConditionalResize(int(pipeline_cfg["image_size"])), transforms.ToTensor()]),
            classes=pipeline_cfg["classes"],
        )
        count = len(dataset) if limit is None else min(int(limit), len(dataset))
        items = [dataset[index] for index in range(count)]
        self.images = torch.stack([item[0] for item in items])
        self.labels = np.array([item[1] for item in items], dtype=np.int64)
        self.paths = [item[2] for item in items]
        self.classes = list(pipeline_cfg["classes"])
        self.conditions = dict(conditions)
        self.seed = int(seed)
        self._cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def release(self, condition: str) -> None:
        """Bỏ ảnh đã tạo của một điều kiện để giải phóng RAM. Gọi `get` lần sau sẽ tạo lại."""
        self._cache.pop(condition, None)

    def get(self, condition: str) -> torch.Tensor:
        """Ảnh uint8 của một điều kiện. `clean` là ảnh không suy giảm."""
        if condition not in self._cache:
            if condition == "clean":
                source = self.images
            else:
                spec = self.conditions[condition]
                source = torch.stack([corrupt_tensor(image, condition, spec, self.seed) for image in self.images])
            self._cache[condition] = torch.round(source * 255).to(torch.uint8)
        return self._cache[condition]


def make_processor(technique: str, params: Mapping[str, Any]) -> Processor:
    spec = {"technique": technique, "params": dict(params)}
    return lambda image: apply_processing(image, spec)


@torch.no_grad()
def predict(
    model: nn.Module,
    images: torch.Tensor,
    processor: Processor | None,
    pipeline_cfg: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Dự đoán của model trên ảnh uint8, sau khi xử lý (nếu có) và chuẩn hóa như pipeline chuẩn."""
    model.eval()
    mean = torch.tensor(pipeline_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(pipeline_cfg["std"], device=device).view(1, 3, 1, 1)
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].float() / 255.0
        if processor is not None:
            batch = torch.stack([processor(image) for image in batch])
        logits = model(((batch.to(device) - mean) / std))
        outputs.append(logits.argmax(dim=1).cpu())
    return torch.cat(outputs).numpy()


def score_row(
    cache: ImageCache,
    preds: np.ndarray,
    role: str,
    technique: str,
    config_id: str,
    condition: str,
) -> dict[str, Any]:
    """Một dòng kết quả: điểm tổng và recall từng lớp."""
    scores = classification_scores(cache.labels, preds, cache.classes)
    family, severity = ("clean", 0) if condition == "clean" else _split_condition(condition)
    row = {
        "role": role,
        "technique": technique,
        "config_id": config_id,
        "condition": condition,
        "family": family,
        "severity": severity,
        "macro_f1": scores["macro_f1"],
        "balanced_accuracy": scores["balanced_accuracy"],
        "accuracy": scores["accuracy"],
    }
    row.update({f"recall_{name}": value for name, value in scores["per_class_recall"].items()})
    return row


def _split_condition(condition: str) -> tuple[str, int]:
    family, severity = condition.rsplit("_s", 1)
    return family, int(severity)


class Runner:
    """Chạy các cấu hình lên cache và gom kết quả, dự đoán từng ảnh để 4.5 dùng lại cho kiểm định ghép cặp."""

    def __init__(
        self,
        model: nn.Module,
        cache: ImageCache,
        pipeline_cfg: Mapping[str, Any],
        device: torch.device,
        progress: Callable[[str], None] = print,
    ) -> None:
        self.model, self.cache, self.pipeline_cfg, self.device, self.progress = model, cache, pipeline_cfg, device, progress
        self.rows: list[dict[str, Any]] = []
        self.predictions: dict[str, np.ndarray] = {}
        self._done: set[tuple[str, str, str]] = set()

    def run(self, role: str, technique: str, config_id: str, params: Mapping[str, Any] | None, condition: str) -> None:
        key = (technique, config_id, condition)
        if key in self._done:
            return
        processor = None if params is None else make_processor(technique, params)
        preds = predict(self.model, self.cache.get(condition), processor, self.pipeline_cfg, self.device)
        self.rows.append(score_row(self.cache, preds, role, technique, config_id, condition))
        self.predictions["|".join(key)] = preds.astype(np.int8)
        self._done.add(key)

    def raw(self) -> None:
        """Điểm không xử lý: ảnh sạch và mọi điều kiện trong scope."""
        for condition in ["clean", *self.cache.conditions]:
            self.run("raw", "none", "none", None, condition)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def load_grid(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    return yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8"))


def scope_conditions(all_conditions: Mapping[str, Mapping[str, Any]], families: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Giữ các điều kiện thuộc scope của task, theo đúng thứ tự họ trong config."""
    return {name: dict(spec) for family in families for name, spec in all_conditions.items() if spec["family"] == family}


def run_tuning(runner: Runner, grid_cfg: Mapping[str, Any]) -> None:
    """Chạy mọi cấu hình của mỗi kỹ thuật trên các họ phù hợp với nó."""
    for technique, spec in grid_cfg["techniques"].items():
        names = [name for name, cond in runner.cache.conditions.items() if cond["family"] in spec["families"]]
        for config_id, params in spec["grid"].items():
            for condition in names:
                runner.run("tune", technique, config_id, params, condition)
            runner.progress(f"{technique:10s} {config_id:16s} đã chạy {len(names)} điều kiện")


def run_clean_cost(runner: Runner, grid_cfg: Mapping[str, Any]) -> None:
    """Mọi cấu hình trên ảnh sạch, để biết chi phí của việc xử lý khi ảnh không hỏng."""
    for technique, spec in grid_cfg["techniques"].items():
        for config_id, params in spec["grid"].items():
            runner.run("clean_cost", technique, config_id, params, "clean")
    runner.progress("Đã đo chi phí trên ảnh sạch")


def _pick(scores: Mapping[str, float], clean: Mapping[str, float], tie_margin: float, order: Sequence[str]) -> str:
    """Cấu hình có điểm cao nhất. Các cấu hình trong khoảng tie_margin thì chọn cấu hình giữ F1 ảnh sạch cao hơn, rồi theo thứ tự lưới."""
    best = max(scores.values())
    close = [config for config in order if config in scores and scores[config] >= best - tie_margin]
    return max(close, key=lambda config: (round(clean.get(config, 0.0), 9), -order.index(config)))


def select_configs(frame: pd.DataFrame, grid_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Chọn cấu hình tốt nhất cho từng họ suy giảm phù hợp, và một cấu hình chung cho cả kỹ thuật.

    Cấu hình chung là cấu hình có điểm trung bình cao nhất qua các họ phù hợp. Nó được dùng làm "kỹ thuật cố định"
    và cho các cặp không phù hợp ở bước đối chứng.
    """
    margin = float(grid_cfg["selection"]["tie_margin"])
    tuned = frame[frame["role"] == "tune"]
    clean_cost = frame[frame["role"] == "clean_cost"].set_index(["technique", "config_id"])["macro_f1"]
    selected: dict[str, Any] = {}
    for technique, spec in grid_cfg["techniques"].items():
        order = list(spec["grid"])
        sub = tuned[tuned["technique"] == technique]
        clean = {config: float(clean_cost.get((technique, config), 0.0)) for config in order}
        per_family_scores = sub.groupby(["family", "config_id"])["macro_f1"].mean()
        per_family = {}
        for family in spec["families"]:
            scores = {config: float(per_family_scores[(family, config)]) for config in order if (family, config) in per_family_scores.index}
            if scores:
                chosen = _pick(scores, clean, margin, order)
                per_family[family] = {"config_id": chosen, "params": spec["grid"][chosen], "mean_macro_f1": scores[chosen]}
        overall = sub.groupby("config_id")["macro_f1"].mean()
        scores = {config: float(overall[config]) for config in order if config in overall.index}
        general = _pick(scores, clean, margin, order)
        selected[technique] = {
            "families": list(spec["families"]),
            "per_family": per_family,
            "general": {"config_id": general, "params": spec["grid"][general], "mean_macro_f1": scores[general], "clean_macro_f1": clean[general]},
        }
    return selected


def run_controls(runner: Runner, grid_cfg: Mapping[str, Any], selected: Mapping[str, Any]) -> None:
    """Cấu hình chung của mỗi kỹ thuật trên các họ không phù hợp trong scope, để thấy kỹ thuật làm hại ở đâu."""
    for technique, spec in grid_cfg["techniques"].items():
        general = selected[technique]["general"]
        for condition, cond in runner.cache.conditions.items():
            if cond["family"] not in spec["families"]:
                runner.run("control", technique, general["config_id"], general["params"], condition)
        runner.progress(f"{technique:10s} đối chứng không phù hợp xong")


def selected_config(selected: Mapping[str, Any], technique: str, family: str) -> tuple[str, Mapping[str, Any]]:
    """Cấu hình dùng cho một họ: cấu hình tốt nhất của họ nếu phù hợp, không thì cấu hình chung."""
    entry = selected[technique]
    if family in entry["per_family"]:
        chosen = entry["per_family"][family]
    else:
        chosen = entry["general"]
    return chosen["config_id"], chosen["params"]


def build_matrix(frame: pd.DataFrame, selected: Mapping[str, Any], min_degradation: float) -> pd.DataFrame:
    """Bảng kỹ thuật × điều kiện với cấu hình đã chọn, kèm degradation, recovery và normalized recovery."""
    raw = frame[frame["role"] == "raw"].set_index("condition")
    clean_f1 = float(raw.loc["clean", "macro_f1"])
    rows = []
    for technique in selected:
        for condition, record in raw.drop(index="clean").iterrows():
            config_id, _ = selected_config(selected, technique, record["family"])
            match = frame[(frame["technique"] == technique) & (frame["config_id"] == config_id) & (frame["condition"] == condition) & (frame["role"].isin(["tune", "control"]))]
            if match.empty:
                continue
            hit = match.iloc[0]
            degradation = clean_f1 - float(record["macro_f1"])
            recovery = float(hit["macro_f1"]) - float(record["macro_f1"])
            rows.append(
                {
                    "technique": technique,
                    "config_id": config_id,
                    "condition": condition,
                    "family": record["family"],
                    "severity": int(record["severity"]),
                    "suitable": hit["role"] == "tune",
                    "raw_macro_f1": float(record["macro_f1"]),
                    "macro_f1": float(hit["macro_f1"]),
                    "degradation": degradation,
                    "recovery": recovery,
                    "normalized_recovery": recovery / degradation if degradation > min_degradation else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def clean_cost_table(frame: pd.DataFrame, selected: Mapping[str, Any]) -> pd.DataFrame:
    """F1 ảnh sạch sau xử lý với cấu hình chung của mỗi kỹ thuật, so với ảnh sạch không xử lý."""
    raw_clean = frame[(frame["role"] == "raw") & (frame["condition"] == "clean")].iloc[0]
    recall_columns = [column for column in frame.columns if column.startswith("recall_")]
    rows = []
    for technique, entry in selected.items():
        config_id = entry["general"]["config_id"]
        hit = frame[(frame["role"] == "clean_cost") & (frame["technique"] == technique) & (frame["config_id"] == config_id)].iloc[0]
        row = {"technique": technique, "config_id": config_id, "macro_f1": float(hit["macro_f1"]), "change_macro_f1": float(hit["macro_f1"] - raw_clean["macro_f1"])}
        row.update({f"d_{column}": float(hit[column] - raw_clean[column]) for column in recall_columns})
        rows.append(row)
    return pd.DataFrame(rows)


def best_per_condition(matrix: pd.DataFrame) -> pd.DataFrame:
    """Kỹ thuật tốt nhất trên mỗi điều kiện theo validation. Chỉ để quan sát, chưa phải kết luận: 4.5 đo lại trên test."""
    index = matrix.groupby("condition")["macro_f1"].idxmax()
    best = matrix.loc[index].set_index("condition").reindex(list(dict.fromkeys(matrix["condition"])))
    return best[["family", "severity", "technique", "config_id", "raw_macro_f1", "macro_f1", "recovery", "normalized_recovery"]]


def save_predictions(runner: Runner, path: str | Path) -> None:
    """Dự đoán từng ảnh của mọi lượt chạy, kèm nhãn và đường dẫn ảnh, để 4.5 kiểm định ghép cặp."""
    np.savez_compressed(path, labels=runner.cache.labels, paths=np.array(runner.cache.paths), **{key.replace("|", "__"): value for key, value in runner.predictions.items()})


def example_grid(
    cache: ImageCache,
    selected: Mapping[str, Any],
    index: int,
    severity: int,
    families: Sequence[str],
) -> list[tuple[str, list[tuple[str, Any]]]]:
    """Một ảnh qua từng họ: ảnh sạch, ảnh suy giảm, rồi kết quả của cấu hình đã chọn cho mỗi kỹ thuật.

    Mỗi dòng là một họ suy giảm, định dạng dùng được với plot_corruption_examples.
    """
    rows = []
    clean = cache.get("clean")[index].float() / 255.0
    for family in families:
        condition = f"{family}_s{severity}"
        degraded = cache.get(condition)[index].float() / 255.0
        cells: list[tuple[str, Any]] = [("sạch", clean.permute(1, 2, 0).numpy()), (condition, degraded.permute(1, 2, 0).numpy())]
        for technique in selected:
            config_id, params = selected_config(selected, technique, family)
            out = make_processor(technique, params)(degraded)
            cells.append((f"{technique}\n{config_id}", out.permute(1, 2, 0).numpy()))
        rows.append((family, cells))
    return rows

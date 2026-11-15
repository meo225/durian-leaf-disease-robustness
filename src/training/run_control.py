"""Đối chứng huấn luyện lại Swin-Tiny với bốn kiểu augmentation, cùng một giao thức (task 5.9, 5.10).

Bốn kiểu: không thêm gì (baseline), augmentation suy giảm của benchmark (4.5), TrivialAugment Wide (5.6) và TrivialAugment chỉ giữ phép hình học (5.10).
Cộng một biến thể suy giảm bỏ một nhóm họ chọn bằng MSD (5.10, leave-group-out).

Giao thức chung nằm ở configs/controls/protocol.yaml và nạp cho mọi nhánh: 30 epoch, không dừng sớm, chọn checkpoint trên validation hỗn hợp, cùng optimizer.
File nhánh chỉ khai báo tên và augmentation. Nhờ vậy khác biệt duy nhất giữa các nhánh là augmentation lúc train (và seed).
Tập test không được mở ở bước này.

Chạy từ dòng lệnh: python -m src.training.run_control configs/controls/baseline.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.corruption import build_loader, load_conditions, load_seed
from src.corruption.augment import RandomCorruption
from src.evaluation.condition_predictions import evaluate_with_predictions
from src.evaluation import seed_summary
from src.evaluation.leave_group import load_pairwise_matrix, select_leave_group
from src.models.build import build_model
from src.training import run_aug, run_generic_aug
from src.training.control_augmentation import GeometricTrivialAugment, operation_names
from src.training.loop import evaluate_loader, fit_model, resolve_device, set_seed
from src.training.run import ROOT, _load_yaml, print_run_report, resource_name, save_run_artifacts

METHODS = ("none", "degradation", "trivialaugment_wide", "trivialaugment_geometric")
CONTROLS_DIR = "configs/controls"
# Khóa mà file nhánh được phép có. Mọi khóa còn lại thuộc protocol.yaml.
ARM_KEYS = {"protocol", "arm", "model_name", "augmentation", "reuse"}
# Hyperparameter trong JSON cũ phải khớp config mới thì mới dùng lại được lần chạy cũ.
REUSE_HYPERPARAMETERS = ("epochs", "learning_rate", "weight_decay", "optimizer", "scheduler", "min_delta", "pretrained")


@dataclass(frozen=True)
class RunSpec:
    """Một lần train: tên nhánh (file configs/controls/<arm>.yaml) và seed."""

    arm: str
    seed: int

    @property
    def config(self) -> str:
        return f"{CONTROLS_DIR}/{self.arm}.yaml"


ARM_NAMES = ("baseline", "degradation", "trivialaugment", "trivialaugment_geometric", "degradation_leave_group")

# Task 5.9: ba nhánh, ba seed. Seed 42 của suy giảm và TrivialAugment dùng lại từ lần chạy cũ nếu khớp giao thức, nên cần 7 lần train.
PLAN_5_9 = tuple(RunSpec(arm, seed) for arm in ("baseline", "degradation", "trivialaugment") for seed in (42, 43, 44))
# Task 5.10: hai nhánh, mỗi nhánh một seed.
PLAN_5_10 = (RunSpec("trivialaugment_geometric", 42), RunSpec("degradation_leave_group", 42))
# Chạy thử trước khi gom các lần còn lại, để bắt lỗi cấu hình sớm.
SMOKE = RunSpec("baseline", 42)


def load_arm_config(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gộp protocol.yaml với file nhánh thành config train, và đọc config pipeline. Không mở tập test."""
    path = Path(config_path)
    path = path if path.is_absolute() else ROOT / path
    arm_cfg = _load_yaml(path)
    unknown = sorted(set(arm_cfg) - ARM_KEYS)
    if unknown:
        raise ValueError(f"{path.name} đặt khóa thuộc giao thức chung hoặc không hợp lệ: {unknown}. Sửa ở protocol.yaml nếu thật sự cần đổi cho mọi nhánh.")
    protocol_cfg = _load_yaml(ROOT / str(arm_cfg["protocol"]))
    train_cfg = {**protocol_cfg, **{key: value for key, value in arm_cfg.items() if key != "protocol"}}
    check_protocol(train_cfg)
    return train_cfg, _load_yaml(ROOT / str(train_cfg["pipeline_config"]))


def check_protocol(train_cfg: Mapping[str, Any]) -> None:
    """Dừng nếu config không đúng giao thức 5.9: dừng sớm phải tắt, chọn theo macro F1 trên validation hỗn hợp, phương pháp phải hợp lệ."""
    if train_cfg.get("early_stopping") is not False:
        raise ValueError("Giao thức 5.9 cần early_stopping: false để mọi nhánh có cùng ngân sách epoch.")
    if train_cfg.get("selection_metric") != "macro_f1":
        raise ValueError("Checkpoint chỉ chọn bằng macro_f1.")
    if "validation" not in train_cfg or "severity" not in train_cfg["validation"]:
        raise ValueError("Thiếu khối validation hỗn hợp.")
    method = train_cfg["augmentation"]["method"]
    if method not in METHODS:
        raise ValueError(f"Phương pháp augmentation không hỗ trợ: {method}. Hợp lệ: {METHODS}")
    if "leave_group" in train_cfg["augmentation"] and method != "degradation":
        raise ValueError("leave_group chỉ dùng cho augmentation suy giảm.")


def protocol_differences(configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Khóa nào ngoài tên, augmentation và reuse mà các nhánh đặt khác nhau. Rỗng nghĩa là các nhánh chỉ khác ở augmentation."""
    ignored = {"arm", "model_name", "augmentation", "reuse"}
    keys = sorted({key for cfg in configs.values() for key in cfg} - ignored)
    return [key for key in keys if len({json.dumps(cfg.get(key), sort_keys=True) for cfg in configs.values()}) > 1]


def resolve_leave_group(train_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """Nhóm họ bị bỏ, tính từ ma trận khoảng cách của task 5.7. None nếu nhánh không dùng leave_group."""
    spec = train_cfg["augmentation"].get("leave_group")
    if spec is None:
        return None
    if train_cfg["augmentation"].get("exclude_families"):
        raise ValueError("Không đặt exclude_families cùng leave_group: nhóm bị bỏ lấy từ ma trận.")
    matrix_path = Path(spec["matrix"])
    matrix = load_pairwise_matrix(matrix_path if matrix_path.is_absolute() else ROOT / matrix_path, spec.get("key", "seen_pairwise_distance"))
    group = select_leave_group(matrix, str(spec["anchor"]), int(spec["size"]))
    seen = list(dict.fromkeys(item["family"] for item in load_conditions(train_cfg["augmentation"]["corruption_config"], "train").values()))
    unknown = [family for family in group["families"] if family not in seen]
    if unknown:
        raise ValueError(f"Họ trong ma trận không có ở benchmark 4.1: {unknown}")
    return group


def build_augmentation(train_cfg: Mapping[str, Any], group: Mapping[str, Any] | None = None) -> Any:
    """Augmentation của nhánh. None với baseline."""
    spec = train_cfg["augmentation"]
    method = spec["method"]
    if method == "none":
        return None
    if method == "degradation":
        excluded = list(group["families"]) if group else list(spec.get("exclude_families") or [])
        return run_aug.build_augmentation({**train_cfg, "augmentation": {**spec, "exclude_families": excluded}})
    if method == "trivialaugment_wide":
        return run_generic_aug.build_generic_augmentation(train_cfg)
    return GeometricTrivialAugment(num_magnitude_bins=int(spec.get("num_magnitude_bins", 31)))


def build_train_loader(train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], augmentation: Any) -> DataLoader:
    """Loader train. Dùng đúng hàm dựng của task 4.5 và 5.6 để nhánh cũ và mới có cùng pipeline.

    Baseline đi qua hàm của TrivialAugment với `Identity`, tức chỉ có lật và xoay.
    """
    method = train_cfg["augmentation"]["method"]
    if method == "degradation":
        return run_aug.build_train_loader(train_cfg, pipeline_cfg, augmentation)
    return run_generic_aug.build_train_loader(train_cfg, pipeline_cfg, nn.Identity() if augmentation is None else augmentation)


def build_validation(train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], group: Mapping[str, Any] | None = None) -> run_aug.CachedImages:
    """Validation hỗn hợp để chọn checkpoint: ảnh sạch cộng mỗi họ ở mức cố định. Chung cho mọi nhánh.

    Riêng nhánh leave-group bỏ khỏi tập này các họ đã bị bỏ khỏi train. Giữ chúng thì checkpoint được chọn bằng chính họ cần coi là chưa thấy,
    làm điểm trên nhóm đó lạc quan. Đây là khác biệt có chủ ý duy nhất về quy tắc chọn, ghi trong JSON kết quả.
    """
    selection = run_generic_aug.selection_config(train_cfg)
    full = run_aug.build_mixed_validation(selection, pipeline_cfg)
    if not group:
        return full
    names = run_aug.validation_condition_names(selection)
    clean_images = len(full) // (1 + len(names))
    excluded = set(group["families"])
    keep = torch.ones(len(full), dtype=torch.bool)
    for position, name in enumerate(names, start=1):
        if name.rsplit("_s", 1)[0] in excluded:
            keep[position * clean_images : (position + 1) * clean_images] = False
    indices = keep.nonzero().squeeze(1).tolist()
    return run_aug.CachedImages(full.images[indices], [full.labels[i] for i in indices], list(pipeline_cfg["mean"]), list(pipeline_cfg["std"]))


def prepare_loaders(
    train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], group: Mapping[str, Any] | None, require_gpu: bool = True
) -> tuple[torch.device, dict[str, DataLoader], Any]:
    """Seed, thiết bị, loader train và loader validation hỗn hợp (chung cho mọi nhánh). Thiếu GPU thì dừng trước khi train."""
    set_seed(int(pipeline_cfg["seed"]))
    device = resolve_device()
    if require_gpu and device.type != "cuda":
        raise RuntimeError("Notebook train cần GPU. Trong Session options chọn Accelerator GPU, rồi chạy lại từ cell đầu.")
    augmentation = build_augmentation(train_cfg, group)
    validation = build_validation(train_cfg, pipeline_cfg, group)
    loaders = {
        "train": build_train_loader(train_cfg, pipeline_cfg, augmentation),
        "val": DataLoader(validation, batch_size=int(pipeline_cfg["batch_size"]) * 2, shuffle=False, num_workers=0),
    }
    return device, loaders, augmentation


def augmentation_record(train_cfg: Mapping[str, Any], augmentation: Any, group: Mapping[str, Any] | None) -> dict[str, Any]:
    """Mô tả augmentation để ghi vào JSON kết quả và để kiểm lần chạy cũ có dùng lại được không."""
    spec = train_cfg["augmentation"]
    method = spec["method"]
    if method == "none":
        return {"method": "none"}
    if isinstance(augmentation, RandomCorruption):
        return {
            "method": "degradation",
            "corruption_config": spec["corruption_config"],
            "probability": augmentation.probability,
            "jitter": augmentation.jitter,
            "families": augmentation.families(),
            "excluded_families": list(group["families"]) if group else list(spec.get("exclude_families") or []),
            "conditions": len(augmentation.specs),
        }
    return {
        "method": method,
        "num_magnitude_bins": int(spec.get("num_magnitude_bins", 31)),
        "operations": operation_names(augmentation),
        "repr": repr(augmentation),
    }


def amend_record(
    paths: Mapping[str, str],
    train_cfg: Mapping[str, Any],
    augmentation: Any,
    group: Mapping[str, Any] | None,
    clean_val: Mapping[str, Any],
    validation_images: int,
) -> dict[str, Any]:
    """Thêm vào JSON kết quả những gì baseline không có: nhánh, augmentation, quy tắc chọn checkpoint và điểm trên validation sạch."""
    record = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
    record["train_mode"] = f"control_{train_cfg['arm']}"
    record["arm"] = train_cfg["arm"]
    record["val_set"] = "mixed"
    record["val_selection"] = {
        "description": "Ảnh sạch cộng mỗi họ suy giảm ở mức cố định, chung cho mọi nhánh của 5.9. Các chỉ số val_* là điểm trên tập hỗn hợp này, không phải validation sạch.",
        "severity": int(train_cfg["validation"]["severity"]),
        "images": int(validation_images),
        "excluded_families": list(group["families"]) if group else [],
    }
    record["augmentation"] = augmentation_record(train_cfg, augmentation, group)
    if group:
        record["leave_group"] = dict(group)
    record["clean_val"] = {key: clean_val[key] for key in ("macro_f1", "balanced_accuracy", "accuracy", "per_class_recall", "confusion_matrix")}
    Path(paths["metrics"]).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def evaluate_seen_conditions(model: nn.Module, pipeline_cfg: Mapping[str, Any], train_cfg: Mapping[str, Any], device: torch.device, seed: int, out_path: Path) -> pd.DataFrame:
    """Đo ảnh sạch và 42 điều kiện đã thấy trên validation, ghi CSV cùng định dạng với task 4.5 và 5.6. Tập test không mở."""
    config = train_cfg["validation"]["corruption_config"]
    conditions = load_conditions(config, "val")
    result = evaluate_with_predictions(model, pipeline_cfg, "val", conditions, device, load_seed(config), "retrained", train_cfg["model_name"], progress=lambda _: None)
    result["frame"].to_csv(out_path, index=False)
    return result["frame"]


def run(config_path: str | Path, seed: int, require_gpu: bool = True) -> dict[str, str]:
    """Train một nhánh với một seed, lưu checkpoint, JSON và bảng điều kiện validation. Chỉ dùng train và validation."""
    train_cfg, pipeline_cfg = load_arm_config(config_path)
    pipeline_cfg = {**pipeline_cfg, "seed": int(seed)}
    group = resolve_leave_group(train_cfg)
    device, loaders, augmentation = prepare_loaders(train_cfg, pipeline_cfg, group, require_gpu)
    print(f"Nhánh: {train_cfg['arm']}, seed {seed}", flush=True)
    print(f"Augmentation: {augmentation_record(train_cfg, augmentation, group)}", flush=True)
    if group:
        print(f"Nhóm họ bị bỏ (MSD): {group['families']}", flush=True)
    print(f"Ảnh train: {len(loaders['train'].dataset)}, validation hỗn hợp: {len(loaders['val'].dataset)}", flush=True)
    print("Tập test: không mở", flush=True)

    model = build_model(train_cfg, num_classes=int(pipeline_cfg["num_classes"]))
    classes = list(pipeline_cfg["classes"])
    result = fit_model(model=model, train_loader=loaders["train"], val_loader=loaders["val"], class_names=classes, train_cfg=train_cfg, device=device)
    paths = save_run_artifacts(result, pipeline_cfg, train_cfg, device, ROOT / "outputs")

    clean_val = evaluate_loader(model, build_loader(pipeline_cfg, "val"), nn.CrossEntropyLoss(), device, classes)
    amend_record(paths, train_cfg, augmentation, group, clean_val, len(loaders["val"].dataset))
    stem = Path(paths["metrics"]).stem
    conditions_path = Path(paths["metrics"]).with_name(f"{stem}_val_conditions.csv")
    evaluate_seen_conditions(model, pipeline_cfg, train_cfg, device, int(seed), conditions_path)
    paths["conditions"] = str(conditions_path)
    print_run_report(result, paths, resource_name(device))
    print(f"Validation sạch (không dùng để chọn checkpoint): macro F1 {clean_val['macro_f1']:.4f}", flush=True)
    print(f"Bảng 42 điều kiện: {conditions_path}", flush=True)
    return paths


def import_previous_results(search_root: str | Path, target_dir: Path) -> list[str]:
    """Chép kết quả (JSON và bảng điều kiện) của các phiên Kaggle trước, đã thêm làm input, vào `target_dir` để `run_plan` bỏ qua phần đã xong.

    Chỉ nhận cặp JSON + CSV của đúng các nhánh trong configs/controls, và không ghi đè file đã có. Checkpoint không chép: 5.9 và 5.10 không cần chúng.
    """
    names = {load_arm_config(f"{CONTROLS_DIR}/{arm}.yaml")[0]["model_name"] for arm in ARM_NAMES}
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for record in sorted(Path(search_root).rglob("*_seed[0-9]*.json")):
        stem = record.stem
        conditions = record.with_name(f"{stem}_val_conditions.csv")
        if stem.rsplit("_seed", 1)[0] not in names or not conditions.is_file():
            continue
        for source in (record, conditions):
            destination = target_dir / source.name
            if not destination.exists():
                destination.write_bytes(source.read_bytes())
                copied.append(source.name)
    return copied


def result_paths(train_cfg: Mapping[str, Any], seed: int, results_dir: Path) -> dict[str, Path]:
    stem = f"{train_cfg['model_name']}_seed{int(seed)}"
    return {"record": results_dir / f"{stem}.json", "conditions": results_dir / f"{stem}_val_conditions.csv"}


def reusable_problems(record: Mapping[str, Any], train_cfg: Mapping[str, Any], pipeline_cfg: Mapping[str, Any], seed: int) -> list[str]:
    """Lý do JSON của lần chạy cũ KHÔNG dùng lại được cho giao thức này. Rỗng nghĩa là dùng lại được.

    Lần chạy cũ dùng lại được khi chạy đủ epoch (dừng sớm không kích hoạt nên giống hệt không dừng sớm), cùng hyperparameter, cùng quy tắc
    chọn checkpoint trên validation hỗn hợp và cùng augmentation.
    """
    problems: list[str] = []
    if int(record.get("seed", -1)) != int(seed):
        problems.append(f"seed {record.get('seed')} khác {seed}")
    if record.get("model_name") != train_cfg["model_name"] or record.get("backbone") != train_cfg["backbone"]:
        problems.append("model hoặc backbone khác")
    hyper = record.get("hyperparameters", {})
    for key in REUSE_HYPERPARAMETERS:
        if hyper.get(key) != train_cfg[key]:
            problems.append(f"hyperparameter {key}: cũ {hyper.get(key)!r}, mới {train_cfg[key]!r}")
    for key in ("batch_size", "image_size"):
        if hyper.get(key) != pipeline_cfg[key]:
            problems.append(f"{key}: cũ {hyper.get(key)!r}, mới {pipeline_cfg[key]!r}")
    if int(record.get("epochs_run", -1)) != int(train_cfg["epochs"]):
        problems.append(f"chỉ chạy {record.get('epochs_run')} epoch, giao thức cần {train_cfg['epochs']} (đã dừng sớm)")
    if record.get("val_set") != "mixed" or record.get("val_selection", {}).get("severity") != int(train_cfg["validation"]["severity"]):
        problems.append("quy tắc chọn checkpoint không phải validation hỗn hợp cùng mức")
    spec = train_cfg["augmentation"]
    old = record.get("augmentation", {})
    if spec["method"] == "degradation":
        expected = {"corruption_config": spec["corruption_config"], "probability": float(spec["probability"]), "jitter": float(spec["jitter"]), "excluded_families": list(spec.get("exclude_families") or [])}
        problems += [f"augmentation {key}: cũ {old.get(key)!r}, mới {value!r}" for key, value in expected.items() if old.get(key) != value]
    else:
        expected = {"method": spec["method"], "num_magnitude_bins": int(spec.get("num_magnitude_bins", 31))}
        problems += [f"augmentation {key}: cũ {old.get(key)!r}, mới {value!r}" for key, value in expected.items() if old.get(key) != value]
    return problems


def find_run(spec: RunSpec, outputs_results: Path) -> dict[str, Any] | None:
    """Kết quả của một RunSpec: lần train mới trong `outputs_results` nếu có, không thì lần chạy cũ trong repo nếu khớp giao thức.

    Trả về {"record", "conditions", "source"} với source là `trained` hoặc `reused`, hoặc None nếu chưa có.
    """
    train_cfg, pipeline_cfg = load_arm_config(spec.config)
    new = result_paths(train_cfg, spec.seed, outputs_results)
    if new["record"].is_file() and new["conditions"].is_file():
        return {**new, "source": "trained"}
    reuse = train_cfg.get("reuse")
    if reuse and int(reuse["seed"]) == spec.seed:
        old = {"record": ROOT / reuse["record"], "conditions": ROOT / reuse["conditions"]}
        record = json.loads(old["record"].read_text(encoding="utf-8"))
        problems = reusable_problems(record, train_cfg, {**pipeline_cfg, "seed": spec.seed}, spec.seed)
        if problems:
            raise ValueError(f"Không dùng lại được {reuse['record']}: " + "; ".join(problems) + ". Train lại nhánh này.")
        return {**old, "source": "reused"}
    return None


def run_table(plan: Sequence[RunSpec], outputs_results: Path, group_families: Sequence[str] | None = None) -> pd.DataFrame:
    """Bảng một dòng mỗi lần chạy đã có (train mới hoặc dùng lại): điểm chọn checkpoint, validation sạch và macro F1 trung bình 42 điều kiện.

    `group_families` thêm hai cột: trung bình trên các họ đó (nhóm bị bỏ ở 5.10) và trên các họ còn lại.
    `val_families` là số họ trong validation chọn checkpoint (14, hoặc ít hơn ở nhánh leave-group), vì điểm chọn của hai loại này không so trực tiếp được.
    """
    rows: list[dict[str, Any]] = []
    for spec in plan:
        found = find_run(spec, outputs_results)
        if found is None:
            continue
        record = json.loads(Path(found["record"]).read_text(encoding="utf-8"))
        frame = pd.read_csv(found["conditions"])
        families = sorted(set(frame.loc[frame["condition"] != "clean", "family"]))
        row = {
            "arm": spec.arm, "seed": spec.seed, "source": found["source"],
            "best_epoch": record["best_epoch"], "epochs_run": record["epochs_run"],
            "val_families": len(families) - len(record.get("val_selection", {}).get("excluded_families", [])),
            "mixed_val_f1": record["val_macro_f1"],
            "clean_val_f1": seed_summary.clean_f1(frame),
            "objective_42": seed_summary.condition_objective(frame),
        }
        if group_families:
            rest = [family for family in families if family not in set(group_families)]
            row["objective_group"] = seed_summary.condition_objective(frame, group_families)
            row["objective_rest"] = seed_summary.condition_objective(frame, rest)
        rows.append(row)
    return pd.DataFrame(rows)


def run_plan(plan: Sequence[RunSpec], outputs_results: Path, require_gpu: bool = True) -> list[RunSpec]:
    """Train các RunSpec chưa có kết quả (mới hoặc dùng lại được). Chạy lại sau khi phiên Kaggle đứt sẽ bỏ qua phần đã xong. Trả về danh sách vừa train."""
    trained: list[RunSpec] = []
    for spec in plan:
        found = find_run(spec, outputs_results)
        if found is not None:
            print(f"Bỏ qua {spec.arm} seed {spec.seed}: đã có ({found['source']})", flush=True)
            continue
        run(spec.config, spec.seed, require_gpu)
        trained.append(spec)
    return trained


def main(argv: Sequence[str] | None = None) -> dict[str, str]:
    parser = argparse.ArgumentParser(description="Train một nhánh đối chứng (task 5.9, 5.10).")
    parser.add_argument("config", help="configs/controls/<nhánh>.yaml")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    return run(args.config, args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])

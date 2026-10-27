import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

import src.corruption.benchmark as corruption_benchmark
import src.evaluation.recovery_map as recovery_map
from src.corruption import load_conditions
from src.evaluation.recovery_map import key_of, load_selected, run_map, save_run
from src.evaluation.recovery_report import build_report, clean_cost, family_summary, recovery_cells
from src.evaluation.recovery_stats import PairedBootstrap

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
CLASSES = list(PIPELINE["classes"])
STATS = {"bootstrap": 300, "seed": 42, "confidence": 0.95, "min_degradation": 0.02, "alpha": 0.05}
EVIDENCE = {"min_significant_levels": 2, "costly_clean_drop": 0.02}


class FakeRun:
    """Dự đoán dựng sẵn với kết quả biết trước: good phục hồi, bad làm hại, neutral không đổi."""

    def __init__(self, n=700, seed=0):
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, 5, n)
        self.paths = [f"img{i}.jpg" for i in range(n)]
        self.rows, self._predictions = [], {}
        self._n = n

    def degrade(self, accuracy, salt):
        local = np.random.default_rng(salt)
        wrong = local.random(self._n) > accuracy
        return np.where(wrong, (self.labels + 1 + local.integers(0, 4, self._n)) % 5, self.labels)

    def add(self, technique, config_id, condition, family, suitable, role, predictions):
        self._predictions[key_of(technique, config_id, condition)] = predictions
        self.rows.append({"technique": technique, "config_id": config_id, "condition": condition, "family": family, "suitable": suitable, "role": role})

    def predictions(self, technique, config_id, condition):
        return self._predictions[key_of(technique, config_id, condition)]

    def has(self, technique, config_id, condition):
        return key_of(technique, config_id, condition) in self._predictions


def make_fake_run():
    run = FakeRun()
    clean = run.degrade(0.95, 1)
    run.add("none", "none", "clean", "clean", None, "none", clean)
    run.add("good", "g", "clean", "clean", None, "clean", run.degrade(0.94, 2))
    run.add("bad", "b", "clean", "clean", None, "clean", run.degrade(0.70, 3))
    run.add("neutral", "n", "clean", "clean", None, "clean", clean.copy())
    for family in ("noise", "blur"):
        for level, accuracy in enumerate((0.85, 0.70, 0.50), start=1):
            name = f"{family}_s{level}"
            corrupted = run.degrade(accuracy, 10 + level)
            run.add("none", "none", name, family, None, "none", corrupted)
            run.add("good", "g", name, family, True, "family", run.degrade(accuracy + 0.15, 20 + level))
            run.add("bad", "b", name, family, False, "family", run.degrade(accuracy - 0.2, 30 + level))
            run.add("neutral", "n", name, family, True, "family", corrupted.copy())
    return run


def test_recovery_clean_cost_and_evidence_follow_known_effects():
    run = make_fake_run()
    boot = PairedBootstrap(run.labels, 5, STATS["bootstrap"], STATS["seed"])
    cells = recovery_cells(run, boot, STATS, CLASSES)
    assert len(cells) == 18
    good, bad, neutral = (cells[cells["technique"] == name] for name in ("good", "bad", "neutral"))
    assert good["gain_significant"].all() and (good["recovery"] > 0.08).all() and (good["recovery_low"] > 0).all()
    assert bad["harm_significant"].all() and (bad["recovery"] < -0.1).all()
    assert (neutral["recovery"] == 0).all() and (neutral["p"] == 1).all() and not neutral["gain_significant"].any()
    assert (cells["p_holm"] >= cells["p"] - 1e-12).all()
    mild = good[good["severity"] == 1].iloc[0]
    assert mild["normalized_recovery"] == pytest.approx(mild["recovery"] / mild["degradation"])
    cost = clean_cost(run, boot, cells, STATS, CLASSES)
    by_technique = cost.set_index("technique")
    assert by_technique.loc["bad", "delta"] < -0.1 and by_technique.loc["bad", "delta_high"] < -0.02 and by_technique.loc["neutral", "delta"] == 0
    assert by_technique.loc["good", "used_for_families"] == "blur,noise" and by_technique.loc["bad", "used_for_families"] == ""
    family = family_summary(cells, cost, run, boot, STATS, EVIDENCE).set_index(["technique", "family"])
    assert family.loc[("good", "noise"), "evidence"] and family.loc[("good", "blur"), "levels_gain_significant"] == 3
    assert not family.loc[("neutral", "noise"), "evidence"] and not family.loc[("bad", "noise"), "evidence"]
    assert family.loc[("bad", "noise"), "costly"] and not family.loc[("good", "noise"), "costly"]


def test_build_report_end_to_end_with_phase_a(monkeypatch, tmp_path):
    original = recovery_map.get_split
    monkeypatch.setattr(recovery_map, "get_split", lambda name: original(name).head(60))
    monkeypatch.setattr(corruption_benchmark, "get_split", lambda name: original(name).head(60))
    names = ("noise_s1", "noise_s2", "noise_s3", "blur_s1")
    conditions = {name: spec for name, spec in load_conditions("configs/corruption.yaml", "val").items() if name in names}
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 5, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5)).eval()
    selected = load_selected(["reports/processing/illumination_selected.json", "reports/processing/restoration_selected.json"])
    result = run_map(model, PIPELINE, "val", conditions, selected, torch.device("cpu"), 42, workers=2, progress=lambda _: None)
    save_run(result, tmp_path / "run.npz")

    order = ["clean", *conditions]
    probs = np.stack([np.eye(5, dtype=np.float16)[result["predictions"][key_of("none", "none", name)]] for name in order])
    np.savez(tmp_path / "reference.npz", classes=np.array(CLASSES), conditions=np.array(order), labels_test=result["labels"], paths_test=np.array(result["paths"]), probs_test=probs)
    retrained = {f"retrained__convnext_tiny_aug__{name}": result["predictions"][key_of("none", "none", name)] for name in order}
    np.savez(tmp_path / "retrained.npz", labels=result["labels"], **retrained)
    config = yaml.safe_load((ROOT / "configs/recovery_map.yaml").read_text(encoding="utf-8"))
    config.update({
        "reference_predictions": str(tmp_path / "reference.npz"),
        "retrained": {"predictions": str(tmp_path / "retrained.npz"), "prefix": "retrained__convnext_tiny_aug"},
        "statistics": {**config["statistics"], "bootstrap": 100},
        "output_dir": str(tmp_path / "out"),
    })
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    built = build_report(tmp_path / "config.yaml", tmp_path / "run.npz")
    out = tmp_path / "out"
    expected_files = [
        "recovery_map_test.csv", "clean_cost_test.csv", "recovery_family_test.csv", "strategies_test.csv", "strategies_by_condition_test.csv",
        "oracle_choices_test.csv", "retrained_vs_oracle_test.csv", "recovery_heatmap.png", "normalized_recovery_heatmap.png", "strategies_by_family.png", "recovery_summary.json",
    ]
    for name in expected_files:
        assert (out / name).exists(), name
    summary = json.loads((out / "recovery_summary.json").read_text(encoding="utf-8"))
    assert summary["same_samples_as_task_4_2"] and summary["mismatch_vs_task_4_2"]["total"] == 0
    assert len(built["cells"]) == len(conditions) * len(selected) + sum(1 for row in result["rows"] if row["role"] == "general")
    strategies = built["strategies"].set_index("strategy")
    assert {"none", "oracle_family_val", "oracle_condition_test", "retrained", "fixed_gamma"} <= set(strategies.index)
    assert strategies.loc["none", "delta_mean_42"] == 0 and strategies.loc["retrained", "delta_mean_42"] == 0
    assert strategies.loc["oracle_condition_test", "f1_mean_42"] >= strategies.loc["none", "f1_mean_42"]

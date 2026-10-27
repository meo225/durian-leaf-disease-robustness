import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

import src.corruption.benchmark as corruption_benchmark
import src.evaluation.recovery_map as recovery_map
from src.corruption import build_loader, load_conditions
from src.evaluation.condition_predictions import predict_probabilities as loader_probabilities
from src.evaluation.recovery_map import clean_jobs, jobs_for_family, key_of, load_selected, run_map, save_run, task_scopes

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
SELECTED = load_selected(["reports/processing/illumination_selected.json", "reports/processing/restoration_selected.json"])
SCOPES = task_scopes(["configs/illumination.yaml", "configs/restoration.yaml"])
TECHNIQUES = {"gamma", "gamma_auto", "clahe", "retinex", "gray_world", "gaussian", "median", "bilateral", "nlm", "unsharp"}


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 5, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5)).eval()


def test_selected_files_merge_and_config_choice_per_family():
    assert set(SELECTED) == TECHNIQUES
    noise_jobs = {job["technique"]: job for job in jobs_for_family(SELECTED, "noise") if job["role"] == "family"}
    assert set(noise_jobs) == TECHNIQUES
    assert noise_jobs["bilateral"]["suitable"] and noise_jobs["bilateral"]["config_id"] == SELECTED["bilateral"]["per_family"]["noise"]["config_id"]
    assert not noise_jobs["unsharp"]["suitable"] and noise_jobs["unsharp"]["config_id"] == SELECTED["unsharp"]["general"]["config_id"]
    with pytest.raises(ValueError):
        load_selected(["reports/processing/restoration_selected.json", "reports/processing/restoration_selected.json"])


def test_general_config_is_added_only_when_it_differs_from_the_family_config():
    for family in ("noise", "sensor_noise", "blur"):
        by_technique = {}
        for job in jobs_for_family(SELECTED, family):
            by_technique.setdefault(job["technique"], []).append(job)
        for technique, jobs in by_technique.items():
            family_config = next(job for job in jobs if job["role"] == "family")["config_id"]
            general = SELECTED[technique]["general"]["config_id"]
            assert len(jobs) == (1 if family_config == general else 2)
            assert (general in {job["config_id"] for job in jobs})


def test_task_scope_matches_the_families_of_each_task_config():
    restoration, illumination = {"gaussian", "median", "bilateral", "nlm", "unsharp"}, {"gamma", "gamma_auto", "clahe", "retinex", "gray_world"}
    assert {job["technique"] for job in jobs_for_family(SELECTED, "noise", SCOPES)} == restoration
    assert {job["technique"] for job in jobs_for_family(SELECTED, "jpeg", SCOPES)} == restoration
    assert {job["technique"] for job in jobs_for_family(SELECTED, "glare", SCOPES)} == illumination
    assert {job["technique"] for job in jobs_for_family(SELECTED, "lowlight", SCOPES)} == TECHNIQUES


def test_clean_jobs_are_unique_and_include_general_configs():
    jobs = clean_jobs(SELECTED)
    ids = [(job["technique"], job["config_id"]) for job in jobs]
    assert len(ids) == len(set(ids))
    for technique, entry in SELECTED.items():
        assert (technique, entry["general"]["config_id"]) in ids


def test_run_map_keys_and_reproduces_the_benchmark_41_path(monkeypatch, tmp_path):
    original = recovery_map.get_split
    monkeypatch.setattr(recovery_map, "get_split", lambda name: original(name).head(6))  # split val: kiểm tra cơ chế, không mở test
    monkeypatch.setattr(corruption_benchmark, "get_split", lambda name: original(name).head(6))
    conditions = {name: spec for name, spec in load_conditions("configs/corruption.yaml", "val").items() if name in ("noise_s2", "blur_s1")}
    model = _model()
    result = run_map(model, PIPELINE, "val", conditions, SELECTED, torch.device("cpu"), 42, scopes=None, workers=2, progress=lambda _: None)
    expected = (1 + len(clean_jobs(SELECTED))) + sum(1 + len(jobs_for_family(SELECTED, spec["family"])) for spec in conditions.values())
    assert len(result["predictions"]) == expected == len(result["rows"])
    assert result["labels"].shape == (6,) and all(value.shape == (6,) for value in result["predictions"].values())
    for name, spec in conditions.items():
        loader = build_loader(PIPELINE, "val", name, spec, 42)
        labels, probabilities = loader_probabilities(model, loader, torch.device("cpu"))
        assert np.array_equal(labels, result["labels"])
        assert np.array_equal(result["predictions"][key_of("none", "none", name)], probabilities.argmax(axis=1))
        assert np.allclose(result["probabilities"][key_of("none", "none", name)].astype(np.float32), probabilities, atol=1e-2)
    save_run(result, tmp_path / "run.npz")
    with np.load(tmp_path / "run.npz") as saved:
        assert key_of("none", "none", "clean") + "__probs" in saved.files
        rows = json.loads(str(saved["rows_json"]))
        assert len(rows) == expected and {"technique", "config_id", "condition", "family", "suitable", "role"} <= set(rows[0])

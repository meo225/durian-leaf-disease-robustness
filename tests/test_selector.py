import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from selector_helpers import scene, tiny_model, uint8_images

from src.corruption.basic import apply_corruption
from src.selector.features import FEATURE_GROUPS, FEATURES, drop_groups, extract_batch, extract_features, group_of
from src.selector.pipeline import AdaptivePipeline, apply_action, freeze, load_calibrated, load_selector_config
from src.selector.rules import Clause, Rule, RuleSelector

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_selector_config("configs/selector.yaml")
ACTIONS = ["keep", "unsharp_strong", "unsharp_mild", "bilateral_wide", "bilateral_narrow"]




def _mean(function, count: int = 8) -> dict[str, float]:
    rows = [extract_features(torch.round(function(scene(seed)) * 255) / 255) for seed in range(count)]
    return {name: float(np.mean([row[name] for row in rows])) for name in FEATURES}


# --- đặc trưng ---------------------------------------------------------------------------------------------------------------------------


def test_feature_groups_cover_every_feature_once():
    flat = [name for names in FEATURE_GROUPS.values() for name in names]
    assert tuple(flat) == FEATURES and len(set(flat)) == len(flat)
    assert group_of("sharpness") == "sharpness" and group_of("mscn_var") == "naturalness"
    with pytest.raises(KeyError):
        group_of("không_có")


def test_features_respond_in_the_expected_direction():
    clean = _mean(lambda image: image)
    blurred = _mean(lambda image: apply_corruption(image, {"type": "blur", "value": 2.0}))
    noisy = _mean(lambda image: apply_corruption(image, {"type": "noise", "value": 0.06}, torch.Generator().manual_seed(0)))
    dark = _mean(lambda image: apply_corruption(image, {"type": "brightness", "value": 0.3}))
    assert blurred["sharpness"] < 0.3 * clean["sharpness"]
    assert blurred["mscn_var"] < clean["mscn_var"]
    assert noisy["noise_sigma"] > 10 * clean["noise_sigma"] and noisy["noise_ratio"] > clean["noise_ratio"]
    assert noisy["mscn_var"] > clean["mscn_var"]
    assert dark["luma_mean"] < 0.4 * clean["luma_mean"] and dark["dark_fraction"] > clean["dark_fraction"]
    # Độ nét đã chuẩn hóa theo độ tương phản nên ảnh tối đi không bị nhận nhầm là mờ.
    assert dark["sharpness"] == pytest.approx(clean["sharpness"], rel=0.3)


def test_noise_sigma_recovers_the_luma_standard_deviation_of_white_noise():
    flat = torch.full((3, 224, 224), 0.5)
    noisy = flat + torch.randn(3, 224, 224, generator=torch.Generator().manual_seed(1)) * 0.05
    # Nhiễu độc lập ở ba kênh nên độ lệch chuẩn của độ sáng là 0,05 nhân căn tổng bình phương hệ số độ sáng.
    expected = 0.05 * float(np.sqrt((np.array([0.299, 0.587, 0.114]) ** 2).sum()))
    assert extract_features(noisy)["noise_sigma"] == pytest.approx(expected, rel=0.1)


def test_color_cast_sign_and_validation():
    warm = torch.stack([torch.full((16, 16), 0.7), torch.full((16, 16), 0.5), torch.full((16, 16), 0.3)])
    assert extract_features(warm)["cast"] > 0 > extract_features(warm.flip(0))["cast"]
    with pytest.raises(ValueError):
        extract_features(torch.rand(1, 32, 32))
    with pytest.raises(ValueError):
        extract_features(torch.rand(3, 4, 4))


def test_batch_matches_single_images_and_threads_do_not_change_results():
    images = uint8_images(4)
    batch = extract_batch(images)
    assert batch.shape == (4, len(FEATURES)) and batch.dtype == np.float32
    single = np.array([[extract_features(image.float() / 255.0)[name] for name in FEATURES] for image in images], dtype=np.float32)
    np.testing.assert_allclose(batch, single, rtol=1e-6)
    np.testing.assert_array_equal(batch, extract_batch(images, workers=3))
    with pytest.raises(TypeError):
        extract_batch(images.float() / 255.0)


def test_drop_groups_removes_candidates_of_the_dropped_group():
    candidates = [{"feature": "noise_sigma", "op": ">="}, {"feature": "mscn_var", "op": ">="}]
    assert [c["feature"] for c in drop_groups(candidates, ["noise"])] == ["mscn_var"]
    assert drop_groups(candidates, ["noise", "naturalness"]) == []
    with pytest.raises(KeyError):
        drop_groups(candidates, ["không_có"])


# --- luật ----------------------------------------------------------------------------------------------------------------------------------


def _selector() -> RuleSelector:
    return RuleSelector([
        Rule("noise", (Clause("noise_sigma", ">=", 0.03),), "bilateral_wide"),
        Rule("blur", (Clause("sharpness", "<=", 0.02), Clause("noise_sigma", "<=", 0.01)), "unsharp_strong"),
    ])


def test_first_matching_rule_wins_and_default_is_keep():
    table = {"noise_sigma": np.array([0.05, 0.001, 0.001, 0.02]), "sharpness": np.array([0.01, 0.01, 0.5, 0.01])}
    assert list(_selector().select_batch(table)) == ["bilateral_wide", "unsharp_strong", "keep", "keep"]
    assert _selector().select({"noise_sigma": 0.05, "sharpness": 0.5}) == "bilateral_wide"
    assert RuleSelector().select({"sharpness": 0.0}) == "keep"


def test_selector_round_trips_through_dict_and_rejects_bad_rules():
    selector = _selector()
    restored = RuleSelector.from_dict(json.loads(json.dumps(selector.to_dict())))
    assert restored.to_dict() == selector.to_dict()
    assert selector.features_used() == {"noise_sigma", "sharpness"}
    assert selector.actions_used() == {"keep", "bilateral_wide", "unsharp_strong"}
    with pytest.raises(ValueError):
        Clause("sharpness", "<", 0.1)
    with pytest.raises(ValueError):
        Clause("sharpness", "<=", float("nan"))
    with pytest.raises(ValueError):
        Rule("rỗng", (), "keep")
    with pytest.raises(ValueError):
        RuleSelector([Rule("a", (Clause("x", ">=", 0.0),), "keep"), Rule("a", (Clause("x", ">=", 1.0),), "keep")])


# --- cấu hình ------------------------------------------------------------------------------------------------------------------------------


def test_actions_are_the_locked_configs_with_evidence_from_the_recovery_map():
    selected = json.loads((ROOT / "reports/processing/restoration_selected.json").read_text(encoding="utf-8"))["selected"]
    summary = json.loads((ROOT / "reports/recovery_map/recovery_summary.json").read_text(encoding="utf-8"))
    evidence = {(row["technique"], row["config_id"], row["family"]) for row in summary["evidence"]}
    for name, spec in CONFIG["actions"].items():
        if spec is None:
            continue
        locked = {chosen["config_id"]: chosen["params"] for entry in [selected[spec["technique"]]] for chosen in [*entry["per_family"].values(), entry["general"]]}
        assert locked[spec["config_id"]] == spec["params"], name
        assert spec["evidence"], name
        for family in spec["evidence"]:
            assert (spec["technique"], spec["config_id"], family) in evidence, f"{name}: {family} chưa có bằng chứng ở task 4.5"


def test_rules_only_use_known_features_and_declared_actions():
    for rule in CONFIG["rules"]:
        assert set(rule["actions"]) <= set(CONFIG["actions"]) - {"keep"}
        for clause in [*rule["candidates"], *rule.get("guard", [])]:
            assert clause["feature"] in FEATURES and clause["op"] in (">=", "<=")
    assert list(CONFIG["actions"]) == ACTIONS


def test_loader_rejects_a_config_without_keep(tmp_path):
    broken = {**CONFIG, "actions": {key: value for key, value in CONFIG["actions"].items() if key != "keep"}}
    path = tmp_path / "selector.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(ValueError):
        load_selector_config(path)


# --- pipeline ------------------------------------------------------------------------------------------------------------------------------




def test_pipeline_applies_only_the_chosen_action():
    pipeline = AdaptivePipeline(_selector(), CONFIG["actions"])
    clean = torch.round(scene(0) * 255) / 255
    blurred = torch.round(apply_corruption(scene(0), {"type": "blur", "value": 2.0}) * 255) / 255
    kept, decision = pipeline.process(clean)
    assert decision.action == "keep" and torch.equal(kept, clean)
    sharpened, decision = pipeline.process(blurred)
    assert decision.action == "unsharp_strong"
    assert torch.equal(sharpened, apply_action(blurred, CONFIG["actions"]["unsharp_strong"]))
    assert set(decision.features) == set(FEATURES)


def test_process_batch_keeps_order_and_matches_single_calls():
    pipeline = AdaptivePipeline(_selector(), CONFIG["actions"])
    images = uint8_images(3)
    images[1] = torch.round(apply_corruption(scene(1), {"type": "blur", "value": 2.0}) * 255).to(torch.uint8)
    batch, decisions = pipeline.process_batch(images, workers=2)
    assert [d.action for d in decisions] == ["keep", "unsharp_strong", "keep"]
    for index in range(3):
        assert torch.equal(batch[index], pipeline.process(images[index].float() / 255.0)[0])


def test_reference_is_frozen_and_probabilities_are_valid():
    model = tiny_model()
    pipeline = AdaptivePipeline(_selector(), CONFIG["actions"], model, {"mean": [0.5] * 3, "std": [0.25] * 3})
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    before = [p.clone() for p in model.parameters()]
    images = uint8_images(5)
    probabilities, decisions = pipeline.predict_proba(images, batch_size=2)
    assert probabilities.shape == (5, 5) and len(decisions) == 5
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    with pytest.raises(RuntimeError):
        AdaptivePipeline(_selector(), CONFIG["actions"]).predict_proba(images)
    with pytest.raises(ValueError):
        AdaptivePipeline(_selector(), CONFIG["actions"], tiny_model())
    with pytest.raises(ValueError):
        AdaptivePipeline(RuleSelector([Rule("x", (Clause("sharpness", "<=", 1.0),), "không_có")]), CONFIG["actions"])
    assert freeze(tiny_model()).training is False


def test_locked_selector_is_required_for_final_evaluation(tmp_path):
    record = {"locked": False, "selector": _selector().to_dict()}
    path = tmp_path / "selector_calibrated.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_calibrated(path).to_dict() == _selector().to_dict()
    with pytest.raises(PermissionError):
        load_calibrated(path, require_locked=True)
    path.write_text(json.dumps({**record, "locked": True}), encoding="utf-8")
    assert load_calibrated(path, require_locked=True).to_dict() == _selector().to_dict()
    assert AdaptivePipeline.from_files("configs/selector.yaml", path, require_locked=True).selector.rules

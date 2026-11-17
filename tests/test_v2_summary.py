import json

import pytest
import yaml

from src.evaluation import v2_summary as S


def _report():
    loo = {name: {"msd": value} for name, value in zip("abcde", (0.5, 1.0, 1.5, 2.0, 3.0))}
    new = [
        {"family": "near_one", "msd": 0.8, "ci_low": 0.7, "ci_high": 0.9},
        {"family": "edge", "msd": 2.5, "ci_low": 2.2, "ci_high": 2.8},
        {"family": "far_one", "msd": 4.0, "ci_low": 3.8, "ci_high": 4.2},
    ]
    return {"seen_leave_one_out": loo, "new_families": new}


def test_thresholds_are_max_second_max_and_median():
    assert S.thresholds(_report()["seen_leave_one_out"]) == {"max": 3.0, "second_max": 2.0, "median": 1.5}


def test_thresholds_need_two_families():
    with pytest.raises(ValueError):
        S.thresholds({"a": {"msd": 1.0}})


def test_sensitivity_flags_only_labels_that_move_with_the_threshold():
    rows = {row["family"]: row for row in S.threshold_sensitivity(_report())}
    assert rows["near_one"]["label_max"] == "near" and rows["near_one"]["changed"] is False
    assert rows["far_one"]["label_max"] == "far" and rows["far_one"]["changed"] is False
    assert rows["edge"]["label_max"] == "near" and rows["edge"]["label_second_max"] == "far" and rows["edge"]["label_median"] == "far"
    assert rows["edge"]["changed"] is True


def test_difficulty_by_group_averages_per_level():
    realised = [
        {"family": "x", "level": 3, "target": 0.3, "surrogate": 0.3, "swin": 0.1},
        {"family": "y", "level": 3, "target": 0.3, "surrogate": 0.3, "swin": 0.3},
        {"family": "z", "level": 3, "target": 0.3, "surrogate": 0.3, "swin": 0.4},
    ]
    rows = S.difficulty_by_group(realised, {"x": "near", "y": "near", "z": "far"})
    near = next(row for row in rows if row["group"] == "near")
    assert near["n_families"] == 2 and near["swin_delta_f1"] == 0.2 and near["swin_abs_error"] == 0.1


def test_read_realised_uses_column_positions(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("a,b,c,d,e,f,g\nfam,2,5.0,0.129,0.131,0.130,0.096\n", encoding="utf-8")
    assert S.read_realised(path) == [{"family": "fam", "level": 2, "target": 0.129, "surrogate": 0.13, "swin": 0.096}]


def test_role_counts_per_group():
    families = {"a": {"group": "near", "role": "test"}, "b": {"group": "near", "role": "train"}, "c": {"group": "far", "role": "val"}}
    assert S.role_counts(families) == [
        {"group": "far", "train": 0, "val": 1, "test": 0},
        {"group": "near", "train": 1, "val": 0, "test": 1},
    ]


def test_write_summary_reads_the_committed_reports(tmp_path):
    report = tmp_path / "r"
    report.mkdir()
    (report / "msd.json").write_text(json.dumps(_report()), encoding="utf-8")
    (report / "realised_difficulty.csv").write_text("a,b,c,d,e,f,g\nnear_one,1,1.0,0.036,0.04,0.04,0.02\n", encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"families": {"near_one": {"group": "near", "role": "test"}}}), encoding="utf-8")
    paths = S.write_summary(report, cfg)
    assert all(path.exists() for path in paths.values())

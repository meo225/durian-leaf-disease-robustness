import pytest

from src.evaluation import check_levels, plot_level_comparison, plot_per_class_recall, plot_severity_curves
from src.evaluation.calibration_report import scores_by_family, split_name

CLASSES = ["A", "B", "C", "D", "E"]


def _record(scores_by_name, clean=0.95):
    return {
        "clean_macro_f1": clean,
        "conditions": {
            name: {"macro_f1": score, "per_class_recall": {c: min(1.0, score + 0.05 * i) for i, c in enumerate(CLASSES)}}
            for name, score in scores_by_name.items()
        },
    }


GOOD = {"good_s1": 0.9, "good_s2": 0.8, "good_s3": 0.6}


def test_split_name_and_grouping_sort_by_severity():
    assert split_name("motion_blur_s3") == ("motion_blur", 3)
    record = _record({"a_s3": 0.5, "a_s1": 0.9, "a_s2": 0.7})
    assert scores_by_family(record) == {"a": [0.9, 0.7, 0.5]}


def test_check_levels_accepts_a_well_calibrated_family():
    (row,) = check_levels(_record(GOOD))
    assert row["status"] == "OK" and row["flags"] == []
    assert row["drop_s3"] == pytest.approx(0.35)


@pytest.mark.parametrize(
    "scores, flag",
    [
        ((0.9, 0.92, 0.6), "non_monotonic"),
        ((0.5, 0.4, 0.35), "collapsed_s1"),
        ((0.94, 0.93, 0.93), "too_mild"),
        ((0.8, 0.5, 0.25), "too_harsh"),
    ],
)
def test_check_levels_flags_each_problem(scores, flag):
    record = _record({f"x_s{k}": score for k, score in enumerate(scores, start=1)})
    assert flag in check_levels(record)[0]["flags"]


def test_small_increase_within_tolerance_is_not_flagged():
    record = _record({"x_s1": 0.9, "x_s2": 0.902, "x_s3": 0.6})
    assert "non_monotonic" not in check_levels(record)[0]["flags"]


def test_plots_write_png_files(tmp_path):
    record = _record({**GOOD, "other_s1": 0.9, "other_s2": 0.7, "other_s3": 0.4})
    plot_severity_curves(record, tmp_path / "curves.png")
    plot_level_comparison({"mới": record, "cũ": _record(GOOD)}, tmp_path / "compare.png")
    plot_per_class_recall(record, tmp_path / "recall.png")
    for name in ("curves.png", "compare.png", "recall.png"):
        assert (tmp_path / name).stat().st_size > 1000

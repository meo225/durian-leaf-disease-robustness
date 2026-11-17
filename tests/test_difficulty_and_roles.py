import numpy as np
import pytest

from src.corruption.family_roles import assign_roles
from src.evaluation.difficulty import match_families, match_levels, target_levels
from src.evaluation.ta_overlap import TYPE_TO_TA_OP, overlap_row, overlap_table, ta_factors

FRACTIONS = {"train": 0.4, "val": 0.2, "test": 0.4}


def test_target_levels_are_quantiles_of_seen_drops():
    seen = {f"c{i}": i / 100 for i in range(1, 43)}
    targets = target_levels(seen, [0.2, 0.5, 0.8])
    assert targets == pytest.approx([float(np.quantile(list(seen.values()), q)) for q in (0.2, 0.5, 0.8)])
    assert targets == sorted(targets)
    with pytest.raises(ValueError):
        target_levels({}, [0.5])
    with pytest.raises(ValueError):
        target_levels(seen, [0.8, 0.2])
    with pytest.raises(ValueError):
        target_levels(seen, [0.0, 0.5])


def test_match_levels_picks_nearest_within_tolerance():
    candidates = {1.0: 0.02, 2.0: 0.10, 3.0: 0.18, 4.0: 0.31, 5.0: 0.52}
    result = match_levels(candidates, [0.10, 0.20, 0.50], tolerance=0.05)
    assert result["ok"] and result["values"] == [2.0, 3.0, 5.0]
    assert result["delta_f1"] == [0.10, 0.18, 0.52]
    assert result["errors"] == pytest.approx([0.0, -0.02, 0.02])


def test_match_levels_drops_when_no_candidate_fits():
    candidates = {1.0: 0.02, 2.0: 0.04, 3.0: 0.06}
    result = match_levels(candidates, [0.05, 0.30], tolerance=0.05)
    assert not result["ok"] and "0.300" in result["reason"] and "0.2" in result["reason"]
    assert not match_levels({}, [0.1], 0.1)["ok"]


def test_match_levels_never_reuses_a_value_and_requires_monotone_order():
    assert not match_levels({1.0: 0.10}, [0.09, 0.11], tolerance=0.05)["ok"]
    # ΔF1 giảm khi mức tăng thì không tạo được thang tăng dần
    result = match_levels({1.0: 0.30, 2.0: 0.10}, [0.10, 0.30], tolerance=0.01)
    assert not result["ok"] and "tăng" in result["reason"]


def test_match_families_splits_kept_and_dropped_deterministically():
    table = {"b": {1.0: 0.1, 2.0: 0.2, 3.0: 0.3}, "a": {1.0: 0.01}}
    kept, dropped = match_families(table, [0.1, 0.2, 0.3], tolerance=0.02)
    assert list(kept) == ["b"] and list(dropped) == ["a"]


def families(n_near, n_far, source="imagenet_c_bar"):
    return {
        **{f"cbar_n{i}": {"group": "near", "source": source} for i in range(n_near)},
        **{f"cbar_f{i}": {"group": "far", "source": source} for i in range(n_far)},
    }


def test_roles_are_deterministic_and_independent_of_declaration_order():
    fam = families(5, 6)
    first = assign_roles(fam, 42, FRACTIONS)
    assert first == assign_roles(dict(reversed(list(fam.items()))), 42, FRACTIONS)
    assert first != assign_roles(fam, 43, FRACTIONS)
    assert set(first) == set(fam) and set(first.values()) <= {"train", "val", "test"}


def test_roles_are_stratified_by_group_and_follow_fractions():
    roles = assign_roles(families(5, 5), 42, FRACTIONS)
    for prefix in ("cbar_n", "cbar_f"):
        counts = [sum(1 for n, r in roles.items() if n.startswith(prefix) and r == role) for role in ("train", "val", "test")]
        assert counts == [2, 1, 2]


def test_every_stratum_gets_a_test_family_when_possible():
    roles = assign_roles(families(2, 1), 7, FRACTIONS)
    assert any(r == "test" for n, r in roles.items() if n.startswith("cbar_n"))
    assert roles["cbar_f0"] == "test"


def test_team_written_families_are_never_test():
    fam = {f"team_{i}": {"group": "near", "source": "team"} for i in range(4)}
    fam.update({f"cbar_{i}": {"group": "near", "source": "imagenet_c_bar"} for i in range(2)})
    for seed in range(20):
        roles = assign_roles(fam, seed, FRACTIONS)
        assert all(roles[n] != "test" for n in fam if n.startswith("team_"))
        assert any(r == "test" for r in roles.values())
    assert all(r != "test" for r in assign_roles({f"t{i}": {"group": "far", "source": "team"} for i in range(5)}, 1, FRACTIONS).values())


def test_assign_roles_validates_inputs():
    with pytest.raises(ValueError):
        assign_roles(families(2, 2), 1, {"train": 0.5, "val": 0.5})
    with pytest.raises(ValueError):
        assign_roles(families(2, 2), 1, {"train": 0.5, "val": 0.2, "test": 0.2})
    with pytest.raises(ValueError):
        assign_roles({"x": {"group": "near", "source": "other"}}, 1, FRACTIONS)


def test_ta_factor_range_and_overlap_table():
    factors = ta_factors("Brightness")
    assert factors.min() == pytest.approx(0.01) and factors.max() == pytest.approx(1.99) and len(factors) == 62
    with pytest.raises(ValueError):
        ta_factors("Posterize")
    inside = overlap_row({"type": "brightness", "family": "brightness_bright", "severity": 3, "value": 2.0})
    # hệ số 2.0 nằm ngoài [0.01, 1.99] nhưng trong dung sai 10% của hệ số 1.99
    assert not inside["inside_ta_range"] and inside["p_ta_within_tolerance"] > 0
    mild = overlap_row({"type": "brightness", "family": "brightness_dark", "severity": 1, "value": 0.7})
    assert mild["inside_ta_range"] and mild["ta_op"] == "Brightness"
    assert 0 < mild["p_ta_within_tolerance"] < 1 / 14
    assert overlap_row({"type": "saturate", "family": "saturate", "severity": 2, "value": 2.0})["ta_op"] == "Color"
    conditions = {
        "brightness_dark_s1": {"type": "brightness", "family": "brightness_dark", "severity": 1, "value": 0.7},
        "blur_s1": {"type": "blur", "family": "blur", "severity": 1, "value": 0.5},
        "contrast_low_s2": {"type": "contrast", "family": "contrast_low", "severity": 2, "value": 0.5},
    }
    table = overlap_table(conditions)
    assert [row["family"] for row in table] == ["brightness_dark", "contrast_low"]
    assert set(TYPE_TO_TA_OP) == {"brightness", "contrast", "saturate"}

from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import get_split


def test_get_split_train_and_val_load_successfully() -> None:
    df_train = get_split("train")
    df_val = get_split("val")

    expected_cols = ["image_path", "class_name", "group_id", "split"]
    assert list(df_train.columns) == expected_cols
    assert list(df_val.columns) == expected_cols

    assert len(df_train) == 3105
    assert len(df_val) == 444

    assert (df_train["split"] == "train").all()
    assert (df_val["split"] == "val").all()


def test_get_split_test_locked_without_final_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINAL_EVAL", raising=False)
    with pytest.raises(PermissionError, match="Tập test đã bị khóa"):
        get_split("test")


def test_get_split_test_accessible_with_final_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAL_EVAL", "1")
    df_test = get_split("test")

    expected_cols = ["image_path", "class_name", "group_id", "split"]
    assert list(df_test.columns) == expected_cols
    assert len(df_test) == 888
    assert (df_test["split"] == "test").all()


def test_total_three_splits_equals_4437(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAL_EVAL", "1")
    df_train = get_split("train")
    df_val = get_split("val")
    df_test = get_split("test")

    total = len(df_train) + len(df_val) + len(df_test)
    assert total == 4437


def test_each_group_id_strictly_in_one_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAL_EVAL", "1")
    df_all = pd.concat([get_split("train"), get_split("val"), get_split("test")], ignore_index=True)

    group_counts = df_all["group_id"].value_counts()
    multi_groups = group_counts[group_counts > 1].index.tolist()

    assert len(multi_groups) == 4

    for gid in multi_groups:
        splits = df_all[df_all["group_id"] == gid]["split"].unique()
        assert len(splits) == 1, f"Nhóm group_id={gid} bị rò rỉ qua các tập: {splits}"


def test_no_duplicate_image_paths_across_all_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAL_EVAL", "1")
    df_all = pd.concat([get_split("train"), get_split("val"), get_split("test")], ignore_index=True)

    assert df_all["image_path"].is_unique
    assert len(df_all) == 4437


def test_get_split_invalid_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Tên tập không hợp lệ"):
        get_split("unknown")

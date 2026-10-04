from pathlib import Path

import numpy as np
from PIL import Image

from src.data.quality_audit import (
    assess_eligibility,
    dhash,
    hamming_uint64,
    infer_split_and_class,
    parse_source_id,
    run_audit,
)


def _eligibility_summary(**overrides: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "readable_images": 4437,
        "corrupt_rate": 0.0,
        "classes_with_at_least_100": 5,
        "images_in_cross_class_exact_duplicates": 0,
        "cross_split_near_pairs": 0,
        "multi_image_groups": 0,
        "unique_resolutions": 1,
        "exif_datetime_count": 0,
    }
    summary.update(overrides)
    return summary


def test_parse_source_id_reads_trailing_number() -> None:
    assert parse_source_id("to_label_1710") == 1710
    assert parse_source_id("leaf") is None


def test_infer_split_normalizes_validation_directory() -> None:
    root = Path("dataset")
    path = root / "validation" / "HEALTHY_LEAF" / "to_label_1.jpg"
    assert infer_split_and_class(path, root) == ("val", "HEALTHY_LEAF")


def test_dhash_is_stable_and_changes_when_the_image_changes() -> None:
    image = Image.new("RGB", (32, 32), "black")
    pixels = image.load()
    for y in range(32):
        for x in range(10):
            pixels[x, y] = (210, 40, 40)
    assert dhash(image) == dhash(image.copy())
    changed = image.copy()
    changed_pixels = changed.load()
    for y in range(32):
        for x in range(22, 32):
            changed_pixels[x, y] = (20, 180, 60)
    assert hamming_uint64(dhash(image), dhash(changed)) > 5


def test_eligibility_requires_enough_readable_labeled_images() -> None:
    assert assess_eligibility(_eligibility_summary())["eligible"] is True
    failed = assess_eligibility(_eligibility_summary(readable_images=100, classes_with_at_least_100=1))
    assert failed["eligible"] is False
    assert failed["failures"]


def test_audit_flags_corrupt_exact_duplicate_mirror_and_adjacent_ids(tmp_path: Path) -> None:
    root = tmp_path / "data"
    train = root / "train" / "ALGAL_LEAF_SPOT"
    test = root / "test" / "ALGAL_LEAF_SPOT"
    other = root / "val" / "HEALTHY_LEAF"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    other.mkdir(parents=True)

    original = Image.new("RGB", (32, 32), "black")
    pixels = original.load()
    for y in range(32):
        for x in range(8):
            pixels[x, y] = (220, 30, 30)
    original_path = train / "to_label_1.jpg"
    original.save(original_path, quality=95)
    copy_path = train / "to_label_1_copy.jpg"
    copy_path.write_bytes(original_path.read_bytes())
    original.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(test / "to_label_2.jpg", quality=95)

    healthy = Image.new("RGB", (32, 32), (20, 140, 40))
    healthy.save(other / "healthy.jpg", quality=95)
    (train / "to_label_3.jpg").write_bytes(b"not-an-image")

    output = tmp_path / "report"
    summary = run_audit(
        root,
        output,
        expected_image_count=4,
        sample_seed=406,
    )

    assert summary["discovered_images"] == 5
    assert summary["corrupt_images"] == 1
    assert summary["readable_images"] == 4
    assert summary["exact_pairs"] >= 1
    assert summary["cross_split_near_pairs"] >= 1
    assert summary["adjacent_id_pairs"] == 1
    assert summary["adjacent_id_cross_split_pairs"] == 1
    assert summary["eligible"] is False
    assert (output / "durian_ldd_data_quality_report.md").exists()
    assert (output / "tables" / "image_inventory.csv").exists()
    assert np.isfinite(summary["corrupt_rate"])

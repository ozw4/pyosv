from __future__ import annotations

from pyosv.evaluation.promotion.rows import SummaryRow


def test_summary_row_preserves_unknown_columns_and_parses_values() -> None:
    row = SummaryRow.from_mapping(
        {"variant": " candidate ", "future_column": " kept ", "finite": "1.5", "nan": "nan"}
    )

    assert row.variant == "candidate"
    assert row.values["future_column"] == "kept"
    assert row.value("finite") == 1.5
    assert row.value("nan") is None
    assert row.value("missing") is None


def test_variant_is_not_part_of_match_key() -> None:
    common = {"case_id": "boundary_plane", "pipeline": "scanner", "shape_n1": 49}

    assert (
        SummaryRow.from_mapping({**common, "variant": "a"}).key
        == SummaryRow.from_mapping({**common, "variant": "b"}).key
    )

from __future__ import annotations

from pathlib import Path

from pyosv.evaluation.reporting.summary_schema_v1 import SUMMARY_CSV_V1_FIELDS


FIXTURE = Path("tests/fixtures/synthetic_quality_refactor/17_quality_ref2_summary.csv")


def test_summary_csv_v1_schema_is_unique_and_matches_committed_header() -> None:
    assert len(SUMMARY_CSV_V1_FIELDS) == 242
    assert len(set(SUMMARY_CSV_V1_FIELDS)) == len(SUMMARY_CSV_V1_FIELDS)

    fixture_header = FIXTURE.read_bytes().splitlines(keepends=True)[0]
    assert fixture_header == (",".join(SUMMARY_CSV_V1_FIELDS) + "\r\n").encode("utf-8")

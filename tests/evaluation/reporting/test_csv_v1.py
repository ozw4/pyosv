from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

from pyosv.evaluation.reporting.csv_v1 import (
    _summary_csv_skin_fallback_v5_guardrail_row,
    write_summary_csv,
)
from pyosv.evaluation.reporting.models import Report
from pyosv.evaluation.reporting.summary_schema_v1 import SUMMARY_CSV_V1_FIELDS


FIXTURE_DIR = Path("tests/fixtures/synthetic_quality_refactor")
METRICS_FIXTURE = FIXTURE_DIR / "17_quality_ref2_metrics.json"
SUMMARY_FIXTURE = FIXTURE_DIR / "17_quality_ref2_summary.csv"


def _fixture_payload() -> dict[str, object]:
    payload = json.loads(METRICS_FIXTURE.read_text(encoding="utf-8"))
    variants = payload["config"]["variants"]
    for case in payload["cases"]:
        for pipeline in case["pipelines"].values():
            reports = pipeline["variants"]
            pipeline["variants"] = {name: reports[name] for name in variants}
    return payload


def test_serializer_reproduces_committed_17_cube_fixture_byte_for_byte(tmp_path: Path) -> None:
    output = write_summary_csv(_fixture_payload(), tmp_path)

    assert output.read_bytes() == SUMMARY_FIXTURE.read_bytes()


def test_serializer_accepts_typed_report_and_preserves_multiple_pipeline_rows(
    tmp_path: Path,
) -> None:
    output = write_summary_csv(Report.from_dict(_fixture_payload()), tmp_path)

    assert output.read_bytes() == SUMMARY_FIXTURE.read_bytes()


def test_single_variant_and_disabled_skinning_have_only_schema_columns(tmp_path: Path) -> None:
    payload = deepcopy(_fixture_payload())
    payload["config"]["input_mode"] = "oracle"
    case = payload["cases"][0]
    pipeline = case["pipelines"]["oracle"]
    variant = pipeline["variants"]["current_default"]
    variant["skinning"] = {"enabled": False}
    variant["quality"]["skin"] = None
    pipeline["variants"] = {"current_default": variant}
    pipeline["variant_comparison"] = {
        "baseline_variant": "current_default",
        "variants": {"current_default": {}},
    }
    case["pipelines"] = {"oracle": pipeline}
    payload["cases"] = [case]

    output = write_summary_csv(payload, tmp_path)
    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == list(SUMMARY_CSV_V1_FIELDS)
    assert len(rows) == 1
    assert None not in rows[0]
    assert rows[0]["skinning_enabled"] == "False"
    assert rows[0]["skin_buffered_f1_r2"] == ""


def test_fallback_v5_guardrail_row_preserves_reason_formatting() -> None:
    row = _summary_csv_skin_fallback_v5_guardrail_row(
        {
            "fallback_v5_guardrail": {
                "enabled": True,
                "passed": False,
                "reasons": ["coverage", "fragmentation"],
                "fallback_skin_count": 3,
            }
        }
    )

    assert row["skin_fallback_v5_guardrail_enabled"] is True
    assert row["skin_fallback_v5_guardrail_passed"] is False
    assert row["skin_fallback_v5_guardrail_reasons"] == "coverage,fragmentation"
    assert row["skin_fallback_v5_guardrail_fallback_skin_count"] == 3

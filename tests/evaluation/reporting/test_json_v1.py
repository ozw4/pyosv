from __future__ import annotations

import json

import pytest

from pyosv.evaluation.reporting.json_v1 import (
    LegacyReportV1Adapter,
    report_to_json,
    write_metrics_json,
)
from pyosv.evaluation.reporting.models import Report


def _payload() -> dict[str, object]:
    comparison = {"baseline_variant": "current_default", "variants": {}}
    variant = {"quality": {"score": 1.0}, "labels": ["a", "b"]}
    pipeline = {
        "variants": {"current_default": variant},
        "variant_comparison": comparison,
    }
    return {
        "format_version": 1,
        "config": {"input_mode": "both", "variants": ["current_default"]},
        "cases": [
            {
                "case_id": "case",
                "shape": [3, 4, 5],
                "truth": {"count": 2},
                "variants": {"current_default": variant},
                "pipelines": {"oracle": pipeline, "scanner": pipeline},
                "variant_comparison": {"pipelines": {"oracle": comparison, "scanner": comparison}},
                "quality": variant["quality"],
            }
        ],
    }


def test_adapter_reproduces_legacy_dict_and_both_pipeline_structure() -> None:
    payload = _payload()
    model = Report.from_dict(payload)

    assert LegacyReportV1Adapter().to_dict(model) == payload


def test_adapter_is_repeatable_and_returned_dict_does_not_mutate_model() -> None:
    model = Report.from_dict(_payload())
    adapter = LegacyReportV1Adapter()
    first = adapter.to_dict(model)
    first["config"]["input_mode"] = "oracle"

    second = adapter.to_dict(model)
    assert second["config"]["input_mode"] == "both"
    assert report_to_json(model) == report_to_json(model)


@pytest.mark.parametrize("pretty", (False, True))
def test_json_format_and_writer_contract(tmp_path, pretty: bool) -> None:
    model = Report.from_dict(_payload())
    serialized = report_to_json(model, pretty=pretty)

    assert serialized.endswith("\n")
    assert serialized == json.dumps(_payload(), sort_keys=True, indent=2 if pretty else None) + "\n"
    assert write_metrics_json(model, tmp_path, pretty=pretty).read_text() == serialized


def test_json_helpers_preserve_generic_mapping_compatibility(tmp_path) -> None:
    payload = {"a": 1, "nested": {"values": [2, None]}}
    expected = json.dumps(payload, sort_keys=True) + "\n"

    assert report_to_json(payload) == expected
    assert write_metrics_json(payload, tmp_path).read_text() == expected


def test_adapter_rejects_non_v1_model() -> None:
    model = Report.from_dict({**_payload(), "format_version": 2})
    with pytest.raises(ValueError, match="format_version=1"):
        LegacyReportV1Adapter().to_dict(model)

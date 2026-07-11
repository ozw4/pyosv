from __future__ import annotations

from types import MappingProxyType

import pytest

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


def test_report_model_types_top_level_case_variant_and_pipeline() -> None:
    report = Report.from_dict(_payload())

    assert report.format_version == 1
    assert report.cases[0].shape == (3, 4, 5)
    assert report.cases[0].pipelines["scanner"].variant_comparison.baseline_variant == (
        "current_default"
    )
    assert report.cases[0].variants["current_default"].metrics["quality"]["score"] == 1.0


def test_report_model_copies_and_recursively_freezes_metric_payloads() -> None:
    payload = _payload()
    report = Report.from_dict(payload)
    payload["config"]["input_mode"] = "oracle"  # type: ignore[index]

    assert report.config.values["input_mode"] == "both"
    assert isinstance(report.config.values, MappingProxyType)
    with pytest.raises(TypeError):
        report.cases[0].truth["count"] = 3  # type: ignore[index]

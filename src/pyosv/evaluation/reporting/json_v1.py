"""Legacy ``format_version=1`` JSON adapter for typed reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from .models import CaseReport, PipelineReport, Report, VariantComparison, thaw_report_value


class LegacyReportV1Adapter:
    """Create detached legacy dictionaries from immutable report models."""

    def to_dict(self, report: Report) -> dict[str, Any]:
        if report.format_version != 1:
            raise ValueError("LegacyReportV1Adapter only supports format_version=1")
        return {
            "format_version": 1,
            "config": thaw_report_value(report.config.values),
            "cases": [self._case_to_dict(case) for case in report.cases],
        }

    def case_to_dict(self, case: CaseReport) -> dict[str, Any]:
        """Create a detached legacy dictionary for a case-level model."""

        return self._case_to_dict(case)

    def _case_to_dict(self, case: CaseReport) -> dict[str, Any]:
        result = {
            "case_id": case.case_id,
            "shape": list(case.shape),
            "truth": thaw_report_value(case.truth),
            "variants": {name: report.to_dict() for name, report in case.variants.items()},
            "pipelines": {
                name: self._pipeline_to_dict(report) for name, report in case.pipelines.items()
            },
            "variant_comparison": thaw_report_value(case.variant_comparison),
        }
        result.update(thaw_report_value(case.aliases))
        return result

    def _pipeline_to_dict(self, pipeline: PipelineReport) -> dict[str, Any]:
        return {
            "variants": {name: report.to_dict() for name, report in pipeline.variants.items()},
            "variant_comparison": self._comparison_to_dict(pipeline.variant_comparison),
        }

    def _comparison_to_dict(self, comparison: VariantComparison) -> dict[str, Any]:
        return {
            "baseline_variant": comparison.baseline_variant,
            "variants": thaw_report_value(comparison.variants),
        }


def _legacy_payload(report: Report | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(report, Report):
        return LegacyReportV1Adapter().to_dict(report)
    return report


def report_to_json(report: Report | Mapping[str, Any], *, pretty: bool = False) -> str:
    """Serialize a report with the exact legacy v1 JSON formatting contract."""

    return json.dumps(_legacy_payload(report), indent=2 if pretty else None, sort_keys=True) + "\n"


def write_metrics_json(
    report: Report | Mapping[str, Any],
    output_dir: str | PathLike[str],
    *,
    pretty: bool = False,
) -> Path:
    """Write ``metrics.json`` using the legacy v1 adapter."""

    output_path = Path(output_dir) / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_to_json(report, pretty=pretty), encoding="utf-8")
    return output_path

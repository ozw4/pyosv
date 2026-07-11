"""Typed evaluation reports and serialization adapters."""

from .json_v1 import LegacyReportV1Adapter, report_to_json, write_metrics_json
from .models import (
    ArtifactReference,
    CaseReport,
    PipelineReport,
    Report,
    ReportConfig,
    VariantComparison,
    VariantReport,
)

__all__ = [
    "ArtifactReference",
    "CaseReport",
    "LegacyReportV1Adapter",
    "PipelineReport",
    "Report",
    "ReportConfig",
    "VariantComparison",
    "VariantReport",
    "report_to_json",
    "write_metrics_json",
]

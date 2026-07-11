"""Typed evaluation reports and serialization adapters."""

from .csv_v1 import write_summary_csv
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
from .summary_schema_v1 import SUMMARY_CSV_V1_FIELDS

__all__ = [
    "ArtifactReference",
    "CaseReport",
    "LegacyReportV1Adapter",
    "PipelineReport",
    "Report",
    "ReportConfig",
    "SUMMARY_CSV_V1_FIELDS",
    "VariantComparison",
    "VariantReport",
    "report_to_json",
    "write_metrics_json",
    "write_summary_csv",
]

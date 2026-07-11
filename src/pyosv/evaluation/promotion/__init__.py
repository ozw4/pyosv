"""Quality-report comparison and default-promotion gates."""

from .comparison import compare_reports, compare_rows, metric_delta
from .gates import add_required_coverage, build_promotion_report
from .rows import SummaryRow, read_summary_rows
from .specifications import GateSpec, SCANNER_BOUNDARY_GATE

__all__ = [
    "GateSpec",
    "SCANNER_BOUNDARY_GATE",
    "SummaryRow",
    "add_required_coverage",
    "build_promotion_report",
    "compare_reports",
    "compare_rows",
    "metric_delta",
    "read_summary_rows",
]

"""Quality-report comparison and default-promotion gates."""

from .comparison import (
    COMPARISON_PROFILES,
    VARIANT_COMPARISON_PROFILE,
    compare_reports,
    compare_rows,
    metric_delta,
)
from .gates import add_required_coverage, build_promotion_report
from .rows import SummaryRow, read_summary_rows
from .scanner_policy import (
    NORMAL_SCANNER_POLICY_ID,
    REFERENCE_SCANNER_POLICY_ID,
    SCANNER_THINNING_POLICY_PROFILE,
)
from .specifications import GateSpec, SCANNER_BOUNDARY_GATE

__all__ = [
    "GateSpec",
    "COMPARISON_PROFILES",
    "NORMAL_SCANNER_POLICY_ID",
    "REFERENCE_SCANNER_POLICY_ID",
    "SCANNER_BOUNDARY_GATE",
    "SCANNER_THINNING_POLICY_PROFILE",
    "SummaryRow",
    "VARIANT_COMPARISON_PROFILE",
    "add_required_coverage",
    "build_promotion_report",
    "compare_reports",
    "compare_rows",
    "metric_delta",
    "read_summary_rows",
]

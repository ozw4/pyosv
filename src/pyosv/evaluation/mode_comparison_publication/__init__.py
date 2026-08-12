"""Derived publication reporting for validated synthetic and F3 comparisons."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from ..f3d_mode_comparison.data import ensure_output_not_in_data_root

from .artifacts import write_publication_bundle
from .config import (
    PUBLICATION_ARTIFACT_SCHEMA_VERSION,
    PUBLICATION_COMPLETION_SCHEMA_VERSION,
    PUBLICATION_FIGURE_CONTRACT_VERSION,
    PUBLICATION_METRIC_SELECTION_VERSION,
    PUBLICATION_TABLE_CONTRACT_VERSION,
)
from .loaders import load_f3_source, load_synthetic_source
from .models import PublicationReport
from .registry import PUBLICATION_METRIC_REGISTRY, PublicationMetric
from .summary import build_tables
from .validation import validate_publication_bundle


def build_publication_report(
    synthetic_bundle: str | PathLike[str],
    f3_bundle: str | PathLike[str],
    f3_data_root: str | PathLike[str],
) -> PublicationReport:
    """Validate both sources and assemble derived tables without running stages."""

    synthetic = load_synthetic_source(synthetic_bundle)
    f3 = load_f3_source(f3_bundle, f3_data_root)
    return PublicationReport(synthetic, f3, build_tables(synthetic, f3), {})


def _assert_output_is_derived_only(report: PublicationReport, output_dir: Path) -> None:
    resolved = output_dir.resolve(strict=False)
    for source_path in (report.synthetic.path, report.f3.path):
        if resolved == source_path or resolved.is_relative_to(source_path):
            raise ValueError("publication output must not be inside a source bundle")
    ensure_output_not_in_data_root(resolved, report.f3.data_root)


def generate_publication_bundle(
    synthetic_bundle: str | PathLike[str],
    f3_bundle: str | PathLike[str],
    f3_data_root: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    pretty: bool = False,
) -> Path:
    """Build and atomically write the fixed publication bundle."""

    report = build_publication_report(synthetic_bundle, f3_bundle, f3_data_root)
    destination = Path(output_dir)
    _assert_output_is_derived_only(report, destination.resolve(strict=False))
    return write_publication_bundle(report, destination, pretty=pretty)


from .v1_bundle import generate_publication_bundle_v1  # noqa: E402


__all__ = [
    "PUBLICATION_ARTIFACT_SCHEMA_VERSION",
    "PUBLICATION_COMPLETION_SCHEMA_VERSION",
    "PUBLICATION_FIGURE_CONTRACT_VERSION",
    "PUBLICATION_METRIC_REGISTRY",
    "PUBLICATION_METRIC_SELECTION_VERSION",
    "PUBLICATION_TABLE_CONTRACT_VERSION",
    "PublicationMetric",
    "PublicationReport",
    "build_publication_report",
    "generate_publication_bundle",
    "generate_publication_bundle_v1",
    "validate_publication_bundle",
    "write_publication_bundle",
]

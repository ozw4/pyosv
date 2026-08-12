"""Derived publication reporting for validated synthetic and F3 comparisons."""

from __future__ import annotations

from os import PathLike

from .loaders import load_f3_source, load_synthetic_source
from .models import PublicationReport
from .registry import PUBLICATION_METRIC_REGISTRY, PublicationMetric
from .summary import build_tables


def build_publication_report(
    synthetic_bundle: str | PathLike[str],
    f3_bundle: str | PathLike[str],
    f3_data_root: str | PathLike[str],
) -> PublicationReport:
    """Validate both sources and assemble derived tables without running stages."""

    synthetic = load_synthetic_source(synthetic_bundle)
    f3 = load_f3_source(f3_bundle, f3_data_root)
    return PublicationReport(synthetic, f3, build_tables(synthetic, f3))


from ..publication_manifest_io import (  # noqa: E402
    validate_publication_directory as validate_publication_bundle,
)
from .v1_bundle import generate_publication_bundle_v1  # noqa: E402

generate_publication_bundle = generate_publication_bundle_v1


__all__ = [
    "PUBLICATION_METRIC_REGISTRY",
    "PublicationMetric",
    "PublicationReport",
    "build_publication_report",
    "generate_publication_bundle",
    "generate_publication_bundle_v1",
    "validate_publication_bundle",
]

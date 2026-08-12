from __future__ import annotations

from pathlib import Path
from typing import Any

from pyosv.evaluation.mode_comparison_publication import build_publication_report
from pyosv.evaluation.mode_comparison_publication.loaders import (
    load_f3_source,
    load_synthetic_source,
)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_loaders_keep_required_v1_source_data_and_preserve_sources(
    source_bundles: dict[str, Any],
) -> None:
    before = {
        "synthetic": _snapshot(source_bundles["synthetic"]),
        "f3": _snapshot(source_bundles["f3"]),
        "data": _snapshot(source_bundles["data_root"]),
    }

    synthetic = load_synthetic_source(source_bundles["synthetic"])
    f3 = load_f3_source(source_bundles["f3"], source_bundles["data_root"])

    assert len(synthetic.completion_sha256) == 64
    assert synthetic.case_order
    assert synthetic.manifest["resolved_plan"]
    assert len(f3.completion_sha256) == 64
    assert f3.dataset_identity["dataset_id"] == f3.dataset_spec.dataset_id
    assert f3.run_manifest["plan"]
    assert _snapshot(source_bundles["synthetic"]) == before["synthetic"]
    assert _snapshot(source_bundles["f3"]) == before["f3"]
    assert _snapshot(source_bundles["data_root"]) == before["data"]


def test_report_builder_uses_validated_sources_without_root_metadata(
    source_bundles: dict[str, Any],
) -> None:
    report = build_publication_report(
        source_bundles["synthetic"],
        source_bundles["f3"],
        source_bundles["data_root"],
    )

    assert set(report.tables) == {
        "publication_metrics.csv",
        "publication_contrasts.csv",
        "publication_summary.csv",
        "f3_regional_summary.csv",
        "f3_orientation_summary.csv",
        "runtime_summary.csv",
    }
    assert not hasattr(report, "manifest")

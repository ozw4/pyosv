from __future__ import annotations

import pytest

from pyosv.cells import FaultCell
from pyosv.experimental.skin_diagnostics import (
    SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT,
    add_primary_skin_diagnostics,
    primary_boundary_degraded_reasons,
    primary_skin_degraded_reasons,
)


def _skin(indices: list[tuple[int, int, int]]) -> list[FaultCell]:
    return [FaultCell(i1, i2, i3, 1.0, 0.0, 90.0) for i3, i2, i1 in indices]


def test_empty_fvt_has_no_degraded_reason() -> None:
    assert (
        primary_skin_degraded_reasons(
            fvt_positive_candidate_count=0,
            skin_count=0,
            cell_coverage_of_fvt_positive=0.0,
            largest_fraction=0.0,
            small_skin_cell_fraction=0.0,
        )
        == []
    )


def test_primary_diagnostics_preserve_threshold_and_reason_order() -> None:
    diagnostics: dict[str, object] = {}
    skins = [
        _skin([(0, 0, index)]) for index in range(SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT)
    ]
    add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=(3, 3, 20),
        fvt_positive_candidate_count=40,
        small_skin_size=10,
    )
    assert diagnostics["skin_primary_count"] == 8
    assert diagnostics["skin_primary_cell_coverage_of_fvt_positive"] == pytest.approx(0.2)
    assert diagnostics["skin_primary_degraded_reasons"] == [
        "low_fvt_positive_coverage",
        "fragmented_primary_skins",
        "high_small_skin_cell_fraction",
    ]


def test_boundary_reasons_require_generic_degradation() -> None:
    arguments = {
        "fvt_positive_candidate_count": 10,
        "cell_coverage_of_fvt_positive": 0.1,
        "fvt_positive_edge_shell_fraction": 0.5,
        "primary_edge_shell_fraction": 0.1,
        "fvt_to_scanner_target_distance_p95": 3.0,
    }
    assert primary_boundary_degraded_reasons(generic_degraded=False, **arguments) == []
    assert primary_boundary_degraded_reasons(generic_degraded=True, **arguments) == [
        "fvt_positive_edge_shell",
        "fvt_far_from_scanner_target",
        "low_primary_coverage_with_edge_local_candidates",
    ]

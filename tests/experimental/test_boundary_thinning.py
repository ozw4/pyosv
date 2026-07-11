from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from pyosv.experimental.boundary_thinning import (
    apply_boundary_edge_thin_v1,
    recenter_edge_fvt_to_target,
)
from pyosv.voting3d import OptimalSurfaceVoter


def test_recenter_is_deterministic_bounded_and_does_not_mutate_input() -> None:
    fvt = np.zeros((3, 5, 3), dtype=np.float32)
    fvt[1, 0, 1] = 2.0
    fvt[1, 2, 1] = 5.0
    original = fvt.copy()
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    target = np.zeros_like(fvt)
    target[1, 1, 1] = 1.0

    first = recenter_edge_fvt_to_target(
        fvt,
        vp,
        vt,
        target=target,
        target_source="scanner_fet",
        max_shift=1,
        edge_margin=2,
    )
    second = recenter_edge_fvt_to_target(
        fvt,
        vp,
        vt,
        target=target,
        target_source="scanner_fet",
        max_shift=1,
        edge_margin=2,
    )

    np.testing.assert_array_equal(fvt, original)
    np.testing.assert_array_equal(first.output, second.output)
    assert first.output[1, 1, 1] == 5.0
    assert np.count_nonzero(first.output) == 1
    assert first.diagnostics["fvt_recenter_collision_count"] == 1
    assert first.diagnostics["fvt_recenter_target_source"] == "scanner_fet"


def test_recenter_empty_input_returns_float32_empty_output() -> None:
    empty = np.zeros((3, 3, 3), dtype=np.float32)
    result = recenter_edge_fvt_to_target(
        empty,
        empty,
        empty,
        target=empty,
        target_source="ft_input",
        edge_margin=1,
    )

    assert result.output.dtype == np.float32
    assert np.count_nonzero(result.output) == 0
    assert result.diagnostics["fvt_recenter_candidate_count"] == 0


def test_boundary_edge_thin_preserves_non_edge_samples_and_input() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 7, 5), dtype=np.float32)
    fv[2, 0:4, 2] = 1.0
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)
    target = np.zeros_like(fv)
    target[2, 1, 2] = 1.0
    fvt = voter.thin(fv, vp, vt, mode="hybrid_v2", reference_sigma=0.0, plateau_tie_breaker=fv)
    original = fvt.copy()

    result = apply_boundary_edge_thin_v1(
        fvt,
        fv,
        vp,
        vt,
        voter=voter,
        target=target,
        target_source="scanner_fet",
        edge_margin=2,
    )

    np.testing.assert_array_equal(fvt, original)
    non_edge = ~np.pad(np.zeros((1, 3, 1), dtype=bool), 2, constant_values=True)
    np.testing.assert_array_equal(result.output[non_edge], original[non_edge])
    assert result.diagnostics["target_source"] == "scanner_fet"


def test_report_thinning_wrappers_match_experimental_results() -> None:
    path = Path(__file__).parents[2] / "examples" / "report_3d_synthetic_quality.py"
    spec = importlib.util.spec_from_file_location("thinning_wrapper_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fvt = np.zeros((3, 3, 3), dtype=np.float32)
    fvt[1, 0, 1] = 1.0
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    target = np.zeros_like(fvt)
    target[1, 1, 1] = 1.0
    kwargs = {"target": target, "target_source": "scanner_fet", "max_shift": 1, "edge_margin": 1}

    expected = recenter_edge_fvt_to_target(fvt, vp, vt, **kwargs)
    output, diagnostics = module._recenter_edge_fvt_to_target(fvt, vp, vt, **kwargs)

    np.testing.assert_array_equal(output, expected.output)
    assert diagnostics == expected.diagnostics

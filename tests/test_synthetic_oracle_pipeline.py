import math

import numpy as np
import pytest

from pyosv.synthetic3d import (
    Synthetic3DCase,
    make_boundary_plane_case,
    make_crossing_planes_case,
    make_curved_surface_case,
    make_parallel_planes_case,
    make_single_dipping_plane_case,
    make_single_vertical_plane_case,
    make_weak_noisy_plane_case,
)
from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    edge_false_positive_ratio,
    masked_orientation_error,
    skin_truth_metrics,
    surface_distance_metrics,
    top_truth_count_mask,
)
from pyosv.skinner import FaultSkinner
from pyosv.voting3d import OptimalSurfaceVoter


def _run_oracle_voting(
    case: Synthetic3DCase,
    *,
    seed_distance: int = 3,
    min_likelihood: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)

    fv, vp, vt = voter.apply_voting(
        d=seed_distance,
        fm=min_likelihood,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
    )
    fvt = voter.thin(fv, vp, vt)
    return fv, vp, vt, fvt


def _assert_oracle_arrays(
    case: Synthetic3DCase,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    fvt: np.ndarray,
) -> None:
    for values in (fv, vp, vt, fvt):
        assert values.shape == case.shape
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
    assert fv.max() > np.float32(0.0)
    assert fvt.max() > np.float32(0.0)


def _truth_quality(
    case: Synthetic3DCase,
    values: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(0.5)
    candidate_mask = top_truth_count_mask(values, truth_surface_mask)
    return {
        "buffered_overlap": buffered_surface_overlap(
            candidate_mask,
            case.truth_fault_mask,
            radius=2.0,
        ),
        "surface_distance": surface_distance_metrics(candidate_mask, truth_surface_mask),
        "orientation_error": masked_orientation_error(
            vp,
            vt,
            case.truth_strike,
            case.truth_dip,
            candidate_mask,
        ),
        "edge_false_positive": edge_false_positive_ratio(
            candidate_mask,
            truth_surface_mask,
            edge_margin=2,
            truth_buffer_radius=2.0,
        ),
    }


def _skin_truth_metrics(case, fvt: np.ndarray, vp: np.ndarray, vt: np.ndarray) -> dict:
    skins = FaultSkinner(min_likelihood=0.5, min_skin_size=1).find_skins(fvt, vp, vt)
    return skin_truth_metrics(
        skins,
        shape=case.shape,
        truth_fault_mask=case.truth_fault_mask,
        truth_surface_mask=np.abs(case.truth_distance) <= np.float32(0.5),
        truth_strike=case.truth_strike,
        truth_dip=case.truth_dip,
        buffer_radius=2.0,
    )


def _assert_skin_quality(
    metrics: dict,
    *,
    min_buffered_f1: float,
    max_candidate_to_truth_p95: float,
    max_strike_median_error: float,
    max_dip_median_error: float,
    min_largest_skin_fraction: float | None = None,
) -> None:
    topology = metrics["topology"]
    overlap = metrics["buffered_overlap_radius2"]
    distances = metrics["surface_distance"]
    orientation = metrics["orientation_error"]

    assert topology["skin_count"] >= 1
    assert topology["cell_count"] > 0
    assert overlap["buffered_f1"] >= min_buffered_f1
    assert distances["candidate_to_truth_p95"] <= max_candidate_to_truth_p95
    assert orientation["strike_median"] <= max_strike_median_error
    assert orientation["dip_median"] <= max_dip_median_error
    if min_largest_skin_fraction is not None:
        assert topology["largest_skin_fraction"] >= min_largest_skin_fraction


def test_single_vertical_plane_oracle_pipeline_smoke() -> None:
    case = make_single_vertical_plane_case(shape=(33, 33, 33))
    fv, vp, vt, fvt = _run_oracle_voting(case)
    _assert_oracle_arrays(case, fv, vp, vt, fvt)

    quality = _truth_quality(case, fvt, vp, vt)
    overlap = quality["buffered_overlap"]
    distances = quality["surface_distance"]
    orientation = quality["orientation_error"]

    assert overlap["buffered_f1"] >= 0.80
    assert distances["candidate_to_truth_p95"] <= 3.0
    assert orientation["strike_median"] <= 10.0
    assert orientation["dip_median"] <= 10.0

    _assert_skin_quality(
        _skin_truth_metrics(case, fvt, vp, vt),
        min_buffered_f1=0.70,
        max_candidate_to_truth_p95=4.0,
        max_strike_median_error=15.0,
        max_dip_median_error=15.0,
        min_largest_skin_fraction=0.60,
    )


def test_single_dipping_plane_oracle_pipeline_smoke() -> None:
    case = make_single_dipping_plane_case(shape=(21, 21, 21))
    fv, vp, vt, fvt = _run_oracle_voting(case)
    _assert_oracle_arrays(case, fv, vp, vt, fvt)

    quality = _truth_quality(case, fvt, vp, vt)
    overlap = quality["buffered_overlap"]
    distances = quality["surface_distance"]
    orientation = quality["orientation_error"]

    assert overlap["buffered_f1"] >= 0.60
    assert distances["candidate_to_truth_p95"] <= 5.0
    assert orientation["strike_median"] <= 15.0
    assert orientation["dip_median"] <= 15.0

    _assert_skin_quality(
        _skin_truth_metrics(case, fvt, vp, vt),
        min_buffered_f1=0.45,
        max_candidate_to_truth_p95=8.0,
        max_strike_median_error=25.0,
        max_dip_median_error=25.0,
        min_largest_skin_fraction=0.40,
    )


def test_curved_surface_oracle_pipeline_smoke() -> None:
    case = make_curved_surface_case(shape=(25, 25, 25))
    fv, vp, vt, fvt = _run_oracle_voting(case)
    _assert_oracle_arrays(case, fv, vp, vt, fvt)

    quality = _truth_quality(case, fvt, vp, vt)
    overlap = quality["buffered_overlap"]
    distances = quality["surface_distance"]
    orientation = quality["orientation_error"]

    assert overlap["buffered_f1"] >= 0.45
    assert distances["candidate_to_truth_p95"] <= 12.0
    assert orientation["strike_median"] <= 45.0
    assert orientation["dip_median"] <= 30.0

    _assert_skin_quality(
        _skin_truth_metrics(case, fvt, vp, vt),
        min_buffered_f1=0.30,
        max_candidate_to_truth_p95=14.0,
        max_strike_median_error=60.0,
        max_dip_median_error=45.0,
    )


@pytest.mark.parametrize(
    (
        "case_factory",
        "min_buffered_f1",
        "max_candidate_to_truth_p95",
        "max_orientation_median_error",
        "require_skin",
    ),
    (
        (make_parallel_planes_case, 0.45, 8.0, 35.0, True),
        (make_crossing_planes_case, 0.25, 12.0, 60.0, False),
        (make_weak_noisy_plane_case, 0.15, 16.0, 70.0, False),
    ),
)
def test_extended_oracle_pipeline_fvt_smoke(
    case_factory,
    min_buffered_f1: float,
    max_candidate_to_truth_p95: float,
    max_orientation_median_error: float,
    require_skin: bool,
) -> None:
    case = case_factory(shape=(17, 17, 17))
    fv, vp, vt, fvt = _run_oracle_voting(case)
    _assert_oracle_arrays(case, fv, vp, vt, fvt)

    quality = _truth_quality(case, fvt, vp, vt)
    overlap = quality["buffered_overlap"]
    distances = quality["surface_distance"]
    orientation = quality["orientation_error"]

    assert overlap["buffered_f1"] >= min_buffered_f1
    assert distances["candidate_to_truth_p95"] <= max_candidate_to_truth_p95
    assert orientation["strike_median"] <= max_orientation_median_error
    assert orientation["dip_median"] <= max_orientation_median_error

    if require_skin:
        skin_metrics = _skin_truth_metrics(case, fvt, vp, vt)
        assert skin_metrics["topology"]["skin_count"] >= 1


def test_boundary_oracle_pipeline_fvt_smoke() -> None:
    case = make_boundary_plane_case(shape=(17, 17, 17))
    assert np.any(case.truth_fault_mask[:, 0, :])
    assert not np.any(case.truth_fault_mask[:, -1, :])

    # The boundary fixture forms a flat edge ridge at the quick smoke shape;
    # denser seeds give thinning a nonzero pipeline result to evaluate.
    fv, vp, vt, fvt = _run_oracle_voting(
        case,
        seed_distance=1,
        min_likelihood=0.1,
    )
    _assert_oracle_arrays(case, fv, vp, vt, fvt)

    quality = _truth_quality(case, fvt, vp, vt)
    overlap = quality["buffered_overlap"]
    distances = quality["surface_distance"]
    edge_false_positive = quality["edge_false_positive"]

    assert overlap["buffered_f1"] >= 0.35
    assert distances["candidate_to_truth_p95"] <= 10.0
    assert math.isfinite(edge_false_positive["edge_false_positive_fraction_of_candidates"])
    assert edge_false_positive["edge_false_positive_fraction_of_candidates"] <= 0.50

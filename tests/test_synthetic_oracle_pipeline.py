import numpy as np

from pyosv.synthetic3d import (
    make_curved_surface_case,
    make_single_dipping_plane_case,
    make_single_vertical_plane_case,
)
from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    masked_orientation_error,
    surface_distance_metrics,
    top_truth_count_mask,
)
from pyosv.voting3d import OptimalSurfaceVoter


def test_single_vertical_plane_oracle_pipeline_smoke() -> None:
    case = make_single_vertical_plane_case(shape=(33, 33, 33))
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)

    fv, vp, vt = voter.apply_voting(
        d=3,
        fm=0.5,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
    )
    fvt = voter.thin(fv, vp, vt)

    for values in (fv, fvt):
        assert values.shape == case.shape
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
        assert values.max() > np.float32(0.0)

    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(0.5)
    candidate_mask = top_truth_count_mask(fvt, truth_surface_mask)

    overlap = buffered_surface_overlap(candidate_mask, case.truth_fault_mask, radius=2.0)
    distances = surface_distance_metrics(candidate_mask, case.truth_fault_mask)
    orientation = masked_orientation_error(
        vp,
        vt,
        case.truth_strike,
        case.truth_dip,
        candidate_mask,
    )

    assert overlap["buffered_f1"] >= 0.80
    assert distances["candidate_to_truth_p95"] <= 3.0
    assert orientation["strike_median"] <= 10.0
    assert orientation["dip_median"] <= 10.0


def test_single_dipping_plane_oracle_pipeline_smoke() -> None:
    case = make_single_dipping_plane_case(shape=(25, 25, 25))
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)

    fv, vp, vt = voter.apply_voting(
        d=3,
        fm=0.5,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
    )
    fvt = voter.thin(fv, vp, vt)

    for values in (fv, fvt):
        assert values.shape == case.shape
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
        assert values.max() > np.float32(0.0)

    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(0.5)
    fvt_top_truth_count = top_truth_count_mask(fvt, truth_surface_mask)

    overlap = buffered_surface_overlap(
        fvt_top_truth_count,
        case.truth_fault_mask,
        radius=2.0,
    )
    distances = surface_distance_metrics(fvt_top_truth_count, case.truth_fault_mask)
    orientation = masked_orientation_error(
        vp,
        vt,
        case.truth_strike,
        case.truth_dip,
        fvt_top_truth_count,
    )

    assert overlap["buffered_f1"] >= 0.60
    assert distances["candidate_to_truth_p95"] <= 5.0
    assert orientation["strike_median"] <= 15.0
    assert orientation["dip_median"] <= 15.0


def test_curved_surface_oracle_pipeline_smoke() -> None:
    case = make_curved_surface_case(shape=(25, 25, 25))
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)

    fv, vp, vt = voter.apply_voting(
        d=3,
        fm=0.5,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
    )
    fvt = voter.thin(fv, vp, vt)

    for values in (fv, fvt):
        assert values.shape == case.shape
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
        assert values.max() > np.float32(0.0)

    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(0.5)
    fvt_top_truth_count = top_truth_count_mask(fvt, truth_surface_mask)

    overlap = buffered_surface_overlap(
        fvt_top_truth_count,
        case.truth_fault_mask,
        radius=2.0,
    )
    distances = surface_distance_metrics(fvt_top_truth_count, case.truth_fault_mask)
    orientation = masked_orientation_error(
        vp,
        vt,
        case.truth_strike,
        case.truth_dip,
        fvt_top_truth_count,
    )

    assert overlap["buffered_f1"] >= 0.45
    assert distances["candidate_to_truth_p95"] <= 12.0
    assert orientation["strike_median"] <= 40.0
    assert orientation["dip_median"] <= 30.0

import os
from pprint import pformat

import numpy as np
import pytest

from pyosv.metrics import finite_value_report, normalized_correlation, top_percentile_overlap
from pyosv.reference import REFERENCE_DATASETS_2D
from pyosv.voting2d import OptimalPathVoter

from reference_fixtures import load_reference_2d_array


def test_f3d2d_seed_picking_smoke() -> None:
    dataset = REFERENCE_DATASETS_2D["f3d2d"]
    ft = load_reference_2d_array("f3d2d", "ft.dat")
    pt = load_reference_2d_array("f3d2d", "pt.dat")
    fm = 0.3

    seeds = OptimalPathVoter(ru=15, rv=30).pick_seeds(d=4, fm=fm, ft=ft, pt=pt)

    assert len(seeds) > 0

    n2, n1 = dataset.shape
    i1 = np.array([seed.i1 for seed in seeds])
    i2 = np.array([seed.i2 for seed in seeds])
    fl = np.array([seed.fl for seed in seeds], dtype=np.float32)
    seed_values = np.array(
        [(seed.i1, seed.i2, seed.fl, seed.fp) for seed in seeds],
        dtype=np.float32,
    )

    assert ((0 <= i1) & (i1 < n1)).all()
    assert ((0 <= i2) & (i2 < n2)).all()
    assert (fl > fm).all()
    assert np.isfinite(seed_values).all()


def test_f3d2d_fvt_fixture_structure_smoke() -> None:
    dataset = REFERENCE_DATASETS_2D["f3d2d"]
    fvt = load_reference_2d_array("f3d2d", "fvt.dat")

    assert fvt.shape == dataset.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    assert fvt.min() >= -1e-6
    assert fvt.max() <= 1.0 + 1e-6
    assert 0 < np.count_nonzero(fvt) < dataset.sample_count


@pytest.mark.slow_reference_voting
def test_f3d2d_apply_voting_and_thin_practical_equivalence_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print optional 2D voting/thinning metrics without imposing thresholds."""

    if os.environ.get("PYOSV_RUN_SLOW_REFERENCE_VOTING") != "1":
        pytest.skip("set PYOSV_RUN_SLOW_REFERENCE_VOTING=1 to run slow reference voting")
    if "PYOSV_REFERENCE_OSV" not in os.environ:
        pytest.skip("set PYOSV_REFERENCE_OSV to the reference_osv root")

    dataset = REFERENCE_DATASETS_2D["f3d2d"]
    ft = load_reference_2d_array("f3d2d", "ft.dat")
    pt = load_reference_2d_array("f3d2d", "pt.dat")
    reference_fv = load_reference_2d_array("f3d2d", "fv.dat")
    reference_fvt = load_reference_2d_array("f3d2d", "fvt.dat")

    voter = OptimalPathVoter(ru=15, rv=30)
    voter.set_strain_max(0.25)
    voter.set_path_smoothing(2)
    pyosv_fv, w1, w2 = voter.apply_voting(d=4, fm=0.3, ft=ft, pt=pt)
    pyosv_fvt = voter.thin(pyosv_fv, w1, w2)

    pyosv_report = finite_value_report(pyosv_fv)
    pyosv_fvt_report = finite_value_report(pyosv_fvt)
    reference_report = finite_value_report(reference_fv)
    reference_fvt_report = finite_value_report(reference_fvt)
    correlation = normalized_correlation(pyosv_fv, reference_fv)
    fvt_correlation = normalized_correlation(pyosv_fvt, reference_fvt)
    overlaps = {
        percentile: top_percentile_overlap(pyosv_fv, reference_fv, percentile=percentile)
        for percentile in (95.0, 99.0)
    }
    fvt_overlaps = {
        percentile: top_percentile_overlap(pyosv_fvt, reference_fvt, percentile=percentile)
        for percentile in (95.0, 99.0)
    }
    report = {
        "dataset": dataset.name,
        "pyosv_fv": pyosv_report,
        "pyosv_fvt": pyosv_fvt_report,
        "reference_fv": reference_report,
        "reference_fvt": reference_fvt_report,
        "normalized_correlation": correlation,
        "thinned_normalized_correlation": fvt_correlation,
        "top_percentile_overlap": overlaps,
        "thinned_top_percentile_overlap": fvt_overlaps,
    }

    with capsys.disabled():
        print(
            "\n2D voting/thinning practical-equivalence report:\n"
            + pformat(report, sort_dicts=False)
        )

    # These are report well-formedness checks only; they intentionally avoid
    # strict practical-equivalence thresholds while the implementation evolves.
    assert pyosv_fv.shape == dataset.shape
    assert pyosv_fvt.shape == dataset.shape
    assert reference_fv.shape == dataset.shape
    assert reference_fvt.shape == dataset.shape
    assert pyosv_report["finite_count"] == dataset.sample_count
    assert pyosv_fvt_report["finite_count"] == dataset.sample_count
    assert reference_report["finite_count"] == dataset.sample_count
    assert reference_fvt_report["finite_count"] == dataset.sample_count
    assert np.isfinite(correlation)
    assert np.isfinite(fvt_correlation)
    assert -1.000001 <= correlation <= 1.000001
    assert -1.000001 <= fvt_correlation <= 1.000001
    assert np.count_nonzero(pyosv_fvt) <= np.count_nonzero(pyosv_fv)
    for overlap in (*overlaps.values(), *fvt_overlaps.values()):
        assert overlap["a_count"] > 0.0
        assert overlap["b_count"] > 0.0

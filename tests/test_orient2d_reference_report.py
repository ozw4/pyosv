import os
from pprint import pformat

import numpy as np
import pytest

from pyosv.metrics import (
    buffered_ridge_overlap,
    finite_value_report,
    normalized_correlation,
    orientation_angle_error,
)
from pyosv.orient2d import FaultOrientScanner2
from pyosv.reference import REFERENCE_DATASETS_2D

from reference_fixtures import load_reference_2d_array


@pytest.mark.slow_reference_scanner
def test_f3d2d_scanner_practical_equivalence_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print optional 2D scanner metrics without imposing fixed thresholds."""

    if os.environ.get("PYOSV_RUN_SLOW_REFERENCE_SCANNER") != "1":
        pytest.skip("set PYOSV_RUN_SLOW_REFERENCE_SCANNER=1 to run scanner report")
    if "PYOSV_REFERENCE_OSV" not in os.environ:
        pytest.skip("set PYOSV_REFERENCE_OSV to the reference_osv root")

    dataset = REFERENCE_DATASETS_2D["f3d2d"]
    crop = (slice(160, 280), slice(60, 156))
    planarity = load_reference_2d_array("f3d2d", "el.dat")[crop]
    reference_ft = load_reference_2d_array("f3d2d", "ft.dat")[crop]
    reference_pt = load_reference_2d_array("f3d2d", "pt.dat")[crop]

    scanner = FaultOrientScanner2(sigma1=2.0)
    pyosv_ft, pyosv_pt = scanner.scan(0.0, 180.0, planarity)

    high_likelihood = reference_ft >= np.percentile(reference_ft, 99.0)
    angle_error = orientation_angle_error(
        pyosv_pt[high_likelihood],
        reference_pt[high_likelihood],
        period=180.0,
    )
    report = {
        "dataset": dataset.name,
        "crop": tuple((axis.start, axis.stop) for axis in crop),
        "pyosv_ft": finite_value_report(pyosv_ft),
        "reference_ft": finite_value_report(reference_ft),
        "ft_normalized_correlation": normalized_correlation(pyosv_ft, reference_ft),
        "ft_buffered_ridge_overlap": buffered_ridge_overlap(
            reference_ft,
            pyosv_ft,
            percentile=99.0,
            radius=2.0,
        ),
        "pt_high_likelihood_angle_error": {
            "count": int(angle_error.size),
            "median": float(np.median(angle_error)) if angle_error.size else None,
            "p90": float(np.percentile(angle_error, 90.0)) if angle_error.size else None,
        },
    }

    with capsys.disabled():
        print("\n2D scanner practical-equivalence report:\n" + pformat(report, sort_dicts=False))

    assert pyosv_ft.shape == planarity.shape
    assert pyosv_pt.shape == planarity.shape
    assert reference_ft.shape == planarity.shape
    assert reference_pt.shape == planarity.shape
    assert pyosv_ft.dtype == np.float32
    assert pyosv_pt.dtype == np.float32
    assert np.isfinite(pyosv_ft).all()
    assert np.isfinite(pyosv_pt).all()
    assert np.isfinite(reference_ft).all()
    assert np.isfinite(reference_pt).all()
    assert -1.000001 <= report["ft_normalized_correlation"] <= 1.000001
    assert report["ft_buffered_ridge_overlap"]["reference_count"] > 0
    assert report["ft_buffered_ridge_overlap"]["candidate_count"] > 0

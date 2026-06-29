import numpy as np
import pytest

from pyosv.metrics import finite_value_report, normalized_correlation, top_percentile_overlap
from pyosv.reference import REFERENCE_DATASETS_2D

from reference_fixtures import load_reference_2d_array

REFERENCE_METRIC_FILES = ("ft.dat", "pt.dat", "fv.dat", "fvt.dat")
FINITE_REPORT_FIELDS = {
    "shape",
    "size",
    "finite_count",
    "nan_count",
    "posinf_count",
    "neginf_count",
    "finite_fraction",
    "finite_min",
    "finite_max",
    "finite_mean",
}


@pytest.mark.parametrize("file_name", REFERENCE_METRIC_FILES)
def test_f3d2d_reference_output_metric_inputs(file_name: str) -> None:
    dataset = REFERENCE_DATASETS_2D["f3d2d"]
    data = load_reference_2d_array("f3d2d", file_name)

    assert data.shape == dataset.shape
    assert np.isfinite(data).all()

    report = finite_value_report(data)

    assert set(report) == FINITE_REPORT_FIELDS
    assert report["shape"] == dataset.shape
    assert report["size"] == dataset.sample_count
    assert report["finite_count"] == dataset.sample_count
    assert report["finite_fraction"] == 1.0
    assert report["nan_count"] == 0
    assert report["posinf_count"] == 0
    assert report["neginf_count"] == 0


@pytest.mark.parametrize("file_name", REFERENCE_METRIC_FILES)
def test_f3d2d_reference_output_self_correlation(file_name: str) -> None:
    data = load_reference_2d_array("f3d2d", file_name)

    correlation = normalized_correlation(data, data)

    if np.ptp(data) == 0.0:
        assert correlation == 0.0
    else:
        assert correlation == pytest.approx(1.0)


@pytest.mark.parametrize("file_name", REFERENCE_METRIC_FILES)
def test_f3d2d_reference_output_self_top_percentile_overlap(file_name: str) -> None:
    data = load_reference_2d_array("f3d2d", file_name)

    overlap = top_percentile_overlap(data, data, percentile=95.0)

    assert overlap["percentile"] == 95.0
    assert overlap["a_count"] == overlap["b_count"]
    assert overlap["a_count"] == overlap["overlap_count"]
    assert overlap["a_count"] == overlap["union_count"]
    assert overlap["overlap_over_a"] == 1.0
    assert overlap["overlap_over_b"] == 1.0
    assert overlap["jaccard"] == 1.0

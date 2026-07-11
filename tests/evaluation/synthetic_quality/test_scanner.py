from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.config import SyntheticScannerConfig
from pyosv.evaluation.synthetic_quality.scanner import (
    SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_BASE,
    SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_SCALE,
    ScannerAttributes,
    scan_ensemble_attributes,
    scanner_attributes_from_case,
)
from pyosv.synthetic3d import make_single_vertical_plane_case


@pytest.mark.parametrize("backend", ("reference-like", "fast", "quality", "ensemble"))
def test_scanner_backends_return_finite_float32_volumes(backend: str) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    config = SyntheticScannerConfig(
        backend=backend,
        phi_min=0.0,
        phi_max=0.0,
        theta_min=90.0,
        theta_max=90.0,
        sigma1=1.0,
        sigma2=1.0,
        scanner_thin_mode="none",
    )

    attributes = scanner_attributes_from_case(case, config)

    assert isinstance(attributes, ScannerAttributes)
    expected_keys = {
        "scanner_input",
        "scanner_ft",
        "scanner_fet",
        "scanner_pt",
        "scanner_fpt",
        "scanner_tt",
        "scanner_ftt",
    }
    assert expected_keys <= attributes.volumes.keys()
    if backend == "quality":
        assert "scanner_confidence" in attributes.volumes
    for volume in attributes.volumes.values():
        assert volume.shape == case.ft_oracle.shape
        assert volume.dtype == np.float32
        assert np.all(np.isfinite(volume))


def test_scanner_attributes_are_deterministic() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    config = SyntheticScannerConfig(
        backend="fast",
        phi_min=0.0,
        phi_max=0.0,
        theta_min=90.0,
        theta_max=90.0,
        sigma1=1.0,
        sigma2=1.0,
        scanner_thin_mode="none",
    )

    first = scanner_attributes_from_case(case, config)
    second = scanner_attributes_from_case(case, config)

    assert first.report == second.report
    assert first.volumes.keys() == second.volumes.keys()
    for name in first.volumes:
        np.testing.assert_array_equal(first.volumes[name], second.volumes[name])


@pytest.mark.parametrize("factor", (0, 5, True, 1.5))
def test_scanner_refinement_factor_validation(factor: object) -> None:
    with pytest.raises(ValueError, match="scanner_refinement_factor"):
        SyntheticScannerConfig(refinement_factor=factor)  # type: ignore[arg-type]


def test_ensemble_quality_confidence_weight_preserves_existing_selection() -> None:
    shape = (1, 1, 2)

    class StubScanner:
        def scan(self, *args):
            return (
                np.array([[[0.0, 1.0]]], dtype=np.float32),
                np.full(shape, 10.0, dtype=np.float32),
                np.full(shape, 20.0, dtype=np.float32),
            )

        def scan_quality(self, *args, refinement_factor, return_confidence):
            assert refinement_factor == 2
            assert return_confidence is True
            return (
                np.array([[[0.0, 1.0]]], dtype=np.float32),
                np.full(shape, 30.0, dtype=np.float32),
                np.full(shape, 40.0, dtype=np.float32),
                np.array([[[0.0, 1.0]]], dtype=np.float32),
            )

        def scan_fast(self, *args):
            return (
                np.array([[[0.0, 1.0]]], dtype=np.float32),
                np.full(shape, 50.0, dtype=np.float32),
                np.full(shape, 60.0, dtype=np.float32),
            )

    config = SyntheticScannerConfig(scanner_thin_mode="none")
    ft, pt, tt, confidence, report = scan_ensemble_attributes(
        StubScanner(), config, np.zeros(shape, dtype=np.float32)
    )

    assert confidence is None
    np.testing.assert_array_equal(ft, np.array([[[0.0, 1.0]]], dtype=np.float32))
    np.testing.assert_array_equal(pt, np.array([[[10.0, 30.0]]], dtype=np.float32))
    np.testing.assert_array_equal(tt, np.array([[[20.0, 40.0]]], dtype=np.float32))
    assert report["quality_confidence_weight"] == {
        "base": SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_BASE,
        "scale": SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_SCALE,
    }


@pytest.mark.parametrize("backend", ("reference-like", "fast", "quality", "ensemble"))
def test_report_wrapper_matches_scanner_api_exactly(backend: str) -> None:
    script = Path(__file__).resolve().parents[3] / "examples" / "report_3d_synthetic_quality.py"
    spec = importlib.util.spec_from_file_location("report_3d_synthetic_quality", script)
    assert spec is not None and spec.loader is not None
    report_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_module)
    case = make_single_vertical_plane_case((9, 9, 9))
    config = SyntheticScannerConfig(
        backend=backend,
        phi_min=0.0,
        phi_max=0.0,
        theta_min=90.0,
        theta_max=90.0,
        sigma1=1.0,
        sigma2=1.0,
        scanner_thin_mode="none",
    )

    expected = scanner_attributes_from_case(case, config)
    report, volumes = report_module._scanner_attributes_from_case(case, config)

    assert report == expected.report
    assert volumes.keys() == expected.volumes.keys()
    for name in volumes:
        np.testing.assert_array_equal(volumes[name], expected.volumes[name])


def test_report_wrapper_preserves_backend_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[3] / "examples" / "report_3d_synthetic_quality.py"
    spec = importlib.util.spec_from_file_location("report_3d_synthetic_quality", script)
    assert spec is not None and spec.loader is not None
    report_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_module)
    case = make_single_vertical_plane_case((9, 9, 9))
    config = SyntheticScannerConfig(
        backend="fast",
        phi_min=0.0,
        phi_max=0.0,
        theta_min=90.0,
        theta_max=90.0,
        sigma1=1.0,
        sigma2=1.0,
        scanner_thin_mode="none",
    )
    calls: list[str] = []

    def fake_backend(scanner, scanner_config, scanner_input, backend):
        del scanner, scanner_config
        calls.append(backend)
        zeros = np.zeros_like(scanner_input, dtype=np.float32)
        return zeros, zeros, zeros, None

    monkeypatch.setattr(report_module, "_scan_backend_attributes", fake_backend)

    _, volumes = report_module._scanner_attributes_from_case(case, config)

    assert calls == ["fast"]
    np.testing.assert_array_equal(volumes["scanner_ft"], 0.0)


def test_report_wrapper_preserves_ensemble_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[3] / "examples" / "report_3d_synthetic_quality.py"
    spec = importlib.util.spec_from_file_location("report_3d_synthetic_quality", script)
    assert spec is not None and spec.loader is not None
    report_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report_module)
    case = make_single_vertical_plane_case((9, 9, 9))
    config = SyntheticScannerConfig(
        backend="ensemble",
        phi_min=0.0,
        phi_max=0.0,
        theta_min=90.0,
        theta_max=90.0,
        sigma1=1.0,
        sigma2=1.0,
        scanner_thin_mode="none",
    )
    calls = 0

    def fake_ensemble(scanner, scanner_config, scanner_input):
        del scanner, scanner_config
        nonlocal calls
        calls += 1
        zeros = np.zeros_like(scanner_input, dtype=np.float32)
        report = {"selection_fraction_by_backend": {"fast": 1.0}}
        return zeros, zeros, zeros, None, report

    monkeypatch.setattr(report_module, "_scan_ensemble_attributes", fake_ensemble)

    report, volumes = report_module._scanner_attributes_from_case(case, config)

    assert calls == 1
    assert report["selection_fraction_by_backend"] == {"fast": 1.0}
    np.testing.assert_array_equal(volumes["scanner_ft"], 0.0)

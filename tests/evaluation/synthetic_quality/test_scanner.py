from __future__ import annotations

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.config import SyntheticScannerConfig
from pyosv.evaluation.synthetic_quality.scanner import (
    SCANNER_BACKENDS,
    SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_BASE,
    SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_SCALE,
    ScannerAttributes,
    scan_backend_attributes,
    scan_ensemble_attributes,
    scanner_attributes_from_case,
    scanner_attributes_from_input,
)
from pyosv.synthetic3d import make_scanner_input_from_case, make_single_vertical_plane_case


def test_scanner_backends_contract() -> None:
    assert SCANNER_BACKENDS == ("reference-like", "fast", "quality", "ensemble")


@pytest.mark.parametrize("backend", SCANNER_BACKENDS)
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


@pytest.mark.parametrize("backend", ("reference-like", "quality"))
def test_scanner_backend_routing_is_independent_of_workflow_profiles(backend: str) -> None:
    shape = (1, 2, 2)
    reference_result = tuple(np.full(shape, value, dtype=np.float32) for value in (1.0, 2.0, 3.0))
    quality_confidence = np.full(shape, 0.75, dtype=np.float32)
    quality_result = (
        *(np.full(shape, value, dtype=np.float32) for value in (4.0, 5.0, 6.0)),
        quality_confidence,
    )

    class StubScanner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def scan(self, *args: object, **kwargs: object):
            self.calls.append(("scan", kwargs))
            return reference_result

        def scan_quality(self, *args: object, **kwargs: object):
            self.calls.append(("scan_quality", kwargs))
            return quality_result

        def scan_fast(self, *args: object, **kwargs: object):
            self.calls.append(("scan_fast", kwargs))
            return reference_result

    # Scanner backend routing is separate from workflow profile resolution. The shared
    # "quality" label is not a combined mode or preset, and no workflow mode can
    # implicitly select a backend or change scanner_thin_mode here.
    scanner = StubScanner()
    config = SyntheticScannerConfig(refinement_factor=3, scanner_thin_mode="normal")
    ft, pt, tt, confidence = scan_backend_attributes(
        scanner,  # type: ignore[arg-type]
        config,
        np.zeros(shape, dtype=np.float32),
        backend,
    )

    expected_arrays = quality_result[:3] if backend == "quality" else reference_result
    for actual, expected in zip((ft, pt, tt), expected_arrays):
        np.testing.assert_array_equal(actual, expected)
    assert config.scanner_thin_mode == "normal"
    if backend == "reference-like":
        assert scanner.calls == [("scan", {})]
        assert confidence is None
    else:
        assert scanner.calls == [
            (
                "scan_quality",
                {"refinement_factor": config.refinement_factor, "return_confidence": True},
            )
        ]
        assert confidence is quality_confidence


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


def test_scanner_attributes_from_case_matches_prepared_input_path() -> None:
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
    scanner_input = make_scanner_input_from_case(case, config.input_config)

    wrapped = scanner_attributes_from_case(case, config)
    prepared = scanner_attributes_from_input(case, config, scanner_input)

    assert wrapped.report == prepared.report
    assert wrapped.volumes.keys() == prepared.volumes.keys()
    for name in wrapped.volumes:
        np.testing.assert_array_equal(wrapped.volumes[name], prepared.volumes[name])


def test_scanner_attributes_from_input_protects_float32_input_without_copy() -> None:
    case = make_single_vertical_plane_case((3, 3, 3))
    scanner_input = np.ones(case.shape, dtype=np.float32)
    original = scanner_input.copy()
    seen_input: np.ndarray | None = None

    def backend_scan(scanner, config, input_array, backend):
        nonlocal seen_input
        seen_input = input_array
        with pytest.raises(ValueError, match="read-only"):
            input_array[0, 0, 0] = 0.0
        volumes = tuple(np.full(case.shape, value, dtype=np.float32) for value in (1, 2, 3))
        return *volumes, None

    attributes = scanner_attributes_from_input(
        case,
        SyntheticScannerConfig(backend="fast", scanner_thin_mode="none"),
        scanner_input,
        backend_scan=backend_scan,
    )

    assert seen_input is not None
    assert np.shares_memory(seen_input, scanner_input)
    assert scanner_input.flags.writeable
    np.testing.assert_array_equal(scanner_input, original)
    assert np.shares_memory(attributes.volumes["scanner_input"], scanner_input)


@pytest.mark.parametrize(
    ("scanner_input", "message"),
    (
        (np.zeros((2, 2, 2), dtype=np.float32), "shape must match"),
        (np.full((3, 3, 3), "x"), "numeric array"),
        (np.full((3, 3, 3), np.nan, dtype=np.float32), "finite values"),
    ),
)
def test_scanner_attributes_from_input_validates_input_before_scanning(
    scanner_input: np.ndarray, message: str
) -> None:
    case = make_single_vertical_plane_case((3, 3, 3))

    def unexpected_scan(*args):
        raise AssertionError("invalid input must not be scanned")

    with pytest.raises(ValueError, match=message):
        scanner_attributes_from_input(
            case,
            SyntheticScannerConfig(backend="fast"),
            scanner_input,
            backend_scan=unexpected_scan,
        )


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

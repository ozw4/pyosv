"""Scanner attribute generation for synthetic quality evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pyosv.orient3d import FaultOrientScanner3
from pyosv.synthetic3d import Synthetic3DCase, make_scanner_input_from_case

if TYPE_CHECKING:
    from .config import SyntheticScannerConfig

SCANNER_BACKENDS = ("reference-like", "fast", "quality", "ensemble")
SCANNER_ENSEMBLE_COMPONENT_BACKENDS = ("reference-like", "quality", "fast")
SCANNER_ENSEMBLE_PRIORS = {
    "reference-like": 1.00,
    "quality": 1.05,
    "fast": 1.00,
}
SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_BASE = 0.75
SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_SCALE = 0.25
_NONZERO_EPSILON = 1.0e-6

BackendScan = Callable[
    [FaultOrientScanner3, "SyntheticScannerConfig", np.ndarray, str],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
]
EnsembleScan = Callable[
    [FaultOrientScanner3, "SyntheticScannerConfig", np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, None, dict[str, Any]],
]


@dataclass(frozen=True, slots=True)
class ScannerAttributes:
    """Scanner report data and generated volumes for one synthetic case."""

    report: Mapping[str, Any]
    volumes: Mapping[str, np.ndarray]


def scanner_attributes_from_case(
    case: Synthetic3DCase,
    scanner_config: SyntheticScannerConfig,
    *,
    backend_scan: BackendScan | None = None,
    ensemble_scan: EnsembleScan | None = None,
) -> ScannerAttributes:
    """Generate scanner input and attributes without running downstream stages."""

    scanner_input = make_scanner_input_from_case(case, scanner_config.input_config)
    return scanner_attributes_from_input(
        case,
        scanner_config,
        scanner_input,
        backend_scan=backend_scan,
        ensemble_scan=ensemble_scan,
    )


def scanner_attributes_from_input(
    case: Synthetic3DCase,
    scanner_config: SyntheticScannerConfig,
    scanner_input: np.ndarray,
    *,
    backend_scan: BackendScan | None = None,
    ensemble_scan: EnsembleScan | None = None,
) -> ScannerAttributes:
    """Generate one backend's attributes from a prepared scanner input."""

    input_array = _validated_scanner_input(case, scanner_input)
    backend_scan = scan_backend_attributes if backend_scan is None else backend_scan
    ensemble_scan = scan_ensemble_attributes if ensemble_scan is None else ensemble_scan
    scanner = FaultOrientScanner3(scanner_config.sigma1, scanner_config.sigma2)
    ensemble_report: dict[str, Any] | None = None
    if scanner_config.backend == "ensemble":
        ft_scan, pt_scan, tt_scan, confidence, ensemble_report = ensemble_scan(
            scanner,
            scanner_config,
            input_array,
        )
    else:
        ft_scan, pt_scan, tt_scan, confidence = backend_scan(
            scanner,
            scanner_config,
            input_array,
            scanner_config.backend,
        )

    if scanner_config.scanner_thin_mode == "none":
        ft_used = ft_scan
        pt_used = pt_scan
        tt_used = tt_scan
    else:
        ft_used, pt_used, tt_used = scanner.thin(
            ft_scan,
            pt_scan,
            tt_scan,
            mode=scanner_config.scanner_thin_mode,
            remove_edge_effects=scanner_config.remove_edge_effects,
        )

    scanner_report = {
        "config": scanner_config.as_report_dict(),
        "input": _array_summary(input_array),
        "ft": _array_summary(ft_scan),
        "fet": _array_summary(ft_used),
        "pt": _array_summary(pt_scan),
        "fpt": _array_summary(pt_used),
        "tt": _array_summary(tt_scan),
        "ftt": _array_summary(tt_used),
    }
    scanner_volumes = {
        "scanner_input": input_array,
        "scanner_ft": ft_scan,
        "scanner_fet": ft_used,
        "scanner_pt": pt_scan,
        "scanner_fpt": pt_used,
        "scanner_tt": tt_scan,
        "scanner_ftt": tt_used,
    }
    if confidence is not None:
        scanner_report["confidence"] = _array_summary(confidence)
        scanner_volumes["scanner_confidence"] = confidence
    if ensemble_report is not None:
        scanner_report["selection_fraction_by_backend"] = ensemble_report[
            "selection_fraction_by_backend"
        ]
        scanner_report["ensemble"] = ensemble_report
    return ScannerAttributes(report=scanner_report, volumes=scanner_volumes)


def _validated_scanner_input(case: Synthetic3DCase, scanner_input: np.ndarray) -> np.ndarray:
    input_array = np.asarray(scanner_input)
    if input_array.shape != case.shape:
        raise ValueError(
            f"scanner_input shape must match case shape {case.shape}, got {input_array.shape}"
        )
    if input_array.dtype.kind not in "iuf":
        raise ValueError("scanner_input must be a numeric array")
    if not np.all(np.isfinite(input_array)):
        raise ValueError("scanner_input must contain only finite values")
    input_float32 = input_array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(input_float32)):
        raise ValueError("scanner_input must contain only finite float32 values")

    if _has_immutable_bytes_backing(input_float32):
        return input_float32

    # A read-only ndarray view is insufficient here: its writeable base remains
    # reachable, and an injected backend can re-enable writes on the view. Copy once
    # into an immutable bytes buffer at the backend boundary so neither the caller's
    # array nor a later backend can be contaminated.
    immutable_buffer = input_float32.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=np.float32).reshape(input_float32.shape)


def _has_immutable_bytes_backing(array: np.ndarray) -> bool:
    base: object = array
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return isinstance(base, bytes)


def scan_backend_attributes(
    scanner: FaultOrientScanner3,
    scanner_config: SyntheticScannerConfig,
    scanner_input: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Run one scanner backend with the report workflow's established settings."""

    if backend == "reference-like":
        ft_scan, pt_scan, tt_scan = scanner.scan(
            scanner_config.phi_min,
            scanner_config.phi_max,
            scanner_config.theta_min,
            scanner_config.theta_max,
            scanner_input,
        )
        return ft_scan, pt_scan, tt_scan, None
    if backend == "quality":
        ft_scan, pt_scan, tt_scan, confidence = scanner.scan_quality(
            scanner_config.phi_min,
            scanner_config.phi_max,
            scanner_config.theta_min,
            scanner_config.theta_max,
            scanner_input,
            refinement_factor=scanner_config.refinement_factor,
            return_confidence=True,
        )
        return ft_scan, pt_scan, tt_scan, confidence
    if backend == "fast":
        ft_scan, pt_scan, tt_scan = scanner.scan_fast(
            scanner_config.phi_min,
            scanner_config.phi_max,
            scanner_config.theta_min,
            scanner_config.theta_max,
            scanner_input,
        )
        return ft_scan, pt_scan, tt_scan, None
    raise ValueError("scanner_backend must be 'reference-like', 'fast', 'quality', or 'ensemble'")


def scan_ensemble_attributes(
    scanner: FaultOrientScanner3,
    scanner_config: SyntheticScannerConfig,
    scanner_input: np.ndarray,
    *,
    backend_scan: BackendScan | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None, dict[str, Any]]:
    """Run and combine the established scanner ensemble components."""

    backend_scan = scan_backend_attributes if backend_scan is None else backend_scan
    components: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for backend in SCANNER_ENSEMBLE_COMPONENT_BACKENDS:
        components[backend] = backend_scan(
            scanner,
            scanner_config,
            scanner_input,
            backend,
        )

    adjusted_scores: list[np.ndarray] = []
    component_reports: dict[str, Any] = {}
    for backend in SCANNER_ENSEMBLE_COMPONENT_BACKENDS:
        ft_scan, pt_scan, tt_scan, confidence = components[backend]
        adjusted_score = unit_range_normalize(ft_scan) * np.float32(
            SCANNER_ENSEMBLE_PRIORS[backend]
        )
        if backend == "quality":
            if confidence is None:
                raise ValueError("quality ensemble component must provide confidence")
            quality_weight = (
                np.float32(SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_BASE)
                + np.float32(SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_SCALE) * confidence
            )
            adjusted_score = adjusted_score * quality_weight
        adjusted_score = adjusted_score.astype(np.float32, copy=False)
        adjusted_scores.append(adjusted_score)
        component_report = {
            "ft": _array_summary(ft_scan),
            "pt": _array_summary(pt_scan),
            "tt": _array_summary(tt_scan),
            "adjusted_score": _array_summary(adjusted_score),
        }
        if confidence is not None:
            component_report["confidence"] = _array_summary(confidence)
        component_reports[backend] = component_report

    selection = np.argmax(np.stack(adjusted_scores, axis=0), axis=0)
    ft_ensemble = np.empty(scanner_input.shape, dtype=np.float32)
    pt_ensemble = np.empty(scanner_input.shape, dtype=np.float32)
    tt_ensemble = np.empty(scanner_input.shape, dtype=np.float32)
    total_count = float(selection.size)
    selection_fraction_by_backend: dict[str, float] = {}
    for index, backend in enumerate(SCANNER_ENSEMBLE_COMPONENT_BACKENDS):
        selected = selection == index
        ft_scan, pt_scan, tt_scan, _ = components[backend]
        ft_ensemble[selected] = ft_scan[selected]
        pt_ensemble[selected] = pt_scan[selected]
        tt_ensemble[selected] = tt_scan[selected]
        selection_fraction_by_backend[backend] = float(np.count_nonzero(selected) / total_count)

    report = {
        "component_backends": list(SCANNER_ENSEMBLE_COMPONENT_BACKENDS),
        "component_priors": {
            backend: float(prior) for backend, prior in SCANNER_ENSEMBLE_PRIORS.items()
        },
        "quality_confidence_weight": {
            "base": float(SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_BASE),
            "scale": float(SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_SCALE),
        },
        "selection_fraction_by_backend": selection_fraction_by_backend,
        "components": component_reports,
    }
    return ft_ensemble, pt_ensemble, tt_ensemble, None, report


def unit_range_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize non-negative values to float32 in the closed unit interval."""

    array_float32 = np.maximum(np.asarray(array, dtype=np.float32), np.float32(0.0))
    low = float(np.min(array_float32))
    high = float(np.max(array_float32))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(array_float32, dtype=np.float32)
    normalized = (array_float32 - np.float32(low)) / np.float32(high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64, copy=False)
    if finite_values.size:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        mean = float(np.mean(finite_values))
    else:
        minimum = float("nan")
        maximum = float("nan")
        mean = float("nan")

    return {
        "shape": [int(size) for size in values.shape],
        "finite_count": int(np.count_nonzero(finite)),
        "finite_fraction": (float(np.count_nonzero(finite) / values.size) if values.size else 0.0),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "nonzero_fraction": (
            float(np.count_nonzero(np.abs(values) > _NONZERO_EPSILON) / values.size)
            if values.size
            else 0.0
        ),
    }

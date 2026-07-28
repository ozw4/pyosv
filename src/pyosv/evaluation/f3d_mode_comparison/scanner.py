"""Shared full-volume scanner stages for the canonical F3 comparison."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from pyosv.candidate_volume import NONZERO_EPSILON, nonzero_count
from pyosv.orient3d import FaultOrientScanner3

from .artifacts import (
    F3RunWorkspace,
    F3StageArtifact,
    F3StageResult,
    F3WorkspaceMismatchError,
    _callable_implementation_identity,
    _workspace_dataset_file_identity,
    canonical_json_bytes,
    stage_fingerprint,
    validate_stage,
)
from .config import F3ScannerConfig
from .data import F3FileIdentity, F3VolumeSource
from .models import F3ModeComparisonPlan, F3ScannerBackend

F3_SCANNER_STAGE_CONTRACT_VERSION = 5
F3_SCANNER_STAGE_IMPLEMENTATION = "pyosv-f3-scanner-stage-v5"
F3_SCANNER_BACKEND_ORDER: tuple[F3ScannerBackend, ...] = (
    "reference-like",
    "quality",
)
_COMMON_VOLUME_NAMES = ("ft", "pt", "tt", "fet", "fpt", "ftt")
_UNIT_RANGE_NAMES = {"ft", "fet", "confidence"}
_DAT_DTYPE = np.dtype(">f4")
_WRITE_SLAB_COUNT = 1


class ScannerProtocol(Protocol):
    """Operations required from an F3 scanner implementation."""

    def scan(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]: ...

    def scan_quality(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]: ...

    def thin(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]: ...

    def reference_like_strike_sampling(self, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def reference_like_dip_sampling(self, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def refined_reference_like_strike_sampling(self, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def refined_reference_like_dip_sampling(self, *args: Any, **kwargs: Any) -> np.ndarray: ...


ScannerFactory = Callable[[float, float], ScannerProtocol]


class _StageRSSRecorder(Protocol):
    def stage_before(
        self,
        stage_kind: str,
        fingerprint: str,
        *,
        phase: str | None = None,
    ) -> object: ...

    def stage_after(
        self,
        stage_kind: str,
        fingerprint: str,
        *,
        phase: str | None = None,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class F3ScannerStageResult:
    """Scalar-only reference to one validated scanner stage."""

    backend: F3ScannerBackend
    path: Path
    fingerprint: str
    reused: bool
    shape: tuple[int, int, int]
    input_fingerprint: str
    report: Mapping[str, Any]
    elapsed_seconds: float = 0.0
    input_bytes: int = 0
    output_bytes: int = 0


@dataclass(slots=True)
class F3LoadedScannerStage:
    """Read-only memory maps opened from one validated scanner stage."""

    backend: F3ScannerBackend
    path: Path
    fingerprint: str
    shape: tuple[int, int, int]
    report: Mapping[str, Any]
    ft: np.memmap
    pt: np.memmap
    tt: np.memmap
    fet: np.memmap
    fpt: np.memmap
    ftt: np.memmap
    confidence: np.memmap | None = None
    _closed: bool = False

    @property
    def closed(self) -> bool:
        """Whether all owned memory maps have been closed."""

        return self._closed

    def close(self) -> None:
        """Close all memory maps owned by this loaded stage."""

        if self._closed:
            return
        for array in (
            self.ft,
            self.pt,
            self.tt,
            self.fet,
            self.fpt,
            self.ftt,
            self.confidence,
        ):
            if array is None:
                continue
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
        self._closed = True

    def __enter__(self) -> F3LoadedScannerStage:
        if self._closed:
            raise RuntimeError("F3LoadedScannerStage is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def scanner_stage_artifacts(
    shape: tuple[int, int, int],
    backend: F3ScannerBackend,
) -> tuple[F3StageArtifact, ...]:
    """Return the fixed artifact schema for one scanner backend."""

    valid_shape = _validated_shape(shape)
    _validate_backend(backend)
    names = (*_COMMON_VOLUME_NAMES, *(("confidence",) if backend == "quality" else ()))
    return (
        *(F3StageArtifact(f"{name}.dat", valid_shape, ">f4") for name in names),
        F3StageArtifact("report.json"),
    )


def scanner_stage_resolved_settings(
    config: F3ScannerConfig,
    shape: tuple[int, int, int],
    *,
    implementation_identity: Mapping[str, Any] | str | None = None,
    sampling_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return every scanner-stage control used by its cache fingerprint."""

    if not isinstance(config, F3ScannerConfig):
        raise TypeError("config must be an F3ScannerConfig")
    valid_shape = _validated_shape(shape)
    implementation = _normalized_implementation_identity(implementation_identity)
    evidence = (
        scanner_sampling_evidence(
            FaultOrientScanner3(config.sigma1, config.sigma2),
            config,
            config.backend,
            implementation_identity=implementation,
        )
        if sampling_evidence is None
        else validate_scanner_sampling_evidence(
            sampling_evidence,
            config,
            config.backend,
            expected_implementation_identity=implementation,
        )
    )
    return {
        "scanner_stage_contract_version": F3_SCANNER_STAGE_CONTRACT_VERSION,
        "scanner_stage_implementation_identity": implementation,
        "shape": list(valid_shape),
        "backend": config.backend,
        "angle_range": {
            "phi_min": config.phi_min,
            "phi_max": config.phi_max,
            "theta_min": config.theta_min,
            "theta_max": config.theta_max,
        },
        "sigma": {"sigma1": config.sigma1, "sigma2": config.sigma2},
        "quality_refinement": config.refinement_factor,
        "orientation_backend": config.orientation_backend,
        "interpolation": {
            "backend": config.interpolation_backend,
            "order": config.interpolation_order,
        },
        "smoothing_sigma": config.smoothing_sigma,
        "normalize": config.normalize,
        "dtype": config.dtype,
        "sampling_evidence": evidence,
        "scanner_thinning": {
            "mode": config.scanner_thin_mode,
            "reference_sigma": config.reference_thin_sigma,
            "requested_remove_edge_effects": config.remove_edge_effects,
            "effective_remove_edge_effects": config.effective_remove_edge_effects,
        },
    }


def canonical_scanner_implementation_identity() -> Mapping[str, Any] | str:
    """Return the implementation identity required by official F3 scanner stages."""

    return _normalized_implementation_identity(None)


def scanner_stage_fingerprint(
    workspace: F3RunWorkspace,
    input_identity: F3FileIdentity,
    config: F3ScannerConfig,
    *,
    implementation_identity: Mapping[str, Any] | str | None = None,
    sampling_evidence: Mapping[str, Any] | None = None,
) -> str:
    """Build the exact fingerprint for one scanner backend stage."""

    _validate_workspace_and_input(workspace, input_identity)
    settings = scanner_stage_resolved_settings(
        config,
        input_identity.shape,
        implementation_identity=implementation_identity,
        sampling_evidence=sampling_evidence,
    )
    return stage_fingerprint(
        "scanner",
        run_fingerprint_value=workspace.fingerprint,
        input_fingerprints={"ep.dat": input_identity.sha256},
        resolved_settings=settings,
        artifacts=scanner_stage_artifacts(input_identity.shape, config.backend),
    )


def run_scanner_stages(
    workspace: F3RunWorkspace,
    volume_source: F3VolumeSource,
    plan: F3ModeComparisonPlan,
    *,
    scanner_factory: ScannerFactory = FaultOrientScanner3,
    backend_order: Sequence[F3ScannerBackend] = F3_SCANNER_BACKEND_ORDER,
    implementation_identity: Mapping[str, Any] | str | None = None,
    rss_recorder: _StageRSSRecorder | None = None,
) -> dict[F3ScannerBackend, F3ScannerStageResult]:
    """Materialize or reuse both scanner backends with one shared input read.

    Backends run sequentially. The input is read lazily, so an all-reuse run
    performs no source read or scanner construction.
    """

    if not isinstance(workspace, F3RunWorkspace):
        raise TypeError("workspace must be an F3RunWorkspace")
    if not isinstance(volume_source, F3VolumeSource) and not (
        hasattr(volume_source, "identity") and hasattr(volume_source, "read_native_volume")
    ):
        raise TypeError("volume_source must provide identity and read_native_volume")
    if not isinstance(plan, F3ModeComparisonPlan):
        raise TypeError("plan must be an F3ModeComparisonPlan")
    if not callable(scanner_factory):
        raise TypeError("scanner_factory must be callable")
    if rss_recorder is not None and (
        not callable(getattr(rss_recorder, "stage_before", None))
        or not callable(getattr(rss_recorder, "stage_after", None))
    ):
        raise TypeError("rss_recorder must provide stage_before and stage_after")
    order = tuple(backend_order)
    if len(order) != 2 or set(order) != set(F3_SCANNER_BACKEND_ORDER):
        raise ValueError("backend_order must contain reference-like and quality exactly once")

    input_identity = volume_source.identity.file_for("input")
    _validate_workspace_and_input(workspace, input_identity)
    resolved_implementation_identity = _scanner_implementation_identity(
        scanner_factory,
        implementation_identity,
    )
    shared_input: np.ndarray | None = None
    results: dict[F3ScannerBackend, F3ScannerStageResult] = {}

    def immutable_input() -> np.ndarray:
        nonlocal shared_input
        if shared_input is None:
            native = volume_source.read_native_volume("input")
            shared_input = _immutable_view(native, input_identity.shape)
            del native
        return shared_input

    try:
        for backend in order:
            config = plan.scanner_config_for(backend)
            scanner = scanner_factory(config.sigma1, config.sigma2)
            sampling_evidence = scanner_sampling_evidence(
                scanner,
                config,
                backend,
                implementation_identity=resolved_implementation_identity,
            )
            settings = scanner_stage_resolved_settings(
                config,
                input_identity.shape,
                implementation_identity=resolved_implementation_identity,
                sampling_evidence=sampling_evidence,
            )
            artifacts = scanner_stage_artifacts(input_identity.shape, backend)
            fingerprint = scanner_stage_fingerprint(
                workspace,
                input_identity,
                config,
                implementation_identity=resolved_implementation_identity,
                sampling_evidence=sampling_evidence,
            )

            def writer(
                temporary_path: Path,
                *,
                backend: F3ScannerBackend = backend,
                config: F3ScannerConfig = config,
                settings: Mapping[str, Any] = settings,
                fingerprint: str = fingerprint,
            ) -> None:
                _write_scanner_stage(
                    temporary_path,
                    backend=backend,
                    config=config,
                    settings=settings,
                    fingerprint=fingerprint,
                    input_identity=input_identity,
                    scanner_input=immutable_input(),
                    scanner=scanner,
                    sampling_evidence=sampling_evidence,
                )

            if rss_recorder is not None:
                rss_recorder.stage_before(
                    "scanner",
                    fingerprint,
                    phase="compute_or_load_validation",
                )
            started = time.perf_counter()
            try:
                stage = workspace.write_or_reuse_stage(
                    "scanner",
                    input_fingerprints={"ep.dat": input_identity.sha256},
                    resolved_settings=settings,
                    artifacts=artifacts,
                    writer=writer,
                    fingerprint=fingerprint,
                )
                result = _result_from_stage(
                    stage,
                    backend=backend,
                    shape=input_identity.shape,
                    input_fingerprint=input_identity.sha256,
                    config=config,
                    settings=settings,
                    input_bytes=input_identity.size,
                )
            finally:
                elapsed_seconds = time.perf_counter() - started
                if rss_recorder is not None:
                    rss_recorder.stage_after(
                        "scanner",
                        fingerprint,
                        phase="compute_or_load_validation",
                    )
            results[backend] = replace(
                result,
                elapsed_seconds=elapsed_seconds,
            )
    finally:
        shared_input = None

    return results


def load_scanner_stage(stage: F3ScannerStageResult) -> F3LoadedScannerStage:
    """Validate and open a scanner stage without copying its full volumes."""

    if not isinstance(stage, F3ScannerStageResult):
        raise TypeError("stage must be an F3ScannerStageResult")
    manifest = _read_json(stage.path / "stage_manifest.json")
    expected_computation = {
        name: value
        for name, value in manifest.items()
        if name not in {"fingerprint", "shape", "dtype", "files"}
    }
    validate_stage(stage.path, expected_computation, stage.fingerprint)
    report = _read_json(stage.path / "report.json")
    _validate_report_identity(report, stage)
    if report != dict(stage.report):
        raise ValueError("scanner report changed after stage result validation")

    opened: list[np.memmap] = []
    try:
        for name in _COMMON_VOLUME_NAMES:
            opened.append(_open_dat(stage.path / f"{name}.dat", stage.shape))
        confidence = (
            _open_dat(stage.path / "confidence.dat", stage.shape)
            if stage.backend == "quality"
            else None
        )
        if confidence is not None:
            opened.append(confidence)
        return F3LoadedScannerStage(
            backend=stage.backend,
            path=stage.path,
            fingerprint=stage.fingerprint,
            shape=stage.shape,
            report=MappingProxyType(report),
            ft=opened[0],
            pt=opened[1],
            tt=opened[2],
            fet=opened[3],
            fpt=opened[4],
            ftt=opened[5],
            confidence=confidence,
        )
    except BaseException:
        for array in opened:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
        raise


def _write_scanner_stage(
    path: Path,
    *,
    backend: F3ScannerBackend,
    config: F3ScannerConfig,
    settings: Mapping[str, Any],
    fingerprint: str,
    input_identity: F3FileIdentity,
    scanner_input: np.ndarray,
    scanner: ScannerProtocol,
    sampling_evidence: Mapping[str, Any],
) -> None:
    confidence: np.ndarray | None = None
    if backend == "reference-like":
        ft, pt, tt = scanner.scan(
            config.phi_min,
            config.phi_max,
            config.theta_min,
            config.theta_max,
            scanner_input,
            backend=config.orientation_backend,
            interpolation_order=config.interpolation_order,
            interpolation_backend=config.interpolation_backend,
            smoothing_sigma=config.smoothing_sigma,
            normalize=config.normalize,
        )
    else:
        ft, pt, tt, confidence = scanner.scan_quality(
            config.phi_min,
            config.phi_max,
            config.theta_min,
            config.theta_max,
            scanner_input,
            backend=config.orientation_backend,
            refinement_factor=config.refinement_factor,
            interpolation_order=config.interpolation_order,
            interpolation_backend=config.interpolation_backend,
            smoothing_sigma=config.smoothing_sigma,
            normalize=config.normalize,
            return_confidence=True,
        )

    raw = {
        "ft": _validated_output(ft, "ft", input_identity.shape, config),
        "pt": _validated_output(pt, "pt", input_identity.shape, config),
        "tt": _validated_output(tt, "tt", input_identity.shape, config),
    }
    if confidence is not None:
        raw["confidence"] = _validated_output(
            confidence,
            "confidence",
            input_identity.shape,
            config,
        )
    fet, fpt, ftt = scanner.thin(
        raw["ft"],
        raw["pt"],
        raw["tt"],
        mode=config.scanner_thin_mode,
        reference_sigma=config.reference_thin_sigma,
        remove_edge_effects=config.effective_remove_edge_effects,
    )
    thinned = {
        "fet": _validated_output(fet, "fet", input_identity.shape, config, thinned=True),
        "fpt": _validated_output(fpt, "fpt", input_identity.shape, config, thinned=True),
        "ftt": _validated_output(ftt, "ftt", input_identity.shape, config, thinned=True),
    }

    report = {
        "scanner_stage_contract_version": F3_SCANNER_STAGE_CONTRACT_VERSION,
        "fingerprint": fingerprint,
        "backend": backend,
        "shape": list(input_identity.shape),
        "input_fingerprint": input_identity.computation_identity,
        "resolved_config": asdict(config),
        "resolved_stage_settings": dict(settings),
        "sampling_evidence": dict(sampling_evidence),
        "sampling_count": sampling_count_from_evidence(sampling_evidence),
        "requested_remove_edge_effects": config.remove_edge_effects,
        "effective_remove_edge_effects": config.effective_remove_edge_effects,
        "raw": {name: scanner_array_summary(values) for name, values in raw.items()},
        "thinned": {name: scanner_array_summary(values) for name, values in thinned.items()},
    }

    for name in ("ft", "pt", "tt"):
        _write_big_endian_dat(path / f"{name}.dat", raw[name])
    for name in ("fet", "fpt", "ftt"):
        _write_big_endian_dat(path / f"{name}.dat", thinned[name])
    if confidence is not None:
        _write_big_endian_dat(path / "confidence.dat", raw["confidence"])
    (path / "report.json").write_bytes(canonical_json_bytes(report) + b"\n")


def _validated_output(
    value: object,
    name: str,
    shape: tuple[int, int, int],
    config: F3ScannerConfig,
    *,
    thinned: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {value.shape}")
    if value.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} dtype must be float32, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    minimum = float(np.min(value))
    maximum = float(np.max(value))
    if name in _UNIT_RANGE_NAMES and (minimum < 0.0 or maximum > 1.0):
        raise ValueError(f"{name} must be in the closed unit interval")
    if name in {"pt", "fpt"}:
        _validate_angle_range(
            value,
            name,
            config.phi_min,
            config.phi_max,
            allow_zero=thinned,
        )
    if name in {"tt", "ftt"}:
        _validate_angle_range(
            value,
            name,
            config.theta_min,
            config.theta_max,
            allow_zero=thinned,
        )
    return value


def _validate_angle_range(
    values: np.ndarray,
    name: str,
    minimum: float,
    maximum: float,
    *,
    allow_zero: bool,
) -> None:
    tolerance = 8.0 * float(np.finfo(np.float32).eps) * max(1.0, abs(minimum), abs(maximum))
    valid = (values >= minimum - tolerance) & (values <= maximum + tolerance)
    if allow_zero:
        valid |= values == np.float32(0.0)
    if not np.all(valid):
        raise ValueError(f"{name} must be within the configured angle range")


def scanner_sampling_evidence(
    scanner: ScannerProtocol,
    config: F3ScannerConfig,
    backend: F3ScannerBackend,
    *,
    implementation_identity: Mapping[str, Any] | str | None = None,
    require_full_protocol: bool = True,
) -> dict[str, Any]:
    """Return validated sampling evidence from the scanner that will scan."""

    if not isinstance(config, F3ScannerConfig):
        raise TypeError("config must be an F3ScannerConfig")
    _validate_backend(backend)
    if backend != config.backend:
        raise ValueError("sampling backend must match config.backend")
    sampling_helpers = (
        "reference_like_strike_sampling",
        "reference_like_dip_sampling",
        "refined_reference_like_strike_sampling",
        "refined_reference_like_dip_sampling",
    )
    required = (
        ("scan", "scan_quality", "thin", *sampling_helpers)
        if require_full_protocol
        else sampling_helpers
    )
    for name in required:
        if not callable(getattr(scanner, name, None)):
            raise TypeError(f"scanner must provide callable {name}")

    if backend == "reference-like":
        strike = scanner.reference_like_strike_sampling(config.phi_min, config.phi_max)
        dip = scanner.reference_like_dip_sampling(config.theta_min, config.theta_max)
        strike_helper = scanner.reference_like_strike_sampling
        dip_helper = scanner.reference_like_dip_sampling
        refinement_factor = 1
    else:
        strike = scanner.refined_reference_like_strike_sampling(
            config.phi_min,
            config.phi_max,
            refinement_factor=config.refinement_factor,
        )
        dip = scanner.refined_reference_like_dip_sampling(
            config.theta_min,
            config.theta_max,
            refinement_factor=config.refinement_factor,
        )
        strike_helper = scanner.refined_reference_like_strike_sampling
        dip_helper = scanner.refined_reference_like_dip_sampling
        refinement_factor = config.refinement_factor

    implementation = (
        _scanner_instance_implementation_identity(scanner)
        if implementation_identity is None
        else _normalized_implementation_identity(implementation_identity)
    )
    evidence = {
        "backend": backend,
        "refinement_factor": refinement_factor,
        "dtype": np.dtype(np.float32).name,
        "strike": _sampling_axis_evidence(strike, "strike"),
        "dip": _sampling_axis_evidence(dip, "dip"),
        "orientation_count": int(len(strike) * len(dip)),
        "scanner_stage_implementation_identity": implementation,
        "sampling_source_implementation_identity": {
            "strike": _callable_implementation_identity(strike_helper),
            "dip": _callable_implementation_identity(dip_helper),
        },
    }
    return validate_scanner_sampling_evidence(
        evidence,
        config,
        backend,
        expected_implementation_identity=implementation,
    )


def validate_scanner_sampling_evidence(
    value: Mapping[str, Any],
    config: F3ScannerConfig,
    backend: F3ScannerBackend,
    *,
    expected_implementation_identity: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Validate and normalize persisted scanner sampling evidence."""

    if not isinstance(value, Mapping):
        raise TypeError("sampling evidence must be a mapping")
    if not isinstance(config, F3ScannerConfig):
        raise TypeError("config must be an F3ScannerConfig")
    _validate_backend(backend)
    if backend != config.backend:
        raise ValueError("sampling backend must match config.backend")
    expected_fields = {
        "backend",
        "refinement_factor",
        "dtype",
        "strike",
        "dip",
        "orientation_count",
        "scanner_stage_implementation_identity",
        "sampling_source_implementation_identity",
    }
    if set(value) != expected_fields:
        raise ValueError("sampling evidence field set mismatch")
    if value["backend"] != backend:
        raise ValueError("sampling evidence backend mismatch")
    expected_refinement = 1 if backend == "reference-like" else config.refinement_factor
    if value["refinement_factor"] != expected_refinement:
        raise ValueError("sampling evidence refinement factor mismatch")
    if value["dtype"] != np.dtype(np.float32).name:
        raise ValueError("sampling evidence dtype mismatch")

    strike = _validate_sampling_axis(value["strike"], "strike")
    dip = _validate_sampling_axis(value["dip"], "dip")
    _validate_sampling_bounds(strike, "strike", config.phi_min, config.phi_max)
    _validate_sampling_bounds(dip, "dip", config.theta_min, config.theta_max)
    orientation_count = value["orientation_count"]
    if (
        isinstance(orientation_count, bool)
        or not isinstance(orientation_count, int)
        or orientation_count != strike["count"] * dip["count"]
    ):
        raise ValueError("sampling evidence orientation count mismatch")

    stage_identity = value["scanner_stage_implementation_identity"]
    if not isinstance(stage_identity, (str, Mapping)):
        raise ValueError("sampling evidence stage implementation identity is invalid")
    if expected_implementation_identity is not None:
        expected_identity = _normalized_implementation_identity(expected_implementation_identity)
        if stage_identity != expected_identity:
            raise ValueError("sampling evidence stage implementation identity mismatch")

    source_identity = value["sampling_source_implementation_identity"]
    if not isinstance(source_identity, Mapping) or set(source_identity) != {"strike", "dip"}:
        raise ValueError("sampling source implementation identity field set mismatch")
    normalized_source = {
        name: _validate_sampling_helper_identity(source_identity[name], name)
        for name in ("strike", "dip")
    }
    return {
        "backend": backend,
        "refinement_factor": expected_refinement,
        "dtype": np.dtype(np.float32).name,
        "strike": strike,
        "dip": dip,
        "orientation_count": orientation_count,
        "scanner_stage_implementation_identity": (
            dict(stage_identity) if isinstance(stage_identity, Mapping) else stage_identity
        ),
        "sampling_source_implementation_identity": normalized_source,
    }


def sampling_count_from_evidence(value: Mapping[str, Any]) -> dict[str, int]:
    """Derive the legacy sampling-count view from validated evidence."""

    strike = value["strike"]
    dip = value["dip"]
    return {
        "strike": int(strike["count"]),
        "dip": int(dip["count"]),
        "orientations": int(value["orientation_count"]),
    }


def scanner_sampling_count(
    scanner: ScannerProtocol,
    config: F3ScannerConfig,
    backend: F3ScannerBackend,
) -> dict[str, int]:
    """Return scanner sampling counts derived from canonical evidence."""

    return sampling_count_from_evidence(scanner_sampling_evidence(scanner, config, backend))


def _sampling_axis_evidence(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} sampling must be a NumPy array")
    if value.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} sampling dtype must be float32")
    if value.ndim != 1 or value.size < 1:
        raise ValueError(f"{name} sampling must be a non-empty 1D array")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} sampling must contain only finite values")
    if value.size > 1 and not np.all(np.diff(value) > np.float32(0.0)):
        raise ValueError(f"{name} sampling must be strictly increasing")
    canonical = np.asarray(value, dtype="<f4")
    return {
        "count": int(canonical.size),
        "values": [float(item) for item in canonical],
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _validate_sampling_axis(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"count", "values", "sha256"}:
        raise ValueError(f"{name} sampling evidence field set mismatch")
    count = value["count"]
    values = value["values"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{name} sampling evidence count is invalid")
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values)
    ):
        raise ValueError(f"{name} sampling evidence values are invalid")
    array = np.asarray(values, dtype=np.float32)
    expected = _sampling_axis_evidence(array, name)
    if value["sha256"] != expected["sha256"]:
        raise ValueError(f"{name} sampling evidence digest mismatch")
    return expected


def _validate_sampling_helper_identity(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} sampling implementation identity must be a mapping")
    required = {"module", "qualname"}
    allowed = required | {
        "code_sha256",
        "immutable_closure_sha256",
        "source_sha256",
    }
    if not required <= set(value) or not set(value) <= allowed:
        raise ValueError(f"{name} sampling implementation identity field set mismatch")
    normalized: dict[str, str] = {}
    for field, item in value.items():
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name} sampling implementation identity is invalid")
        if field.endswith("sha256") and (
            len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError(f"{name} sampling implementation digest is invalid")
        normalized[field] = item
    return normalized


def _validate_sampling_bounds(
    axis: Mapping[str, Any],
    name: str,
    minimum: float,
    maximum: float,
) -> None:
    tolerance = (
        8.0
        * float(np.finfo(np.float32).eps)
        * max(
            1.0,
            abs(minimum),
            abs(maximum),
        )
    )
    values = axis["values"]
    if values[0] < minimum - tolerance or values[-1] > maximum + tolerance:
        raise ValueError(f"{name} sampling must be within the configured angle range")


def scanner_array_summary(array: np.ndarray) -> dict[str, Any]:
    """Return the canonical semantic summary for one finite scanner array."""

    array = np.asarray(array)
    if array.dtype.kind != "f" or array.dtype.itemsize != np.dtype(np.float32).itemsize:
        raise ValueError("scanner summary array must use float32 storage")
    count = nonzero_count(array)
    return {
        "shape": list(array.shape),
        "dtype": np.dtype(np.float32).name,
        "finite_count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "nonzero_epsilon": NONZERO_EPSILON,
        "nonzero_count": count,
        "nonzero_fraction": float(count / array.size),
    }


def _write_big_endian_dat(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as output:
        for start in range(0, values.shape[0], _WRITE_SLAB_COUNT):
            slab = values[start : start + _WRITE_SLAB_COUNT]
            np.asarray(slab, dtype=_DAT_DTYPE, order="C").tofile(output)


def _immutable_view(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise ValueError("scanner input must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"scanner input shape must be {shape}, got {array.shape}")
    if array.dtype != np.dtype("=f4"):
        raise ValueError(f"scanner input dtype must be native float32, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("scanner input must be C-contiguous")
    if not np.all(np.isfinite(array)):
        raise ValueError("scanner input must contain only finite values")
    readonly = memoryview(array).toreadonly()
    result = np.frombuffer(readonly, dtype=np.dtype("=f4")).reshape(shape)
    result.flags.writeable = False
    return result


def _result_from_stage(
    stage: F3StageResult,
    *,
    backend: F3ScannerBackend,
    shape: tuple[int, int, int],
    input_fingerprint: str,
    config: F3ScannerConfig,
    settings: Mapping[str, Any],
    input_bytes: int = 0,
) -> F3ScannerStageResult:
    report = _read_json(stage.path / "report.json")
    result = F3ScannerStageResult(
        backend=backend,
        path=stage.path,
        fingerprint=stage.fingerprint,
        reused=stage.reused,
        shape=shape,
        input_fingerprint=input_fingerprint,
        report=MappingProxyType(report),
        elapsed_seconds=0.0,
        input_bytes=int(input_bytes),
        output_bytes=sum(
            path.stat().st_size
            for path in stage.path.iterdir()
            if path.is_file() and path.name not in {"stage_manifest.json", "complete.json"}
        ),
    )
    _validate_report_identity(report, result)
    if report.get("resolved_config") != asdict(config):
        raise ValueError("scanner report mismatch: resolved_config")
    if report.get("resolved_stage_settings") != dict(settings):
        raise ValueError("scanner report mismatch: resolved_stage_settings")
    if report.get("requested_remove_edge_effects") != config.remove_edge_effects:
        raise ValueError("scanner report mismatch: requested_remove_edge_effects")
    if report.get("effective_remove_edge_effects") != config.effective_remove_edge_effects:
        raise ValueError("scanner report mismatch: effective_remove_edge_effects")
    return result


def _validate_report_identity(
    report: Mapping[str, Any],
    stage: F3ScannerStageResult,
) -> None:
    expected = {
        "fingerprint": stage.fingerprint,
        "backend": stage.backend,
        "shape": list(stage.shape),
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(f"scanner report mismatch: {name}")
    input_fingerprint = report.get("input_fingerprint")
    if (
        not isinstance(input_fingerprint, Mapping)
        or input_fingerprint.get("sha256") != stage.input_fingerprint
    ):
        raise ValueError("scanner report mismatch: input_fingerprint")


def _normalized_implementation_identity(
    value: Mapping[str, Any] | str | None,
) -> Mapping[str, Any] | str:
    if value is not None:
        if isinstance(value, str):
            if not value:
                raise ValueError("implementation_identity must not be empty")
            return value
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("implementation_identity must be a mapping, string, or None")
    modules = {
        identifier: {
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size": source.stat().st_size,
        }
        for identifier, source in sorted(_scanner_implementation_source_files().items())
    }
    return {
        "name": F3_SCANNER_STAGE_IMPLEMENTATION,
        "algorithm_modules": modules,
    }


def _scanner_implementation_identity(
    scanner_factory: ScannerFactory,
    declared_identity: Mapping[str, Any] | str | None,
) -> Mapping[str, Any] | str:
    if declared_identity is not None or scanner_factory is FaultOrientScanner3:
        return _normalized_implementation_identity(declared_identity)
    return {
        "name": F3_SCANNER_STAGE_IMPLEMENTATION,
        "algorithm": _normalized_implementation_identity(None),
        "scanner_factory": _callable_implementation_identity(scanner_factory),
    }


def _scanner_instance_implementation_identity(
    scanner: ScannerProtocol,
) -> Mapping[str, Any] | str:
    if type(scanner) is FaultOrientScanner3:
        return _normalized_implementation_identity(None)
    return {
        "name": F3_SCANNER_STAGE_IMPLEMENTATION,
        "scanner_class": _callable_implementation_identity(type(scanner)),
    }


def _scanner_implementation_source_files() -> dict[str, Path]:
    package_root = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__),
        package_root / "orient3d.py",
        package_root / "geometry.py",
        package_root / "interp.py",
        package_root / "thinning3d.py",
        *sorted((package_root / "_orient3d").glob("*.py")),
    )
    return {source.relative_to(package_root.parent).as_posix(): source for source in sources}


def _validate_workspace_and_input(
    workspace: F3RunWorkspace,
    input_identity: F3FileIdentity,
) -> None:
    if not isinstance(workspace, F3RunWorkspace):
        raise TypeError("workspace must be an F3RunWorkspace")
    if not isinstance(input_identity, F3FileIdentity):
        raise TypeError("input_identity must be an F3FileIdentity")
    if input_identity.role != "input" or input_identity.filename != "ep.dat":
        raise ValueError("input_identity must identify ep.dat with role 'input'")
    manifest_identity = _workspace_dataset_file_identity(workspace, "input")
    if manifest_identity != input_identity.computation_identity:
        raise F3WorkspaceMismatchError(
            "scanner input identity does not match the run manifest dataset identity"
        )


def _validated_shape(value: object) -> tuple[int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in value)
    ):
        raise ValueError("shape must contain three positive integers")
    return value


def _validate_backend(value: object) -> None:
    if value not in F3_SCANNER_BACKEND_ORDER:
        raise ValueError("backend must be 'reference-like' or 'quality'")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid scanner JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"scanner JSON artifact must contain an object: {path}")
    return value


def _open_dat(path: Path, shape: tuple[int, int, int]) -> np.memmap:
    return np.memmap(path, dtype=_DAT_DTYPE, mode="r", shape=shape, order="C")

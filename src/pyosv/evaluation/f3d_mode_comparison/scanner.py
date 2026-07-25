"""Shared full-volume scanner stages for the canonical F3 comparison."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from pyosv.orient3d import FaultOrientScanner3

from .artifacts import (
    F3RunWorkspace,
    F3StageArtifact,
    F3StageResult,
    canonical_json_bytes,
    stage_fingerprint,
    validate_stage,
)
from .config import F3ScannerConfig
from .data import F3FileIdentity, F3VolumeSource
from .models import F3ModeComparisonPlan, F3ScannerBackend

F3_SCANNER_STAGE_CONTRACT_VERSION = 1
F3_SCANNER_STAGE_IMPLEMENTATION = "pyosv-f3-scanner-stage-v1"
F3_SCANNER_BACKEND_ORDER: tuple[F3ScannerBackend, ...] = (
    "reference-like",
    "quality",
)
_COMMON_VOLUME_NAMES = ("ft", "pt", "tt", "fet", "fpt", "ftt")
_UNIT_RANGE_NAMES = {"ft", "fet", "confidence"}
_DAT_DTYPE = np.dtype(">f4")
_WRITE_SLAB_COUNT = 1

ScannerFactory = Callable[[float, float], FaultOrientScanner3]


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
) -> dict[str, Any]:
    """Return every scanner-stage control used by its cache fingerprint."""

    if not isinstance(config, F3ScannerConfig):
        raise TypeError("config must be an F3ScannerConfig")
    valid_shape = _validated_shape(shape)
    implementation = _normalized_implementation_identity(implementation_identity)
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
        "scanner_thinning": {
            "mode": config.scanner_thin_mode,
            "reference_sigma": config.reference_thin_sigma,
            "requested_remove_edge_effects": config.remove_edge_effects,
            "effective_remove_edge_effects": config.effective_remove_edge_effects,
        },
    }


def scanner_stage_fingerprint(
    workspace: F3RunWorkspace,
    input_identity: F3FileIdentity,
    config: F3ScannerConfig,
    *,
    implementation_identity: Mapping[str, Any] | str | None = None,
) -> str:
    """Build the exact fingerprint for one scanner backend stage."""

    _validate_workspace_and_input(workspace, input_identity)
    settings = scanner_stage_resolved_settings(
        config,
        input_identity.shape,
        implementation_identity=implementation_identity,
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
    order = tuple(backend_order)
    if len(order) != 2 or set(order) != set(F3_SCANNER_BACKEND_ORDER):
        raise ValueError("backend_order must contain reference-like and quality exactly once")

    input_identity = volume_source.identity.file_for("input")
    _validate_workspace_and_input(workspace, input_identity)
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
            settings = scanner_stage_resolved_settings(
                config,
                input_identity.shape,
                implementation_identity=implementation_identity,
            )
            artifacts = scanner_stage_artifacts(input_identity.shape, backend)
            fingerprint = scanner_stage_fingerprint(
                workspace,
                input_identity,
                config,
                implementation_identity=implementation_identity,
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
                    scanner_factory=scanner_factory,
                )

            stage = workspace.write_or_reuse_stage(
                "scanner",
                input_fingerprints={"ep.dat": input_identity.sha256},
                resolved_settings=settings,
                artifacts=artifacts,
                writer=writer,
                fingerprint=fingerprint,
            )
            results[backend] = _result_from_stage(
                stage,
                backend=backend,
                shape=input_identity.shape,
                input_fingerprint=input_identity.sha256,
                config=config,
                settings=settings,
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
    scanner_factory: ScannerFactory,
) -> None:
    scanner = scanner_factory(config.sigma1, config.sigma2)
    confidence: np.ndarray | None = None
    if backend == "reference-like":
        ft, pt, tt = scanner.scan(
            config.phi_min,
            config.phi_max,
            config.theta_min,
            config.theta_max,
            scanner_input,
            interpolation_backend=config.interpolation_backend,
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
        "sampling_count": _sampling_count(scanner, config, backend),
        "requested_remove_edge_effects": config.remove_edge_effects,
        "effective_remove_edge_effects": config.effective_remove_edge_effects,
        "raw": {name: _array_summary(values) for name, values in raw.items()},
        "thinned": {name: _array_summary(values) for name, values in thinned.items()},
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


def _sampling_count(
    scanner: FaultOrientScanner3,
    config: F3ScannerConfig,
    backend: F3ScannerBackend,
) -> dict[str, int]:
    if backend == "reference-like":
        strike = scanner.reference_like_strike_sampling(config.phi_min, config.phi_max)
        dip = scanner.reference_like_dip_sampling(config.theta_min, config.theta_max)
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
    return {
        "strike": int(len(strike)),
        "dip": int(len(dip)),
        "orientations": int(len(strike) * len(dip)),
    }


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "finite_count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "nonzero_count": int(np.count_nonzero(array)),
        "nonzero_fraction": float(np.count_nonzero(array) / array.size),
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

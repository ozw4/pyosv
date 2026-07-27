"""Numerical runtime identity for F3 full-volume comparison workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import PureWindowsPath
from typing import Any

import numpy as np

from pyosv import _accel

F3_RUNTIME_IDENTITY_SCHEMA_VERSION = 2

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
NUMBA_ENVIRONMENT_VARIABLES = (
    "NUMBA_DISABLE_JIT",
    "NUMBA_NUM_THREADS",
    "NUMBA_THREADING_LAYER",
    "NUMBA_CPU_NAME",
    "NUMBA_CPU_FEATURES",
)

_RUNTIME_IDENTITY_FIELDS = {
    "runtime_identity_schema_version",
    "python_implementation",
    "platform_system",
    "platform_machine",
    "byte_order",
    "requested_acceleration_mode",
    "pyosv_accel",
    "numba_available",
    "numba_version",
    "effective_acceleration_state",
    "thread_environment",
    "python_hash_seed",
    "numpy_disable_cpu_features",
    "numba_environment",
    "openblas_coretype",
    "numpy_build",
    "numpy_runtime_cpu",
    "numpy_runtime_blas",
    "scipy_build",
}
_ACCELERATION_MODES = {"auto", "off", "required"}
_ACCELERATION_STATES = {"numba_enabled", "python_only"}
_AVAILABILITY_STATUSES = {"available", "not_available"}
_RUNTIME_LIBRARY_FIELDS = {
    "implementation",
    "version",
    "threading_layer",
    "architecture",
    "effective_thread_count",
}
_SHA256_LENGTH = 64


def numerical_runtime_identity() -> dict[str, Any]:
    """Return the process identity that selects the numerical execution path."""

    numba_available = bool(_accel.NUMBA_AVAILABLE)
    numba_version: str | None = None
    if numba_available:
        try:
            import numba

            candidate = getattr(numba, "__version__", None)
        except ImportError:
            candidate = None
        if isinstance(candidate, str) and candidate:
            numba_version = candidate

    return validate_numerical_runtime_identity(
        {
            "runtime_identity_schema_version": F3_RUNTIME_IDENTITY_SCHEMA_VERSION,
            "python_implementation": platform.python_implementation(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "byte_order": sys.byteorder,
            "requested_acceleration_mode": _accel._ACCEL_MODE,
            "pyosv_accel": os.environ.get("PYOSV_ACCEL"),
            "numba_available": numba_available,
            "numba_version": numba_version,
            "effective_acceleration_state": ("numba_enabled" if numba_available else "python_only"),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
            },
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "numpy_disable_cpu_features": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
            "numba_environment": {
                name: os.environ.get(name) for name in NUMBA_ENVIRONMENT_VARIABLES
            },
            "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"),
            "numpy_build": _numpy_build_identity(),
            "numpy_runtime_cpu": _numpy_runtime_cpu_identity(),
            "numpy_runtime_blas": _numpy_runtime_blas_identity(),
            "scipy_build": _scipy_build_identity(),
        }
    )


def validate_numerical_runtime_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize an injected numerical runtime identity."""

    if not isinstance(identity, Mapping):
        raise ValueError("runtime identity must be an object")
    if set(identity) != _RUNTIME_IDENTITY_FIELDS:
        raise ValueError("runtime identity field set mismatch")

    schema_version = identity["runtime_identity_schema_version"]
    if type(schema_version) is not int or schema_version != F3_RUNTIME_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"runtime identity schema version must equal {F3_RUNTIME_IDENTITY_SCHEMA_VERSION}"
        )

    normalized: dict[str, Any] = {
        "runtime_identity_schema_version": schema_version,
    }
    for name in ("python_implementation", "platform_system", "platform_machine"):
        normalized[name] = _nonempty_nonpath_string(identity[name], f"runtime identity {name}")

    byte_order = identity["byte_order"]
    if not isinstance(byte_order, str) or byte_order not in {"little", "big"}:
        raise ValueError("runtime identity byte_order must be 'little' or 'big'")
    normalized["byte_order"] = byte_order

    requested_mode = identity["requested_acceleration_mode"]
    if not isinstance(requested_mode, str) or requested_mode not in _ACCELERATION_MODES:
        raise ValueError(
            "runtime identity requested_acceleration_mode must be auto, off, or required"
        )
    normalized["requested_acceleration_mode"] = requested_mode
    pyosv_accel = _optional_string(identity["pyosv_accel"], "runtime identity pyosv_accel")
    if pyosv_accel is not None and pyosv_accel not in _ACCELERATION_MODES:
        raise ValueError("runtime identity pyosv_accel must be auto, off, required, or null")
    if pyosv_accel is not None and pyosv_accel != requested_mode:
        raise ValueError("runtime identity pyosv_accel does not match requested mode")
    normalized["pyosv_accel"] = pyosv_accel

    numba_available = identity["numba_available"]
    if type(numba_available) is not bool:
        raise ValueError("runtime identity numba_available must be a bool")
    normalized["numba_available"] = numba_available

    numba_version = identity["numba_version"]
    if numba_version is not None:
        numba_version = _nonempty_nonpath_string(
            numba_version,
            "runtime identity numba_version",
        )
    if numba_available and numba_version is None:
        raise ValueError("runtime identity available Numba must have a non-null version")
    if not numba_available and numba_version is not None:
        raise ValueError("runtime identity unavailable Numba must have a null version")
    normalized["numba_version"] = numba_version

    acceleration_state = identity["effective_acceleration_state"]
    if not isinstance(acceleration_state, str) or acceleration_state not in _ACCELERATION_STATES:
        raise ValueError(
            "runtime identity effective_acceleration_state must be numba_enabled or python_only"
        )
    expected_state = "numba_enabled" if numba_available else "python_only"
    if acceleration_state != expected_state:
        raise ValueError("runtime identity acceleration state does not match numba availability")
    if requested_mode == "off" and numba_available:
        raise ValueError("runtime identity acceleration mode off cannot enable Numba")
    normalized["effective_acceleration_state"] = acceleration_state

    normalized["thread_environment"] = _validate_environment(
        identity["thread_environment"],
        THREAD_ENVIRONMENT_VARIABLES,
        "thread_environment",
    )
    normalized["python_hash_seed"] = _optional_string(
        identity["python_hash_seed"],
        "runtime identity python_hash_seed",
    )
    normalized["numpy_disable_cpu_features"] = _optional_string(
        identity["numpy_disable_cpu_features"],
        "runtime identity numpy_disable_cpu_features",
    )
    normalized["numba_environment"] = _validate_environment(
        identity["numba_environment"],
        NUMBA_ENVIRONMENT_VARIABLES,
        "numba_environment",
    )
    normalized["openblas_coretype"] = _optional_string(
        identity["openblas_coretype"],
        "runtime identity openblas_coretype",
    )

    normalized["numpy_build"] = _validate_digest_identity(identity["numpy_build"], "numpy_build")
    normalized["scipy_build"] = _validate_digest_identity(identity["scipy_build"], "scipy_build")
    normalized["numpy_runtime_cpu"] = _validate_cpu_identity(identity["numpy_runtime_cpu"])
    normalized["numpy_runtime_blas"] = _validate_blas_identity(identity["numpy_runtime_blas"])

    # Reject values that only imitate JSON primitives (for example NumPy scalars).
    json.dumps(normalized, allow_nan=False, sort_keys=True)
    return normalized


def validate_publication_runtime_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the fixed numerical environment required for publication runs."""

    normalized = validate_numerical_runtime_identity(identity)
    errors: list[str] = []
    if normalized["python_hash_seed"] != "0":
        errors.append("PYTHONHASHSEED must equal 0")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if normalized["thread_environment"][name] != "1":
            errors.append(f"{name} must equal 1")
    if normalized["pyosv_accel"] is None:
        errors.append("PYOSV_ACCEL must be explicitly set")
    for field in ("numpy_runtime_cpu", "numpy_runtime_blas", "scipy_build"):
        if normalized[field]["status"] != "available":
            errors.append(f"{field} must be available")
    libraries = normalized["numpy_runtime_blas"]["libraries"]
    if libraries is not None and any(
        library["effective_thread_count"] != 1 for library in libraries
    ):
        errors.append("NumPy runtime BLAS effective thread count must equal 1")
    if errors:
        raise ValueError("publication runtime contract violation: " + "; ".join(errors))
    return normalized


def _numpy_build_identity() -> dict[str, str | None]:
    return _build_digest_identity(np)


def _scipy_build_identity() -> dict[str, str | None]:
    try:
        import scipy
    except ImportError:
        return _unavailable_digest()
    return _build_digest_identity(scipy)


def _build_digest_identity(module: Any) -> dict[str, str | None]:
    unavailable = _unavailable_digest()
    try:
        config_module = getattr(module, "__config__", None)
        config = getattr(config_module, "CONFIG", None)
        if not isinstance(config, Mapping) or not config:
            get_info = getattr(config_module, "get_info", None)
            if not callable(get_info):
                return unavailable
            config = {
                name: info
                for name in (
                    "blas_opt_info",
                    "lapack_opt_info",
                    "openblas_info",
                    "blas_ilp64_opt_info",
                    "lapack_ilp64_opt_info",
                )
                if (info := get_info(name))
            }
        normalized = _normalize_build_value(config)
        if not isinstance(normalized, dict) or not normalized:
            return unavailable
        payload = _canonical_bytes(normalized)
    except Exception:
        # Build metadata is optional on supported NumPy and SciPy distributions.
        return unavailable
    return {
        "status": "available",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _numpy_runtime_cpu_identity() -> dict[str, Any]:
    try:
        try:
            from numpy._core import _multiarray_umath
        except ImportError:
            from numpy.core import _multiarray_umath

        raw = getattr(_multiarray_umath, "__cpu_features__", None)
        if not isinstance(raw, Mapping) or not raw:
            return {"status": "not_available", "features": None}
        features = sorted(
            name
            for name, enabled in raw.items()
            if isinstance(name, str) and type(enabled) is bool and enabled
        )
        if not features:
            return {"status": "not_available", "features": None}
        return {"status": "available", "features": features}
    except Exception:
        return {"status": "not_available", "features": None}


def _numpy_runtime_blas_identity() -> dict[str, Any]:
    """Return loaded BLAS/LAPACK identities without filesystem or process data."""

    try:
        from threadpoolctl import threadpool_info

        raw_libraries = threadpool_info()
    except (ImportError, RuntimeError):
        return {"status": "not_available", "libraries": None}
    if not isinstance(raw_libraries, list):
        return {"status": "not_available", "libraries": None}

    libraries: list[dict[str, Any]] = []
    for raw in raw_libraries:
        if not isinstance(raw, Mapping) or raw.get("user_api") not in {"blas", "lapack"}:
            continue
        implementation = raw.get("internal_api") or raw.get("prefix")
        thread_count = raw.get("num_threads")
        if not isinstance(implementation, str) or not implementation:
            continue
        if type(thread_count) is not int or thread_count < 1:
            continue
        libraries.append(
            {
                "implementation": implementation,
                "version": raw.get("version") if isinstance(raw.get("version"), str) else None,
                "threading_layer": (
                    raw.get("threading_layer")
                    if isinstance(raw.get("threading_layer"), str)
                    else None
                ),
                "architecture": (
                    raw.get("architecture") if isinstance(raw.get("architecture"), str) else None
                ),
                "effective_thread_count": thread_count,
            }
        )
    if not libraries:
        return {"status": "not_available", "libraries": None}
    libraries.sort(key=lambda item: _canonical_bytes(item))
    return {"status": "available", "libraries": libraries}


def _validate_environment(
    value: Any,
    names: tuple[str, ...],
    context: str,
) -> dict[str, str | None]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"runtime identity {context} field set mismatch")
    return {
        name: _optional_string(
            value[name],
            f"runtime identity {context} {name}",
        )
        for name in names
    }


def _validate_digest_identity(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"status", "sha256"}:
        raise ValueError(f"runtime identity {context} field set mismatch")
    status = value["status"]
    digest = value["sha256"]
    _validate_status(status, context)
    if status == "available":
        _validate_sha256(digest, f"runtime identity {context} sha256")
    elif digest is not None:
        raise ValueError(f"runtime identity unavailable {context} must have a null sha256")
    return {"status": status, "sha256": digest}


def _validate_cpu_identity(value: Any) -> dict[str, Any]:
    context = "numpy_runtime_cpu"
    if not isinstance(value, Mapping) or set(value) != {"status", "features"}:
        raise ValueError(f"runtime identity {context} field set mismatch")
    status = value["status"]
    features = value["features"]
    _validate_status(status, context)
    if status == "not_available":
        if features is not None:
            raise ValueError(f"runtime identity unavailable {context} must have null features")
        return {"status": status, "features": None}
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(item, str) or not item for item in features)
        or features != sorted(set(features))
    ):
        raise ValueError(f"runtime identity {context} features must be a sorted feature set")
    if any(_is_absolute_path(item) for item in features):
        raise ValueError(f"runtime identity {context} features must not contain paths")
    return {"status": status, "features": list(features)}


def _validate_blas_identity(value: Any) -> dict[str, Any]:
    context = "numpy_runtime_blas"
    if not isinstance(value, Mapping) or set(value) != {"status", "libraries"}:
        raise ValueError(f"runtime identity {context} field set mismatch")
    status = value["status"]
    libraries = value["libraries"]
    _validate_status(status, context)
    if status == "not_available":
        if libraries is not None:
            raise ValueError(f"runtime identity unavailable {context} must have null libraries")
        return {"status": status, "libraries": None}
    if not isinstance(libraries, list) or not libraries:
        raise ValueError(f"runtime identity available {context} must have libraries")
    normalized: list[dict[str, Any]] = []
    for library in libraries:
        if not isinstance(library, Mapping) or set(library) != _RUNTIME_LIBRARY_FIELDS:
            raise ValueError(f"runtime identity {context} library field set mismatch")
        thread_count = library["effective_thread_count"]
        if type(thread_count) is not int or thread_count < 1:
            raise ValueError(
                f"runtime identity {context} effective_thread_count must be a positive integer"
            )
        item = {
            "implementation": _nonempty_nonpath_string(
                library["implementation"], f"runtime identity {context} implementation"
            ),
            "version": _optional_nonpath_string(
                library["version"], f"runtime identity {context} version"
            ),
            "threading_layer": _optional_nonpath_string(
                library["threading_layer"], f"runtime identity {context} threading_layer"
            ),
            "architecture": _optional_nonpath_string(
                library["architecture"], f"runtime identity {context} architecture"
            ),
            "effective_thread_count": thread_count,
        }
        normalized.append(item)
    sorted_libraries = sorted(normalized, key=_canonical_bytes)
    if normalized != sorted_libraries:
        raise ValueError(f"runtime identity {context} libraries must be sorted")
    return {"status": status, "libraries": normalized}


def _validate_status(value: Any, context: str) -> None:
    if not isinstance(value, str) or value not in _AVAILABILITY_STATUSES:
        raise ValueError(f"runtime identity {context} status must be available or not_available")


def _unavailable_digest() -> dict[str, str | None]:
    return {"status": "not_available", "sha256": None}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _normalize_build_value(value: Any) -> Any:
    if value is None or type(value) in {bool, int, float, str}:
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("non-finite build value")
        if isinstance(value, str) and _is_absolute_path(value):
            return "<absolute-path>"
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            if not isinstance(key, str):
                raise TypeError("build keys must be strings")
            if _is_build_path_field(key):
                result[key] = "<normalized-build-path>"
            else:
                result[key] = _normalize_build_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_normalize_build_value(item) for item in value]
    raise TypeError("unsupported build value")


def _is_absolute_path(value: str) -> bool:
    return os.path.isabs(value) or PureWindowsPath(value).is_absolute()


def _is_build_path_field(name: str) -> bool:
    normalized = name.lower().replace(" ", "_")
    return "path" in normalized or "directory" in normalized or "dirs" in normalized


def _nonempty_nonpath_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    if _is_absolute_path(value):
        raise ValueError(f"{context} must not be an absolute path")
    return value


def _optional_nonpath_string(value: Any, context: str) -> str | None:
    value = _optional_string(value, context)
    if value is not None and _is_absolute_path(value):
        raise ValueError(f"{context} must not be an absolute path")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{context} must be a string or null")
    return value


def _validate_sha256(value: Any, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")


__all__ = [
    "F3_RUNTIME_IDENTITY_SCHEMA_VERSION",
    "NUMBA_ENVIRONMENT_VARIABLES",
    "THREAD_ENVIRONMENT_VARIABLES",
    "numerical_runtime_identity",
    "validate_numerical_runtime_identity",
    "validate_publication_runtime_identity",
]

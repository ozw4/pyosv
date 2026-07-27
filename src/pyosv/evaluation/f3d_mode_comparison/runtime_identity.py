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

F3_RUNTIME_IDENTITY_SCHEMA_VERSION = 1

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_RUNTIME_IDENTITY_FIELDS = {
    "runtime_identity_schema_version",
    "python_implementation",
    "platform_system",
    "platform_machine",
    "byte_order",
    "requested_acceleration_mode",
    "numba_available",
    "numba_version",
    "effective_acceleration_state",
    "thread_environment",
    "python_hash_seed",
    "numpy_build",
}
_ACCELERATION_MODES = {"auto", "off", "required"}
_ACCELERATION_STATES = {"numba_enabled", "python_only"}
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
            "numba_available": numba_available,
            "numba_version": numba_version,
            "effective_acceleration_state": ("numba_enabled" if numba_available else "python_only"),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
            },
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "numpy_build": _numpy_build_identity(),
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
        normalized[name] = _nonempty_string(identity[name], f"runtime identity {name}")

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

    numba_available = identity["numba_available"]
    if type(numba_available) is not bool:
        raise ValueError("runtime identity numba_available must be a bool")
    normalized["numba_available"] = numba_available

    numba_version = identity["numba_version"]
    if numba_version is not None:
        numba_version = _nonempty_string(
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

    thread_environment = identity["thread_environment"]
    if not isinstance(thread_environment, Mapping) or set(thread_environment) != set(
        THREAD_ENVIRONMENT_VARIABLES
    ):
        raise ValueError("runtime identity thread_environment field set mismatch")
    normalized["thread_environment"] = {
        name: _optional_string(
            thread_environment[name],
            f"runtime identity thread_environment {name}",
        )
        for name in THREAD_ENVIRONMENT_VARIABLES
    }
    normalized["python_hash_seed"] = _optional_string(
        identity["python_hash_seed"],
        "runtime identity python_hash_seed",
    )

    numpy_build = identity["numpy_build"]
    if not isinstance(numpy_build, Mapping) or set(numpy_build) != {"status", "sha256"}:
        raise ValueError("runtime identity numpy_build field set mismatch")
    build_status = numpy_build["status"]
    build_digest = numpy_build["sha256"]
    if not isinstance(build_status, str) or build_status not in {"available", "not_available"}:
        raise ValueError("runtime identity numpy_build status must be available or not_available")
    if build_status == "available":
        _validate_sha256(build_digest, "runtime identity numpy_build sha256")
    elif build_digest is not None:
        raise ValueError("runtime identity unavailable numpy_build must have a null sha256")
    normalized["numpy_build"] = {
        "status": build_status,
        "sha256": build_digest,
    }

    # Reject values that only imitate JSON primitives (for example NumPy scalars).
    json.dumps(normalized, allow_nan=False, sort_keys=True)
    return normalized


def _numpy_build_identity() -> dict[str, str | None]:
    unavailable = {"status": "not_available", "sha256": None}
    try:
        config = getattr(np.__config__, "CONFIG", None)
        if not isinstance(config, Mapping) or not config:
            get_info = getattr(np.__config__, "get_info", None)
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
        payload = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except Exception:
        # Build metadata is optional on supported NumPy distributions.
        return unavailable
    return {
        "status": "available",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _normalize_build_value(value: Any) -> Any:
    if value is None or type(value) in {bool, int, float, str}:
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("non-finite NumPy build value")
        if isinstance(value, str) and _is_absolute_path(value):
            return "<absolute-path>"
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            if not isinstance(key, str):
                raise TypeError("NumPy build keys must be strings")
            if _is_build_path_field(key):
                result[key] = "<normalized-build-path>"
            else:
                result[key] = _normalize_build_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_normalize_build_value(item) for item in value]
    raise TypeError("unsupported NumPy build value")


def _is_absolute_path(value: str) -> bool:
    return os.path.isabs(value) or PureWindowsPath(value).is_absolute()


def _is_build_path_field(name: str) -> bool:
    normalized = name.lower().replace(" ", "_")
    return "path" in normalized or "directory" in normalized or "dirs" in normalized


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
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
    "THREAD_ENVIRONMENT_VARIABLES",
    "numerical_runtime_identity",
    "validate_numerical_runtime_identity",
]

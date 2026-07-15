"""Optional acceleration backend helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_ACCEL_MODES = ("auto", "off", "required")
_ACCEL_MODE = os.environ.get("PYOSV_ACCEL", "auto").strip().lower()
if _ACCEL_MODE not in _ACCEL_MODES:
    raise ValueError(
        f"PYOSV_ACCEL must be one of {_ACCEL_MODES}, got {_ACCEL_MODE!r}",
    )

if _ACCEL_MODE == "off":
    NUMBA_AVAILABLE = False
    _numba_njit = None
else:
    try:
        from numba import njit as _numba_njit
    except ImportError as exc:
        if _ACCEL_MODE == "required":
            raise ImportError(
                "Numba is required when PYOSV_ACCEL=required; install Numba or set "
                f"PYOSV_ACCEL=auto/off. Original Numba import error: {exc}",
            ) from exc
        NUMBA_AVAILABLE = False
        _numba_njit = None
    else:
        NUMBA_AVAILABLE = True


def njit(*args: Any, **kwargs: Any) -> Any:
    """Return ``numba.njit`` when available, otherwise a no-op decorator."""
    if _numba_njit is not None:
        return _numba_njit(*args, **kwargs)

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorate(func: F) -> F:
        return func

    return decorate

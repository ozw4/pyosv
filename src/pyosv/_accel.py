"""Optional acceleration backend helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from numba import njit as _numba_njit
except ImportError:
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

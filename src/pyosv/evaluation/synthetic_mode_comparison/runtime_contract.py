"""Shared numeric policy for synthetic mode-comparison runtime algebra."""

from __future__ import annotations

RUNTIME_REL_TOL = 1.0e-9
RUNTIME_ABS_TOL = 1.0e-12


def runtime_exceeds(value: float, upper_bound: float) -> bool:
    """Return whether ``value`` exceeds ``upper_bound`` beyond canonical tolerance."""

    tolerance = max(
        RUNTIME_ABS_TOL,
        RUNTIME_REL_TOL * max(abs(value), abs(upper_bound)),
    )
    return value > upper_bound + tolerance


__all__ = ["RUNTIME_ABS_TOL", "RUNTIME_REL_TOL", "runtime_exceeds"]

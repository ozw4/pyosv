"""Geometry angle helpers."""

from __future__ import annotations

import numpy as np


def _as_scalar_or_array(phi):
    values = np.asarray(phi)
    return values, values.ndim == 0


def range360(phi):
    """Wrap angles to the half-open range [0, 360)."""
    values, is_scalar = _as_scalar_or_array(phi)
    wrapped = np.mod(values, 360.0)
    if is_scalar:
        return float(wrapped)
    return wrapped


def range180(phi):
    """Wrap angles to the closed range [-180, 180]."""
    values, is_scalar = _as_scalar_or_array(phi)
    wrapped360 = np.mod(values, 360.0)
    wrapped = np.where(wrapped360 > 180.0, wrapped360 - 360.0, wrapped360)
    wrapped = np.where((wrapped360 == 180.0) & (values < 0.0), -180.0, wrapped)
    if is_scalar:
        return float(wrapped)
    return wrapped

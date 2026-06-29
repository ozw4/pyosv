"""Geometry angle and vector helpers."""

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


def _vector3(vector):
    values = np.asarray(vector, dtype=np.float32)
    if values.shape != (3,):
        msg = "vector must have shape (3,)"
        raise ValueError(msg)
    return values


def fault_dip_vector_from_strike_and_dip(phi: float, theta: float) -> np.ndarray:
    """Return the fault dip vector for strike and dip angles in degrees."""
    p = np.deg2rad(phi)
    t = np.deg2rad(theta)
    cp = np.cos(p)
    sp = np.sin(p)
    ct = np.cos(t)
    st = np.sin(t)
    return np.array([st, ct * cp, -ct * sp], dtype=np.float32)


def fault_strike_vector_from_strike_and_dip(phi: float, theta: float) -> np.ndarray:
    """Return the fault strike vector for strike and dip angles in degrees."""
    del theta
    p = np.deg2rad(phi)
    cp = np.cos(p)
    sp = np.sin(p)
    return np.array([0.0, sp, cp], dtype=np.float32)


def fault_normal_vector_from_strike_and_dip(phi: float, theta: float) -> np.ndarray:
    """Return the fault normal vector for strike and dip angles in degrees."""
    p = np.deg2rad(phi)
    t = np.deg2rad(theta)
    cp = np.cos(p)
    sp = np.sin(p)
    ct = np.cos(t)
    st = np.sin(t)
    return np.array([-ct, st * cp, -st * sp], dtype=np.float32)


def fault_strike_from_dip_vector(u) -> float:
    """Return the fault strike angle in degrees for a fault dip vector."""
    _, u2, u3 = _vector3(u)
    if u2 == 0.0 and u3 == 0.0:
        msg = "dip vector is not vertical"
        raise ValueError(msg)
    return range360(np.rad2deg(np.arctan2(-u3, u2)))


def fault_dip_from_dip_vector(u) -> float:
    """Return the fault dip angle in degrees for a fault dip vector."""
    u1, _, _ = _vector3(u)
    return float(np.rad2deg(np.arcsin(u1)))


def fault_strike_from_strike_vector(v) -> float:
    """Return the fault strike angle in degrees for a fault strike vector."""
    _, v2, v3 = _vector3(v)
    return range360(np.rad2deg(np.arctan2(v2, v3)))


def fault_strike_from_normal_vector(w) -> float:
    """Return the fault strike angle in degrees for a fault normal vector."""
    _, w2, w3 = _vector3(w)
    if w2 == 0.0 and w3 == 0.0:
        msg = "normal vector is not vertical"
        raise ValueError(msg)
    return range360(np.rad2deg(np.arctan2(-w3, w2)))


def fault_dip_from_normal_vector(w) -> float:
    """Return the fault dip angle in degrees for a fault normal vector."""
    w1, _, _ = _vector3(w)
    return float(np.rad2deg(np.arccos(-w1)))


def cross_product(u, v) -> np.ndarray:
    """Return the OSV cross product of two 3-component vectors."""
    u1, u2, u3 = _vector3(u)
    v1, v2, v3 = _vector3(v)
    return np.array(
        [
            u3 * v2 - u2 * v3,
            u1 * v3 - u3 * v1,
            u2 * v1 - u1 * v2,
        ],
        dtype=np.float32,
    )

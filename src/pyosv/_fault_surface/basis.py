"""Cubic tensor basis and exact constrained least squares."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def basis_1d(values: np.ndarray, bounds: np.ndarray, count: int) -> np.ndarray:
    normalized = (values - bounds[0]) / (bounds[1] - bounds[0])
    interior = np.linspace(0.0, 1.0, count - 2)[1:-1]
    knots = np.r_[np.zeros(4), interior, np.ones(4)]
    return BSpline(knots, np.eye(count), 3, extrapolate=False)(normalized)


def basis_matrix(uv: np.ndarray, bounds: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    first = basis_1d(uv[:, 0], bounds[0], shape[0])
    second = basis_1d(uv[:, 1], bounds[1], shape[1])
    return (first[:, :, None] * second[:, None, :]).reshape(len(uv), shape[0] * shape[1])


def second_differences(count: int) -> np.ndarray:
    knots = np.r_[np.zeros(4), np.linspace(0.0, 1.0, count - 2)[1:-1], np.ones(4)]
    greville = np.array([np.mean(knots[index + 1 : index + 4]) for index in range(count)])
    slopes = np.diff(np.eye(count), axis=0) / np.diff(greville)[:, None]
    return (
        np.diff(slopes, axis=0) / ((greville[2:] - greville[:-2]) / 2)[:, None] / (count - 1) ** 2
    )


def curvature_penalty(shape: tuple[int, int]) -> np.ndarray:
    first = np.kron(second_differences(shape[0]), np.eye(shape[1]))
    second = np.kron(np.eye(shape[0]), second_differences(shape[1]))
    return np.vstack((first, second))


def solve_coefficients(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, int],
    smoothness: float,
    prior: np.ndarray | None = None,
    constraints: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    weighted = design * np.sqrt(weights[:, None])
    penalty = curvature_penalty(shape)
    hessian = weighted.T @ weighted + smoothness * penalty.T @ penalty
    gradient = design.T @ (weights * targets)
    if prior is not None:
        # The guide keeps unsupported regions anchored without creating holes.
        hessian += 0.03 * design.T @ design
        gradient += 0.03 * design.T @ (design @ prior.ravel())
    hessian += np.eye(hessian.shape[0]) * 1e-10
    if constraints is None or len(constraints[0]) == 0:
        coefficients = np.linalg.solve(hessian, gradient)
    else:
        coefficients = solve_hard_constraints(hessian, gradient, *constraints)
    return coefficients.reshape(shape)


def solve_hard_constraints(
    hessian: np.ndarray, gradient: np.ndarray, design: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    _, singular, right = np.linalg.svd(design, full_matrices=len(design) < design.shape[1])
    threshold = singular[0] * max(design.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular > threshold))
    particular = np.linalg.lstsq(design, targets, rcond=None)[0]
    if np.max(np.abs(design @ particular - targets)) > 1e-5:
        raise ValueError("Manual constraints conflict or exceed the spline's degrees of freedom")
    null = right[rank:].T
    if null.shape[1]:
        correction = np.linalg.solve(
            null.T @ hessian @ null, null.T @ (gradient - hessian @ particular)
        )
        particular = particular + null @ correction
    if np.max(np.abs(design @ particular - targets)) > 1e-5:
        raise ValueError("Manual constraints could not be represented exactly")
    return particular

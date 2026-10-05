"""Candidate-guided fitting and probability-free hard-constrained refitting."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pyosv._fault_surface.basis import basis_matrix, solve_coefficients
from pyosv._fault_surface.evidence import corridor_targets, sample_probability
from pyosv._fault_surface.geometry import (
    evaluate_graph,
    grid_coordinates,
    numerical_grid_shape,
    readonly,
    sample_fault_surface,
)
from pyosv._fault_surface.models import (
    FaultSurfaceFitConfig,
    FaultSurfaceFitResult,
    FaultSurfaceModel,
)
from pyosv._fault_surface.validation import (
    validate_config,
    validate_model,
    validate_points,
    validate_volume,
)


def candidate_model(
    points: np.ndarray, volume_shape: tuple[int, int, int], config: FaultSurfaceFitConfig
) -> FaultSurfaceModel:
    if len(points) < 4:
        raise ValueError("At least four candidate points spanning one surface are required")
    _, singular, right = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    if singular[1] < 1e-6:
        raise ValueError("Candidate points must span a surface, not a point or line")
    dependent = int(np.argmax(np.abs(right[-1])))
    axes = tuple(axis for axis in range(3) if axis != dependent)
    uv = points[:, axes]
    bounds = np.column_stack((uv.min(axis=0), uv.max(axis=0)))
    validate_single_graph(uv, points[:, dependent])
    if np.any(np.diff(bounds, axis=1) <= 1e-6):
        raise ValueError("Candidate domain must span two independent native axes")
    design = basis_matrix(uv, bounds, config.control_shape)
    coefficients = solve_coefficients(
        design,
        points[:, dependent],
        np.ones(len(points)),
        config.control_shape,
        config.smoothness,
    )
    residual = np.abs(design @ coefficients.ravel() - points[:, dependent])
    if np.max(residual) > max(1.0, config.search_radius):
        raise ValueError("Candidate is not a single smooth graph within the configured corridor")
    return FaultSurfaceModel(
        dependent_axis=dependent,
        independent_axes=axes,
        bounds=bounds,
        coefficients=coefficients,
        volume_shape=volume_shape,
        mesh_shape=config.mesh_shape,
        smoothness=config.smoothness,
        stick_count=config.stick_count,
        stick_point_count=config.stick_point_count,
    )


def validate_single_graph(uv: np.ndarray, dependent: np.ndarray) -> None:
    _, inverse = np.unique(np.round(uv, decimals=6), axis=0, return_inverse=True)
    low = np.full(int(inverse.max()) + 1, np.inf)
    high = np.full_like(low, -np.inf)
    np.minimum.at(low, inverse, dependent)
    np.maximum.at(high, inverse, dependent)
    if np.any(high - low > 0.5):
        raise ValueError("Candidate has multiple sheets at one graph coordinate; split the patch")


def manual_constraints(
    model: FaultSurfaceModel, constraints: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray] | None:
    if constraints is None:
        return None
    points = validate_points(constraints, model.volume_shape)
    if len(points) > 4096:
        raise ValueError("At most 4096 manual constraints are supported")
    uv = points[:, model.independent_axes]
    if np.any(uv < model.bounds[:, 0]) or np.any(uv > model.bounds[:, 1]):
        raise ValueError("Manual constraints must lie inside the fixed surface domain")
    return basis_matrix(uv, model.bounds, model.coefficients.shape), points[:, model.dependent_axis]


def fit_fault_surface(
    probability: np.ndarray,
    candidate_points: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    config: FaultSurfaceFitConfig | None = None,
    constraints: np.ndarray | None = None,
) -> FaultSurfaceFitResult:
    """Fit one complete cubic graph to DL support near the selected candidate.

    ``probability`` is a floating (iline, xline, sample) array. Candidate and
    manual points are finite (N, 3) sample/xline/iline indices. Only values
    sampled inside the candidate corridor are read and validated, allowing
    memory-mapped full surveys. Invalid-mask locations provide no evidence.
    The rectangular domain comes from the selected candidate's extent.

    The candidate fixes the graph axis and search corridor; its triangles,
    holes and density are not constraints. Branches and multiple sheets at
    identical graph coordinates are unsupported. Manual points are exact
    equality constraints, and impossible edits raise ValueError atomically.
    """
    config = config or FaultSurfaceFitConfig()
    validate_config(config)
    volume = validate_volume(probability, valid_mask)
    points = validate_points(candidate_points, volume.shape)
    guide_model = candidate_model(points, volume.shape, config)
    uv = grid_coordinates(guide_model.bounds, numerical_grid_shape(config.control_shape))
    guide = evaluate_graph(guide_model, uv)
    targets, weights = corridor_targets(volume, valid_mask, guide_model, guide, config)
    design = basis_matrix(uv, guide_model.bounds, config.control_shape)
    coefficients = solve_coefficients(
        design,
        targets,
        weights,
        config.control_shape,
        config.smoothness,
        prior=guide_model.coefficients,
        constraints=manual_constraints(guide_model, constraints),
    )
    result = sample_fault_surface(replace(guide_model, coefficients=coefficients))
    values, valid = sample_probability(volume, result.vertices, valid_mask)
    return replace(
        result,
        probability=readonly(values, np.float32),
        supported=readonly(valid & (values >= config.support_threshold)),
    )


def refit_fault_surface(
    model: FaultSurfaceModel, constraints: np.ndarray | None = None
) -> FaultSurfaceFitResult:
    """Refit a prior graph under exact manual points without rerunning DL.

    The objective preserves the supplied model and penalizes curvature. Pass
    a fixed baseline and the accumulated accepted manual points to avoid
    history-dependent drift. An empty constraint set returns the unchanged
    model. Coefficients, mesh, and controls supplied by callers are not mutated.
    """
    validate_model(model)
    hard = manual_constraints(model, constraints)
    if hard is None or len(hard[0]) == 0:
        return sample_fault_surface(model)
    uv = grid_coordinates(model.bounds, numerical_grid_shape(model.coefficients.shape))
    design = basis_matrix(uv, model.bounds, model.coefficients.shape)
    coefficients = solve_coefficients(
        design,
        design @ model.coefficients.ravel(),
        np.ones(len(uv)),
        model.coefficients.shape,
        model.smoothness,
        constraints=hard,
    )
    return sample_fault_surface(replace(model, coefficients=coefficients))

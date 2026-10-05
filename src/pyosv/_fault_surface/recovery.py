"""Local DL evidence for candidates that do not determine a surface plane."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from pyosv._fault_surface.evidence import sample_probability
from pyosv._fault_surface.models import FaultSurfaceFitConfig


def needs_evidence_plane(points: np.ndarray) -> bool:
    if len(points) < 4:
        return True
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return bool(singular[1] < 1e-6)


def local_supported_component(
    probability: np.ndarray,
    valid_mask: np.ndarray | None,
    center: np.ndarray,
    config: FaultSurfaceFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    # Plane orientation is local; a larger search corridor must not cause a
    # survey-sized allocation or merge remote branches during initialization.
    radius = min(config.search_radius, 16.0)
    maximum = np.asarray(probability.shape[::-1]) - 1
    low = np.maximum(np.ceil(center - radius), 0).astype(int)
    high = np.minimum(np.floor(center + radius), maximum).astype(int)
    shape = high - low + 1
    if np.any(shape < 1):
        raise ValueError("No valid prediction support can establish a local surface plane")
    coordinates = np.indices(tuple(shape)).reshape(3, -1).T + low
    inside = np.sum((coordinates - center) ** 2, axis=1) <= radius**2 + 1e-9
    values = np.zeros(len(coordinates))
    valid = np.zeros(len(coordinates), dtype=bool)
    values[inside], valid[inside] = sample_probability(probability, coordinates[inside], valid_mask)
    supported = valid & (values >= config.support_threshold)
    components, count = label(supported.reshape(tuple(shape)), structure=np.ones((3, 3, 3)))
    if count == 0:
        raise ValueError("No valid prediction support can establish a local surface plane")
    components = components.ravel()
    distances = np.full(count + 1, np.inf)
    np.minimum.at(
        distances, components[supported], np.sum((coordinates[supported] - center) ** 2, axis=1)
    )
    closest = np.flatnonzero(np.isclose(distances, distances.min(), rtol=0, atol=1e-9))
    if len(closest) != 1:
        raise ValueError("Ambiguous prediction bands are equally near the candidate seed")
    selected = components == closest[0]
    return coordinates[selected].astype(np.float64), values[selected] ** 2


def recover_candidate_plane(
    probability: np.ndarray,
    valid_mask: np.ndarray | None,
    points: np.ndarray,
    config: FaultSurfaceFitConfig,
) -> tuple[np.ndarray, int]:
    """Return a bounded evidence-plane rectangle retaining the seed extent."""
    if len(points) == 0:
        raise ValueError("At least one candidate point is required")
    center = points[np.argmin(np.sum((points - np.median(points, axis=0)) ** 2, axis=1))]
    supported, weights = local_supported_component(probability, valid_mask, center, config)
    centroid = np.average(supported, weights=weights, axis=0)
    centered = supported - centroid
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues[1] < 0.25 or eigenvalues[1] < 4 * max(eigenvalues[0], 1e-12):
        raise ValueError(
            "Local prediction support does not establish a two-dimensional surface plane"
        )
    normal = vectors[:, 0]
    for dependent in np.argsort(-np.abs(normal), kind="stable"):
        if abs(normal[dependent]) < 1e-6:
            continue
        recovered = plane_rectangle(points, supported, centroid, normal, int(dependent), config)
        if recovered is None:
            continue
        maximum = np.asarray(probability.shape[::-1]) - 1
        if np.all(recovered >= -1e-9) and np.all(recovered <= maximum + 1e-9):
            return recovered, int(dependent)
    raise ValueError("Local prediction plane has no supported rectangular domain inside the volume")


def plane_rectangle(
    points: np.ndarray,
    supported: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
    dependent: int,
    config: FaultSurfaceFitConfig,
) -> np.ndarray | None:
    axes = tuple(axis for axis in range(3) if axis != dependent)
    slopes = -normal[list(axes)] / normal[dependent]
    seed_prediction = centroid[dependent] + (points[:, axes] - centroid[list(axes)]) @ slopes
    if np.max(np.abs(points[:, dependent] - seed_prediction)) > max(config.search_radius, 1.0):
        return None
    extent = np.vstack((points[:, axes], supported[:, axes]))
    low, high = extent.min(axis=0), extent.max(axis=0)
    uv = np.array([[low[0], low[1]], [low[0], high[1]], [high[0], low[1]], [high[0], high[1]]])
    recovered = np.empty((4, 3), dtype=np.float64)
    recovered[:, axes] = uv
    recovered[:, dependent] = centroid[dependent] + (uv - centroid[list(axes)]) @ slopes
    return recovered

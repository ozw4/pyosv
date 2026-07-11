"""Experimental boundary-aware seed selection for synthetic-quality studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from pyosv.cells import FaultCell
from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.voting3d import OptimalSurfaceVoter


class VotingConfig(Protocol):
    ru: int
    rv: int
    rw: int
    seed_distance: int
    seed_threshold: float


@dataclass(frozen=True, slots=True)
class BoundarySeedSelectionResult:
    """Seeds and JSON-safe diagnostics produced by an experimental selector."""

    default_seeds: tuple[FaultCell, ...]
    selected_seeds: tuple[FaultCell, ...]
    diagnostics: dict[str, Any]


def select_boundary_seed_retention_v1(
    *,
    voting_config: VotingConfig,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    target: np.ndarray,
    target_source: str,
    edge_margin: int,
) -> BoundarySeedSelectionResult:
    """Select default seeds plus deterministic boundary-target seeds."""

    ft_array = np.asarray(ft, dtype=np.float32)
    pt_array = np.asarray(pt, dtype=np.float32)
    tt_array = np.asarray(tt, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if ft_array.ndim != 3:
        raise ValueError("ft must be a 3D array")
    if (
        pt_array.shape != ft_array.shape
        or tt_array.shape != ft_array.shape
        or target_array.shape != ft_array.shape
    ):
        raise ValueError("boundary_seed_retention_v1 input shapes must match")
    if not (
        np.all(np.isfinite(ft_array))
        and np.all(np.isfinite(pt_array))
        and np.all(np.isfinite(tt_array))
        and np.all(np.isfinite(target_array))
    ):
        raise ValueError("boundary_seed_retention_v1 inputs must contain only finite values")

    voter = OptimalSurfaceVoter(ru=voting_config.ru, rv=voting_config.rv, rw=voting_config.rw)
    default_seeds = voter.pick_seeds(
        d=voting_config.seed_distance,
        fm=voting_config.seed_threshold,
        ft=ft_array,
        pt=pt_array,
        tt=tt_array,
    )
    edge_shell = quality_metrics.edge_mask(ft_array.shape, edge_margin)
    edge_ft = ft_array[edge_shell]
    edge_ft_max = float(np.max(edge_ft)) if edge_ft.size else 0.0
    ft_threshold = min(float(voting_config.seed_threshold), 0.5 * edge_ft_max)
    boundary_candidate = (
        edge_shell
        & quality_metrics.positive_candidate_mask(target_array)
        & (ft_array > np.float32(ft_threshold))
    )
    boundary_candidate_count = int(np.count_nonzero(boundary_candidate))
    existing_coordinates = {(seed.i1, seed.i2, seed.i3) for seed in default_seeds}
    added_coordinates: set[tuple[int, int, int]] = set()
    added_seeds: list[FaultCell] = []
    distance = max(0, int(voting_config.seed_distance))
    candidate_records = []
    for index in np.argwhere(boundary_candidate):
        i3, i2, i1 = (int(index[0]), int(index[1]), int(index[2]))
        candidate_records.append(
            (
                -float(target_array[i3, i2, i1]),
                -float(ft_array[i3, i2, i1]),
                int(np.ravel_multi_index((i3, i2, i1), ft_array.shape)),
                i1,
                i2,
                i3,
            )
        )
    for _, _, _, i1, i2, i3 in sorted(candidate_records):
        coordinate = (i1, i2, i3)
        if coordinate in existing_coordinates:
            continue
        if any(
            abs(i1 - a1) <= distance and abs(i2 - a2) <= distance and abs(i3 - a3) <= distance
            for a1, a2, a3 in added_coordinates
        ):
            continue
        added_coordinates.add(coordinate)
        added_seeds.append(
            FaultCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            )
        )

    retained_seeds = [*default_seeds, *added_seeds]
    added_target_values = np.asarray(
        [target_array[seed.i3, seed.i2, seed.i1] for seed in added_seeds], dtype=np.float32
    )
    added_seed_mask = np.zeros(ft_array.shape, dtype=bool)
    for seed in added_seeds:
        added_seed_mask[seed.i3, seed.i2, seed.i1] = True
    diagnostics = {
        "enabled": True,
        "target_source": target_source,
        "edge_margin": int(edge_margin),
        "default_seed_count": int(len(default_seeds)),
        "boundary_candidate_count": boundary_candidate_count,
        "added_seed_count": int(len(added_seeds)),
        "total_seed_count": int(len(retained_seeds)),
        "added_seed_edge_shell_fraction": quality_metrics.edge_candidate_fraction(
            added_seed_mask, edge_margin=edge_margin
        ),
        "added_seed_target_mean": (
            float(np.mean(added_target_values)) if added_target_values.size else 0.0
        ),
        "added_seed_target_p95": (
            float(np.percentile(added_target_values, 95)) if added_target_values.size else 0.0
        ),
    }
    return BoundarySeedSelectionResult(tuple(default_seeds), tuple(retained_seeds), diagnostics)

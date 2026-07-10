"""Case registry for synthetic quality evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pyosv.synthetic3d import (
    Synthetic3DCase,
    make_boundary_plane_case,
    make_crossing_planes_case,
    make_curved_surface_case,
    make_parallel_planes_case,
    make_single_dipping_plane_case,
    make_single_vertical_plane_case,
    make_weak_noisy_plane_case,
)


@dataclass(frozen=True, slots=True)
class SyntheticQualityCaseDefinition:
    """A controlled synthetic report case definition."""

    case_id: str
    factory: Callable[[tuple[int, int, int]], Synthetic3DCase]


MINIMAL_CASES = (
    SyntheticQualityCaseDefinition(
        case_id="single_vertical_plane",
        factory=make_single_vertical_plane_case,
    ),
)
GEOMETRY_CASES = (
    *MINIMAL_CASES,
    SyntheticQualityCaseDefinition(
        case_id="single_dipping_plane",
        factory=make_single_dipping_plane_case,
    ),
    SyntheticQualityCaseDefinition(
        case_id="curved_surface",
        factory=make_curved_surface_case,
    ),
)
EXTENDED_CASES = (
    *GEOMETRY_CASES,
    SyntheticQualityCaseDefinition("parallel_planes", make_parallel_planes_case),
    SyntheticQualityCaseDefinition("crossing_planes", make_crossing_planes_case),
    SyntheticQualityCaseDefinition("boundary_plane", make_boundary_plane_case),
    SyntheticQualityCaseDefinition("weak_noisy_plane", make_weak_noisy_plane_case),
)
CASE_SETS = {
    "minimal": MINIMAL_CASES,
    "geometry": GEOMETRY_CASES,
    "extended": EXTENDED_CASES,
}
CASE_IDS = tuple(definition.case_id for definition in EXTENDED_CASES)


def validate_case_set(case_set: str) -> tuple[SyntheticQualityCaseDefinition, ...]:
    """Return the registered definitions for a case set."""

    try:
        return CASE_SETS[case_set]
    except KeyError as error:
        raise ValueError(f"unknown case_set: {case_set}") from error


def validate_case_ids(
    case_ids: Sequence[str],
    *,
    description: str = "",
    sequence_name: str = "case_ids",
) -> tuple[str, ...]:
    """Validate a non-empty, unique sequence of registered case IDs."""

    valid_case_ids = tuple(case_ids)
    if not valid_case_ids:
        raise ValueError(f"{sequence_name} must include at least one case ID")
    unknown = sorted(set(valid_case_ids).difference(CASE_IDS))
    qualifier = f"{description} " if description else ""
    if unknown:
        raise ValueError(
            f"unknown {qualifier}case ID(s): {','.join(unknown)}; choices: {','.join(CASE_IDS)}"
        )
    duplicates = {case_id for case_id in valid_case_ids if valid_case_ids.count(case_id) > 1}
    if duplicates:
        raise ValueError(f"duplicate {qualifier}case ID(s): {','.join(sorted(duplicates))}")
    return valid_case_ids

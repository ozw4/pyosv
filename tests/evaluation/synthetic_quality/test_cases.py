from __future__ import annotations

import pytest

from pyosv.evaluation.synthetic_quality.cases import (
    CASE_IDS,
    CASE_SETS,
    EXTENDED_CASES,
    GEOMETRY_CASES,
    MINIMAL_CASES,
    validate_case_ids,
    validate_case_set,
)


EXPECTED_CASE_IDS = (
    "single_vertical_plane",
    "single_dipping_plane",
    "curved_surface",
    "parallel_planes",
    "crossing_planes",
    "boundary_plane",
    "weak_noisy_plane",
)


def test_case_sets_preserve_registry_order_and_size() -> None:
    assert tuple(CASE_SETS) == ("minimal", "geometry", "extended")
    assert tuple(definition.case_id for definition in MINIMAL_CASES) == EXPECTED_CASE_IDS[:1]
    assert tuple(definition.case_id for definition in GEOMETRY_CASES) == EXPECTED_CASE_IDS[:3]
    assert tuple(definition.case_id for definition in EXTENDED_CASES) == EXPECTED_CASE_IDS
    assert CASE_IDS == EXPECTED_CASE_IDS


@pytest.mark.parametrize("definition", EXTENDED_CASES, ids=CASE_IDS)
def test_case_factory_returns_registered_id(definition) -> None:
    case = definition.factory((17, 17, 17))

    assert case.case_id == definition.case_id
    assert case.ft_oracle.shape == (17, 17, 17)


def test_validate_case_set_rejects_unknown_set() -> None:
    with pytest.raises(ValueError, match=r"^unknown case_set: missing$"):
        validate_case_set("missing")


def test_validate_case_ids_rejects_unknown_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unknown case ID"):
        validate_case_ids(("missing",))
    with pytest.raises(ValueError, match="duplicate case ID"):
        validate_case_ids((CASE_IDS[0], CASE_IDS[0]))

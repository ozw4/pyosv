from __future__ import annotations

import pytest

from pyosv.cells import FaultCell
from pyosv.evaluation.f3d_mode_comparison.reskin_policy_comparison import (
    _validate_reported_link_topology,
)
from pyosv.evaluation.f3d_mode_comparison.skin_artifacts import (
    ParsedSkinArtifacts,
    SkinCellRecord,
)
from pyosv.skin import FaultSkin
from pyosv.synthetic_metrics import skin_link_topology_metrics, skin_topology_metrics


def _parsed_skin_artifact(*skin_sizes: int) -> ParsedSkinArtifacts:
    cells = []
    index = 0
    for skin_size in skin_sizes:
        skin = []
        for _ in range(skin_size):
            skin.append(
                SkinCellRecord(
                    float(index),
                    0.0,
                    0.0,
                    index,
                    0,
                    0,
                    0.8,
                    0.0,
                    90.0,
                    "grown",
                    None,
                )
            )
            index += 1
        cells.append(tuple(skin))
    return ParsedSkinArtifacts(skins=tuple(cells), format_version=2)


def _link_metrics(
    *,
    component_count: int = 1,
    isolated_cell_count: int = 0,
    quad_candidate_count: int = 0,
    quad_match_count: int = 0,
    quad_mismatch_count: int = 0,
) -> dict[str, int]:
    return {
        "reciprocal_link_violation_count": 0,
        "cross_skin_link_count": 0,
        "self_link_count": 0,
        "linked_component_count": component_count,
        "isolated_cell_count": isolated_cell_count,
        "quad_closure_candidate_count": quad_candidate_count,
        "quad_closure_match_count": quad_match_count,
        "quad_closure_mismatch_count": quad_mismatch_count,
    }


@pytest.mark.parametrize(
    ("skin_sizes", "component_count", "isolated_cell_count"),
    (
        ((), 0, 0),  # empty
        ((1, 4), 5, 5),  # all cells isolated across two containers
        ((5,), 1, 0),  # one linked component
        ((1, 2, 5), 4, 2),  # multiple containers and mixed graph
    ),
)
def test_shallow_link_topology_graph_algebra_accepts_valid_cases(
    skin_sizes: tuple[int, ...],
    component_count: int,
    isolated_cell_count: int,
) -> None:
    parsed = _parsed_skin_artifact(*skin_sizes)
    expected_topology = {
        "cell_count": parsed.cell_count,
        "skin_count": len(parsed.skins),
    }
    _validate_reported_link_topology(
        _link_metrics(
            component_count=component_count,
            isolated_cell_count=isolated_cell_count,
        ),
        parsed=parsed,
        expected_topology=expected_topology,
    )


@pytest.mark.parametrize(
    ("fallback_used", "reskin_policy"),
    (
        (False, "existing_cells_v1"),
        (False, "reference_dense_v1"),
        (True, "existing_cells_v1"),
        (True, "reference_dense_v1"),
    ),
)
def test_shallow_graph_algebra_does_not_derive_counts_from_provenance(
    fallback_used: bool,
    reskin_policy: str,
) -> None:
    parsed = _parsed_skin_artifact(1, 2, 5)
    _validate_reported_link_topology(
        _link_metrics(component_count=4, isolated_cell_count=2),
        parsed=parsed,
        expected_topology={"cell_count": 8, "skin_count": 3},
    )
    assert isinstance(fallback_used, bool)
    assert reskin_policy in {"existing_cells_v1", "reference_dense_v1"}


@pytest.mark.parametrize(
    ("parsed", "metrics", "message"),
    (
        (_parsed_skin_artifact(2, 3), _link_metrics(component_count=1), "component bounds"),
        (_parsed_skin_artifact(5), _link_metrics(component_count=6), "component bounds"),
        (
            _parsed_skin_artifact(5),
            _link_metrics(component_count=2, isolated_cell_count=3),
            "isolated bounds",
        ),
        (_parsed_skin_artifact(1, 4), _link_metrics(component_count=5), "isolated bounds"),
        (_parsed_skin_artifact(), _link_metrics(component_count=1), "empty link topology"),
        (_parsed_skin_artifact(5), _link_metrics(component_count=3), "component size algebra"),
        (
            _parsed_skin_artifact(5),
            _link_metrics(quad_candidate_count=2, quad_match_count=1),
            "quad closure counts mismatch",
        ),
        (
            _parsed_skin_artifact(5),
            _link_metrics(quad_candidate_count=6, quad_match_count=6),
            "quad closure candidate count exceeds cells",
        ),
        (
            _parsed_skin_artifact(5),
            {
                **_link_metrics(),
                "reciprocal_link_violation_count": 1,
            },
            "safety violation",
        ),
        (
            _parsed_skin_artifact(5),
            {**_link_metrics(), "linked_component_count": True},
            "non-negative integers",
        ),
    ),
)
def test_shallow_link_topology_graph_algebra_rejects_invalid_cases(
    parsed: ParsedSkinArtifacts,
    metrics: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_reported_link_topology(
            metrics,
            parsed=parsed,
            expected_topology={
                "cell_count": parsed.cell_count,
                "skin_count": len(parsed.skins),
            },
        )


@pytest.mark.parametrize(
    ("fallback_used", "reskin_policy"),
    (
        (True, "existing_cells_v1"),
        (True, "reference_dense_v1"),
        (False, "existing_cells_v1"),
        (False, "reference_dense_v1"),
    ),
)
def test_live_linked_skin_metrics_pass_shallow_graph_validation(
    fallback_used: bool,
    reskin_policy: str,
) -> None:
    left = FaultCell(1, 1, 1, 0.8, 0.0, 90.0)
    right = FaultCell(2, 1, 1, 0.8, 0.0, 90.0)
    isolated = FaultCell(3, 1, 1, 0.8, 0.0, 90.0)
    object.__setattr__(left, "cr", right)
    object.__setattr__(right, "cl", left)
    skins = (FaultSkin.from_cells((left, right, isolated)),)

    links = skin_link_topology_metrics(skins)
    parsed = _parsed_skin_artifact(3)
    expected_topology = skin_topology_metrics(
        skins,
        (4, 4, 4),
        small_skin_size=10,
    )

    assert links["linked_component_count"] == 2
    assert links["isolated_cell_count"] == 1
    assert links["linked_component_count"] < expected_topology["cell_count"]
    assert links["isolated_cell_count"] < expected_topology["cell_count"]
    _validate_reported_link_topology(
        links,
        parsed=parsed,
        expected_topology=expected_topology,
    )
    assert isinstance(fallback_used, bool)
    assert reskin_policy in {"existing_cells_v1", "reference_dense_v1"}

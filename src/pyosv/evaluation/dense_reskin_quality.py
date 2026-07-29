"""Controlled evidence and promotion gate for the dense reskin policy."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pyosv._skinner.models import _ReskinContext, _SkinCell
from pyosv._skinner.occupancy import _SkinOccupancyMask
from pyosv._skinner.reskin import (
    RESKIN_POLICY_EXISTING_CELLS_V1,
    RESKIN_POLICY_REFERENCE_DENSE_V1,
    _empty_reskin_diagnostics,
    _reskin_reference,
    _reskin_reference_dense_v1,
)
from pyosv._skinner.transforms import _local_index_to_world, _update_transform_map
from pyosv.cells import FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED, FaultCell
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.experimental.boundary_skinning import find_synthetic_skins
from pyosv.skin import FaultSkin
from pyosv.synthetic_metrics import (
    reskin_generation_metrics,
    reskin_truth_correspondence_metrics,
    rounded_duplicate_cell_count,
    skin_link_topology_metrics,
    skin_topology_metrics,
)

DENSE_RESKIN_GATE_SCHEMA_VERSION = 1
DENSE_RESKIN_CANDIDATE = "quality_reference_dense_v1"
DENSE_RESKIN_BASELINE = "quality_existing_cells_v1"


@dataclass(frozen=True, slots=True)
class DenseReskinSurface:
    """One known local surface and its deliberately incomplete observations."""

    origin: tuple[float, float, float]
    truth_keys: tuple[tuple[int, int], ...]
    observed_keys: tuple[tuple[int, int], ...]
    likelihood: float = 0.9
    strike: float = 0.0
    dip: float = 90.0
    u_by_key: Mapping[tuple[int, int], float] | None = None
    v_scale: float = 1.0
    w_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class DenseReskinCase:
    """Skin-phase-only controlled input shared by baseline and candidate."""

    case_id: str
    shape: tuple[int, int, int]
    surfaces: tuple[DenseReskinSurface, ...]
    truth_surface_mask: np.ndarray
    truth_fault_id: np.ndarray
    truth_strike: np.ndarray
    truth_dip: np.ndarray
    truth_hole_mask: np.ndarray
    valid_mask: np.ndarray | None = None
    prior_occupancy_mask: np.ndarray | None = None


def controlled_dense_reskin_cases() -> tuple[DenseReskinCase, ...]:
    """Build the deterministic dense-reskin acceptance set."""

    shape = (21, 21, 21)
    three = tuple((iv, iw) for iw in range(4, 7) for iv in range(4, 7))
    four = tuple((iv, iw) for iw in range(4, 8) for iv in range(4, 8))
    center_hole = ((5, 5),)
    block_hole = ((5, 5), (5, 6), (6, 5), (6, 6))
    boundary = tuple((iv, iw) for iw in range(4, 8) for iv in range(3, 7))
    boundary_holes = ((3, 5), (4, 5))
    boundary_u = {key: 1.0 if key[0] == 3 else (2.0 if key == (4, 5) else 3.0) for key in boundary}

    definitions: list[tuple[str, tuple[DenseReskinSurface, ...], np.ndarray | None]] = [
        (
            "plane_3x3_center_hole",
            (
                DenseReskinSurface(
                    (7.0, 7.0, 7.0),
                    three,
                    tuple(key for key in three if key not in center_hole),
                ),
            ),
            None,
        ),
        (
            "plane_4x4_internal_2x2_hole",
            (
                DenseReskinSurface(
                    (7.0, 7.0, 7.0),
                    four,
                    tuple(key for key in four if key not in block_hole),
                ),
            ),
            None,
        ),
        (
            "low_support_gap",
            (
                DenseReskinSurface(
                    (7.0, 7.0, 7.0),
                    ((4, 4), (5, 4), (6, 4)),
                    ((4, 4), (6, 4)),
                    likelihood=0.2,
                ),
            ),
            None,
        ),
        (
            "dipping_surface_internal_hole",
            (
                DenseReskinSurface(
                    (7.0, 7.0, 7.0),
                    four,
                    tuple(key for key in four if key != (6, 6)),
                    dip=75.0,
                    u_by_key={key: 3.0 + 0.15 * (key[0] - 4) for key in four},
                ),
            ),
            None,
        ),
        (
            "parallel_surfaces",
            (
                DenseReskinSurface((7.0, 6.0, 7.0), three, three),
                DenseReskinSurface((7.0, 12.0, 7.0), three, three),
            ),
            None,
        ),
        (
            "corner_touch_orientation_boundary",
            (
                DenseReskinSurface((7.0, 7.0, 7.0), three, three),
                DenseReskinSurface(
                    (10.0, 8.0, 9.0),
                    three,
                    three,
                    strike=90.0,
                ),
            ),
            None,
        ),
        (
            "rounded_subvoxel_surface",
            (
                DenseReskinSurface(
                    (7.2, 7.0, 7.2),
                    three,
                    three,
                    v_scale=0.2,
                    w_scale=0.2,
                ),
            ),
            None,
        ),
        (
            "volume_boundary_surface",
            (
                DenseReskinSurface(
                    (0.55, 7.0, 7.0),
                    boundary,
                    tuple(key for key in boundary if key not in boundary_holes),
                    dip=75.0,
                    u_by_key=boundary_u,
                ),
            ),
            None,
        ),
    ]

    valid_mask = np.ones(shape, dtype=np.bool_)
    valid_mask[7, 7, 8] = False
    definitions.append(
        (
            "valid_mask_barrier",
            (
                DenseReskinSurface(
                    (7.0, 7.0, 7.0),
                    three,
                    tuple(key for key in three if key != (5, 4)),
                ),
            ),
            valid_mask,
        )
    )
    cases = [
        _materialize_case(case_id, shape, surfaces, valid_mask=mask)
        for case_id, surfaces, mask in definitions
    ]
    prior_occupancy = np.zeros(shape, dtype=np.bool_)
    prior_keys = tuple(key for key in three if key != (5, 5))
    prior_surface = DenseReskinSurface(
        (7.0, 7.0, 7.0),
        prior_keys,
        prior_keys,
    )
    prior_index = _rounded_index(
        _surface_world(prior_surface, _surface_transform(prior_surface), (5, 5))
    )
    prior_occupancy[prior_index[2], prior_index[1], prior_index[0]] = True
    cases.append(
        _materialize_case(
            "prior_occupancy_barrier",
            shape,
            (prior_surface,),
            valid_mask=None,
            prior_occupancy_mask=prior_occupancy,
        )
    )
    return tuple(cases)


def evaluate_dense_reskin_case(case: DenseReskinCase) -> dict[str, Any]:
    """Run both policies from identical skin-phase parent inputs."""

    baseline_skins: list[FaultSkin] = []
    candidate_skins: list[FaultSkin] = []
    baseline_diagnostics = _empty_reskin_diagnostics(RESKIN_POLICY_EXISTING_CELLS_V1)
    candidate_diagnostics = _empty_reskin_diagnostics(RESKIN_POLICY_REFERENCE_DENSE_V1)
    if len(case.surfaces) > 1:
        baseline_skins, baseline_diagnostics = _multi_surface_policy_skins(
            case, RESKIN_POLICY_EXISTING_CELLS_V1
        )
        candidate_skins, candidate_diagnostics = _multi_surface_policy_skins(
            case, RESKIN_POLICY_REFERENCE_DENSE_V1
        )
    else:
        surface = case.surfaces[0]
        input_skin, context = _surface_input(case, surface)
        item_baseline = _empty_reskin_diagnostics(RESKIN_POLICY_EXISTING_CELLS_V1)
        item_candidate = _empty_reskin_diagnostics(RESKIN_POLICY_REFERENCE_DENSE_V1)
        baseline_skins.append(_reskin_reference(input_skin, _diagnostics=item_baseline))
        candidate_skins.append(
            _reskin_reference_dense_v1(
                input_skin,
                context=context,
                _diagnostics=item_candidate,
            )
        )
        _merge_diagnostics(baseline_diagnostics, item_baseline)
        _merge_diagnostics(candidate_diagnostics, item_candidate)

    baseline = _policy_metrics(case, baseline_skins, baseline_diagnostics)
    candidate = _policy_metrics(case, candidate_skins, candidate_diagnostics)
    return {
        "case_id": case.case_id,
        "surface_count": len(case.surfaces),
        "skinning_invocation_count": 1,
        "parent_fingerprint": _case_parent_fingerprint(case),
        "baseline": baseline,
        "candidate": candidate,
        "contrast": _contrast_metrics(baseline, candidate),
    }


def build_dense_reskin_promotion_gate(
    cases: Sequence[DenseReskinCase] | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed pair and return a deterministic machine-readable gate."""

    case_set = tuple(controlled_dense_reskin_cases() if cases is None else cases)
    first_results = tuple(evaluate_dense_reskin_case(case) for case in case_set)
    second_results = tuple(evaluate_dense_reskin_case(case) for case in case_set)
    deterministic = _canonical_json(_nonfinite_to_none(first_results)) == _canonical_json(
        _nonfinite_to_none(second_results)
    )
    aggregate = _aggregate(first_results)
    reasons = _gate_reasons(first_results, aggregate, deterministic=deterministic)
    gate = {
        "schema_version": DENSE_RESKIN_GATE_SCHEMA_VERSION,
        "candidate": DENSE_RESKIN_CANDIDATE,
        "baseline": DENSE_RESKIN_BASELINE,
        "passed": not reasons,
        "reasons": reasons,
        "case_results": list(first_results),
        "aggregate": {
            **aggregate,
            "deterministic_reexecution": deterministic,
        },
    }
    return _nonfinite_to_none(gate)


def write_dense_reskin_evidence(
    gate: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Write canonical JSON plus compact CSV and Markdown evidence."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "dense_reskin_promotion_gate.json"
    csv_path = root / "dense_reskin_case_metrics.csv"
    markdown_path = root / "dense_reskin_promotion_gate.md"
    json_path.write_text(_canonical_json(gate) + "\n", encoding="utf-8")
    csv_path.write_text(_gate_csv(gate), encoding="utf-8")
    markdown_path.write_text(dense_reskin_gate_markdown(gate), encoding="utf-8")
    return json_path, csv_path, markdown_path


def dense_reskin_gate_markdown(gate: Mapping[str, Any]) -> str:
    """Render the dedicated evidence schema without coercing null metrics to zero."""

    status = "PASS" if gate["passed"] else "FAIL"
    lines = [
        "# Dense reskin promotion gate",
        "",
        f"- Status: **{status}**",
        f"- Baseline: `{gate['baseline']}`",
        f"- Candidate: `{gate['candidate']}`",
        "",
        "| Case | Policy | Generated | Fraction | Support min/mean/max | Recall | "
        "Buffered precision/recall | Chamfer | Strike/dip mean | Holes | Outside | "
        "Links v/c/i/q |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case_result in gate["case_results"]:
        for policy in ("baseline", "candidate"):
            result = case_result[policy]
            generation = result["generation"]
            truth = result["truth"]
            overlap = truth["overlap"]
            links = result["link_topology"]
            violations = (
                links["reciprocal_link_violation_count"]
                + links["cross_skin_link_count"]
                + links["self_link_count"]
            )
            link_summary = (
                f"{violations}/{links['linked_component_count']}/"
                f"{links['isolated_cell_count']}/{links['quad_closure_mismatch_count']}"
            )
            support = "/".join(
                _display(generation[name])
                for name in ("reskin_support_min", "reskin_support_mean", "reskin_support_max")
            )
            lines.append(
                f"| `{case_result['case_id']}` | {policy} | "
                f"{generation['reskin_generated_cell_count']} | "
                f"{_display(generation['reskin_generated_cell_fraction'])} | {support} | "
                f"{_display(overlap['recall'])} | "
                f"{_display(overlap['buffered_precision'])}/"
                f"{_display(overlap['buffered_recall'])} | "
                f"{_display(truth['surface_distance']['symmetric_chamfer_mean'])} | "
                f"{_display(truth['orientation_error']['strike_mean'])}/"
                f"{_display(truth['orientation_error']['dip_mean'])} | "
                f"{truth['truth_hole_recovered_count']}/{truth['truth_hole_count']} | "
                f"{truth['truth_buffer_outside_generated_cell_count']} | "
                f"{link_summary} |"
            )
    lines.extend(["", "## Gate reasons", ""])
    reasons = list(gate["reasons"])
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- None"])
    return "\n".join(lines) + "\n"


def _materialize_case(
    case_id: str,
    shape: tuple[int, int, int],
    surfaces: tuple[DenseReskinSurface, ...],
    *,
    valid_mask: np.ndarray | None,
    prior_occupancy_mask: np.ndarray | None = None,
) -> DenseReskinCase:
    truth = np.zeros(shape, dtype=np.bool_)
    ids = np.zeros(shape, dtype=np.int16)
    strike = np.zeros(shape, dtype=np.float32)
    dip = np.zeros(shape, dtype=np.float32)
    holes = np.zeros(shape, dtype=np.bool_)
    for fault_id, surface in enumerate(surfaces, start=1):
        transform = _surface_transform(surface)
        for key in surface.truth_keys:
            world = _surface_world(surface, transform, key)
            index = _rounded_index(world)
            if _index_in_shape(index, shape):
                i1, i2, i3 = index
                truth[i3, i2, i1] = True
                ids[i3, i2, i1] = fault_id
                strike[i3, i2, i1] = np.float32(surface.strike)
                dip[i3, i2, i1] = np.float32(surface.dip)
                if key not in surface.observed_keys:
                    holes[i3, i2, i1] = True
    return DenseReskinCase(
        case_id=case_id,
        shape=shape,
        surfaces=surfaces,
        truth_surface_mask=truth,
        truth_fault_id=ids,
        truth_strike=strike,
        truth_dip=dip,
        truth_hole_mask=holes,
        valid_mask=None if valid_mask is None else valid_mask.copy(),
        prior_occupancy_mask=(
            None if prior_occupancy_mask is None else prior_occupancy_mask.copy()
        ),
    )


def _surface_transform(surface: DenseReskinSurface) -> Any:
    seed = FaultCell(*surface.origin, surface.likelihood, surface.strike, surface.dip)
    transform = _update_transform_map(
        3,
        4,
        4,
        seed.fault_normal(),
        seed.fault_dip_vector(),
        seed.fault_strike_vector(),
    )
    transform.vs[:] *= np.float32(surface.v_scale)
    transform.ws[:] *= np.float32(surface.w_scale)
    return transform


def _surface_world(surface: DenseReskinSurface, transform: Any, key: tuple[int, int]) -> tuple:
    iu = float((surface.u_by_key or {}).get(key, 3.0))
    return tuple(
        float(value)
        for value in _local_index_to_world(
            int(round(iu)), key[0], key[1], surface.origin, transform
        )
    )


def _surface_input(
    case: DenseReskinCase,
    surface: DenseReskinSurface,
) -> tuple[FaultSkin, _ReskinContext]:
    transform = _surface_transform(surface)
    accepted = tuple(
        _SkinCell(
            float((surface.u_by_key or {}).get(key, 3.0)),
            key[0],
            key[1],
            surface.likelihood,
            surface.strike,
            surface.dip,
        )
        for key in surface.observed_keys
    )
    cells = [
        FaultCell(
            *_surface_world(surface, transform, key),
            surface.likelihood,
            surface.strike,
            surface.dip,
        )
        for key in surface.observed_keys
    ]
    fv = np.zeros(case.shape, dtype=np.float32)
    for index in np.argwhere(case.truth_surface_mask):
        fv[tuple(index)] = np.float32(surface.likelihood)
    seed = FaultCell(*surface.origin, surface.likelihood, surface.strike, surface.dip)
    collision_grid = None
    if case.prior_occupancy_mask is not None:
        collision_grid = _SkinOccupancyMask(case.shape)
        for i3, i2, i1 in np.argwhere(case.prior_occupancy_mask):
            collision_grid.mark_box(int(i1), int(i2), int(i3), 0, 0, 0)
    return FaultSkin.from_cells(cells), _ReskinContext(
        seed=_SkinCell(seed.x1, seed.x2, seed.x3, seed.fl, seed.fp, seed.ft),
        origin=surface.origin,
        transform_map=transform,
        accepted_cells=accepted,
        fv=fv,
        volume_shape=case.shape,
        collision_grid=collision_grid,
        du=5.0,
        valid_mask=case.valid_mask,
    )


def _multi_surface_policy_skins(
    case: DenseReskinCase,
    policy: str,
) -> tuple[list[FaultSkin], dict[str, Any]]:
    """Run all nearby surfaces through one shared skinning invocation."""

    fv = np.zeros(case.shape, dtype=np.float32)
    vp = np.zeros(case.shape, dtype=np.float32)
    vt = np.zeros(case.shape, dtype=np.float32)
    for surface in case.surfaces:
        transform = _surface_transform(surface)
        for key in surface.observed_keys:
            i1, i2, i3 = _rounded_index(_surface_world(surface, transform, key))
            fv[i3, i2, i1] = np.float32(surface.likelihood)
            vp[i3, i2, i1] = np.float32(surface.strike)
            vt[i3, i2, i1] = np.float32(surface.dip)
    config = SyntheticSkinningConfig(
        method="quality",
        growth_source="pre_thin",
        min_likelihood=0.5,
        min_skin_size=1,
        d=1,
        ru=3,
        rv=5,
        rw=5,
        max_steps=10,
        du=5.0,
        max_delta_strike=30.0,
        reskin=True,
        accepted_occupancy_radius=0,
    )
    diagnostics: dict[str, Any] = {}
    skins = find_synthetic_skins(
        fv,
        fv.copy(),
        vp,
        vt,
        skinning_config=replace(config, reskin_policy=policy),
        diagnostics=diagnostics,
    )
    return skins, diagnostics["reskin"]


def _policy_metrics(
    case: DenseReskinCase,
    skins: Sequence[FaultSkin],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    generation = reskin_generation_metrics(skins, diagnostics=diagnostics)
    truth = reskin_truth_correspondence_metrics(
        skins,
        shape=case.shape,
        truth_surface_mask=case.truth_surface_mask,
        truth_strike=case.truth_strike,
        truth_dip=case.truth_dip,
        buffer_radius=2.0,
        truth_hole_mask=case.truth_hole_mask,
    )
    generated_invalid = 0
    generated_prior = 0
    out_of_volume = 0
    clamped_artifacts = 0
    touched_ids_by_skin: list[set[int]] = []
    for skin in skins:
        touched: set[int] = set()
        for cell in skin:
            index = cell.index
            if not _index_in_shape(index, case.shape):
                out_of_volume += 1
                continue
            i1, i2, i3 = index
            if case.truth_fault_id[i3, i2, i1] > 0:
                touched.add(int(case.truth_fault_id[i3, i2, i1]))
            if cell.generation == FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED:
                if (
                    i1 in {0, case.shape[2] - 1}
                    or i2 in {0, case.shape[1] - 1}
                    or i3 in {0, case.shape[0] - 1}
                ) and not case.truth_surface_mask[i3, i2, i1]:
                    clamped_artifacts += 1
                if case.valid_mask is not None and not case.valid_mask[i3, i2, i1]:
                    generated_invalid += 1
                if case.prior_occupancy_mask is not None and case.prior_occupancy_mask[i3, i2, i1]:
                    generated_prior += 1
        touched_ids_by_skin.append(touched)
    topology = skin_topology_metrics(skins, case.shape, small_skin_size=10)
    result = {
        "generation": generation,
        "truth": truth,
        "link_topology": skin_link_topology_metrics(skins),
        "duplicate_rounded_cell_index_count": rounded_duplicate_cell_count(skins),
        "generated_on_invalid_mask_count": generated_invalid,
        "generated_on_prior_occupancy_count": generated_prior,
        "truth_fault_id_bridge_count": sum(len(ids) > 1 for ids in touched_ids_by_skin),
        "out_of_volume_cell_count": out_of_volume,
        "clamped_artifact_count": clamped_artifacts,
        "small_skin_cell_fraction": float(topology["small_skin_cell_fraction"]),
        "canonical_skin_payload": _canonical_skin_payload(skins),
    }
    result["finite_failure_count"] = int(_contains_nonfinite(result))
    return result


def _contrast_metrics(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    baseline_overlap = baseline["truth"]["overlap"]
    candidate_overlap = candidate["truth"]["overlap"]
    baseline_distance = baseline["truth"]["surface_distance"]
    candidate_distance = candidate["truth"]["surface_distance"]
    return {
        "exact_recall_delta": float(candidate_overlap["recall"] - baseline_overlap["recall"]),
        "buffered_recall_delta": float(
            candidate_overlap["buffered_recall"] - baseline_overlap["buffered_recall"]
        ),
        "buffered_precision_delta": float(
            candidate_overlap["buffered_precision"] - baseline_overlap["buffered_precision"]
        ),
        "symmetric_chamfer_mean_delta": float(
            candidate_distance["symmetric_chamfer_mean"]
            - baseline_distance["symmetric_chamfer_mean"]
        ),
        "small_skin_cell_fraction_delta": float(
            candidate["small_skin_cell_fraction"] - baseline["small_skin_cell_fraction"]
        ),
    }


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(policy: str, path: tuple[str, ...]) -> float:
        values = []
        for result in results:
            value: Any = result[policy]
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values)) if values else 0.0

    baseline_recall = mean("baseline", ("truth", "overlap", "buffered_recall"))
    candidate_recall = mean("candidate", ("truth", "overlap", "buffered_recall"))
    baseline_precision = mean("baseline", ("truth", "overlap", "buffered_precision"))
    candidate_precision = mean("candidate", ("truth", "overlap", "buffered_precision"))
    baseline_chamfer = mean("baseline", ("truth", "surface_distance", "symmetric_chamfer_mean"))
    candidate_chamfer = mean("candidate", ("truth", "surface_distance", "symmetric_chamfer_mean"))
    baseline_small = mean("baseline", ("small_skin_cell_fraction",))
    candidate_small = mean("candidate", ("small_skin_cell_fraction",))
    return {
        "case_count": len(results),
        "case_ids": [result["case_id"] for result in results],
        "baseline_buffered_recall_mean": baseline_recall,
        "candidate_buffered_recall_mean": candidate_recall,
        "buffered_recall_delta": candidate_recall - baseline_recall,
        "baseline_buffered_precision_mean": baseline_precision,
        "candidate_buffered_precision_mean": candidate_precision,
        "buffered_precision_delta": candidate_precision - baseline_precision,
        "baseline_symmetric_chamfer_mean": baseline_chamfer,
        "candidate_symmetric_chamfer_mean": candidate_chamfer,
        "symmetric_chamfer_mean_delta": candidate_chamfer - baseline_chamfer,
        "baseline_small_skin_cell_fraction_mean": baseline_small,
        "candidate_small_skin_cell_fraction_mean": candidate_small,
        "small_skin_cell_fraction_delta": candidate_small - baseline_small,
    }


def _gate_reasons(
    results: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    *,
    deterministic: bool,
) -> list[str]:
    reasons: list[str] = []
    by_id = {result["case_id"]: result for result in results}
    for result in results:
        baseline = result["baseline"]
        candidate = result["candidate"]
        if baseline["finite_failure_count"]:
            reasons.append(f"{result['case_id']}:baseline_nonfinite_metric")
        if candidate["finite_failure_count"]:
            reasons.append(f"{result['case_id']}:nonfinite_metric")
        if candidate["duplicate_rounded_cell_index_count"]:
            reasons.append(f"{result['case_id']}:duplicate_rounded_cell_index")
        links = candidate["link_topology"]
        for field in (
            "reciprocal_link_violation_count",
            "cross_skin_link_count",
            "self_link_count",
        ):
            if links[field]:
                reasons.append(f"{result['case_id']}:{field}")
        for field in (
            "generated_on_invalid_mask_count",
            "generated_on_prior_occupancy_count",
            "out_of_volume_cell_count",
            "clamped_artifact_count",
        ):
            if candidate[field]:
                reasons.append(f"{result['case_id']}:{field}")

    one = by_id["plane_3x3_center_hole"]["candidate"]
    if one["truth"]["truth_hole_recovered_count"] != 1:
        reasons.append("plane_3x3_center_hole:hole_not_recovered")
    if one["truth"]["truth_buffer_outside_generated_cell_count"]:
        reasons.append("plane_3x3_center_hole:generated_outside_truth_buffer")
    block = by_id["plane_4x4_internal_2x2_hole"]
    if (
        block["candidate"]["truth"]["overlap"]["recall"]
        <= block["baseline"]["truth"]["overlap"]["recall"]
    ):
        reasons.append("plane_4x4_internal_2x2_hole:recall_not_improved")
    if block["candidate"]["truth"]["truth_buffer_outside_generated_cell_count"]:
        reasons.append("plane_4x4_internal_2x2_hole:generated_outside_truth_buffer")
    if by_id["low_support_gap"]["candidate"]["generation"]["reskin_generated_cell_count"]:
        reasons.append("low_support_gap:generated_cell")
    valid_mask = by_id["valid_mask_barrier"]["candidate"]
    if not valid_mask["generation"]["reskin_rejected_invalid_mask_count"]:
        reasons.append("valid_mask_barrier:invalid_mask_rejection_not_exercised")
    prior = by_id["prior_occupancy_barrier"]["candidate"]
    if not prior["generation"]["reskin_rejected_prior_skin_collision_count"]:
        reasons.append("prior_occupancy_barrier:collision_not_exercised")
    boundary = by_id["volume_boundary_surface"]["candidate"]
    if not boundary["truth"]["truth_hole_recovered_count"]:
        reasons.append("volume_boundary_surface:boundary_hole_not_recovered")
    if not boundary["generation"]["reskin_generated_cell_count"]:
        reasons.append("volume_boundary_surface:dense_generation_not_exercised")
    if not boundary["generation"]["reskin_rejected_out_of_bounds_count"]:
        reasons.append("volume_boundary_surface:out_of_bounds_rejection_not_exercised")
    for case_id in ("parallel_surfaces", "corner_touch_orientation_boundary"):
        if by_id[case_id]["candidate"]["truth_fault_id_bridge_count"]:
            reasons.append(f"{case_id}:truth_fault_id_bridge")
    if not deterministic:
        reasons.append("canonical_payload_not_deterministic")
    if aggregate["candidate_buffered_recall_mean"] < aggregate["baseline_buffered_recall_mean"]:
        reasons.append("aggregate:buffered_recall_regressed")
    if aggregate["buffered_precision_delta"] < -0.02:
        reasons.append("aggregate:buffered_precision_regressed_over_0.02")
    if aggregate["symmetric_chamfer_mean_delta"] > 0.25:
        reasons.append("aggregate:symmetric_chamfer_regressed_over_0.25")
    if aggregate["small_skin_cell_fraction_delta"] > 0.05:
        reasons.append("aggregate:small_skin_cell_fraction_increased_over_0.05")
    return reasons


def _canonical_skin_payload(skins: Sequence[FaultSkin]) -> list[dict[str, Any]]:
    cell_owner = {id(cell): index for index, skin in enumerate(skins) for cell in skin}

    def link(cell: FaultCell | None) -> list[int] | None:
        if cell is None:
            return None
        return [cell_owner.get(id(cell), -1), cell.i1, cell.i2, cell.i3]

    payload = []
    for skin_index, skin in enumerate(skins):
        cells = sorted(
            skin, key=lambda cell: (cell.i3, cell.i2, cell.i1, cell.x3, cell.x2, cell.x1)
        )
        payload.append(
            {
                "skin_index": skin_index,
                "cells": [
                    {
                        "x1": cell.x1,
                        "x2": cell.x2,
                        "x3": cell.x3,
                        "i1": cell.i1,
                        "i2": cell.i2,
                        "i3": cell.i3,
                        "fl": cell.fl,
                        "fp": cell.fp,
                        "ft": cell.ft,
                        "generation": cell.generation,
                        "reskin_support": cell.reskin_support,
                        "ca": link(cell.ca),
                        "cb": link(cell.cb),
                        "cl": link(cell.cl),
                        "cr": link(cell.cr),
                    }
                    for cell in cells
                ],
            }
        )
    return payload


def _merge_diagnostics(target: dict[str, Any], item: Mapping[str, Any]) -> None:
    target["reskin_applied"] = bool(target["reskin_applied"] or item["reskin_applied"])
    for key in target:
        if key in {"reskin_policy", "reskin_applied"}:
            continue
        if key == "max_generated_chebyshev_distance_from_observed":
            target[key] = max(int(target[key]), int(item[key]))
        else:
            target[key] = int(target[key]) + int(item[key])


def _case_parent_fingerprint(case: DenseReskinCase) -> str:
    import hashlib

    payload = {
        "case_id": case.case_id,
        "shape": case.shape,
        "surfaces": [
            {
                "origin": surface.origin,
                "truth_keys": surface.truth_keys,
                "observed_keys": surface.observed_keys,
                "likelihood": surface.likelihood,
                "strike": surface.strike,
                "dip": surface.dip,
                "v_scale": surface.v_scale,
                "w_scale": surface.w_scale,
                "u_by_key": (
                    None
                    if surface.u_by_key is None
                    else sorted((list(key), value) for key, value in surface.u_by_key.items())
                ),
            }
            for surface in case.surfaces
        ],
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8"))
    for array in (
        case.truth_surface_mask,
        case.truth_fault_id,
        case.truth_hole_mask,
        case.valid_mask,
        case.prior_occupancy_mask,
    ):
        if array is None:
            digest.update(b"null")
        else:
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _gate_csv(gate: Mapping[str, Any]) -> str:
    fields = (
        "schema_version",
        "row_type",
        "case_id",
        "policy",
        "reskin_policy",
        "input_cell_count",
        "output_cell_count",
        "observed_output_cell_count",
        "generated_cell_count",
        "generated_cell_fraction",
        "dropped_input_cell_count",
        "projected_local_duplicate_count",
        "rejected_support_count",
        "rejected_invalid_mask_count",
        "rejected_prior_skin_collision_count",
        "rejected_out_of_bounds_count",
        "rejected_duplicate_world_index_count",
        "max_generated_chebyshev_distance_from_observed",
        "support_min",
        "support_mean",
        "support_max",
        "generated_likelihood_min",
        "generated_likelihood_mean",
        "exact_precision",
        "exact_recall",
        "exact_f1",
        "jaccard",
        "buffered_precision",
        "buffered_recall",
        "buffered_f1",
        "symmetric_chamfer_mean",
        "hausdorff_p95",
        "strike_error_mean",
        "dip_error_mean",
        "truth_hole_recovered_count",
        "truth_buffer_outside_generated_cell_count",
        "reciprocal_link_violation_count",
        "cross_skin_link_count",
        "self_link_count",
        "linked_component_count",
        "isolated_cell_count",
        "quad_closure_candidate_count",
        "quad_closure_match_count",
        "quad_closure_mismatch_count",
        "duplicate_rounded_cell_index_count",
        "truth_fault_id_bridge_count",
        "generated_on_invalid_mask_count",
        "generated_on_prior_occupancy_count",
        "out_of_volume_cell_count",
        "clamped_artifact_count",
        "exact_recall_delta",
        "buffered_recall_delta",
        "buffered_precision_delta",
        "symmetric_chamfer_mean_delta",
        "small_skin_cell_fraction_delta",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for case_result in gate["case_results"]:
        for policy, reskin_policy in (
            ("baseline", RESKIN_POLICY_EXISTING_CELLS_V1),
            ("candidate", RESKIN_POLICY_REFERENCE_DENSE_V1),
        ):
            result = case_result[policy]
            generation = result["generation"]
            overlap = result["truth"]["overlap"]
            distance = result["truth"]["surface_distance"]
            links = result["link_topology"]
            writer.writerow(
                {
                    "schema_version": gate["schema_version"],
                    "row_type": "policy",
                    "case_id": case_result["case_id"],
                    "policy": policy,
                    "reskin_policy": reskin_policy,
                    "input_cell_count": generation["reskin_input_cell_count"],
                    "output_cell_count": generation["reskin_output_cell_count"],
                    "observed_output_cell_count": generation["reskin_observed_output_cell_count"],
                    "generated_cell_count": generation["reskin_generated_cell_count"],
                    "generated_cell_fraction": generation["reskin_generated_cell_fraction"],
                    "dropped_input_cell_count": generation["reskin_dropped_input_cell_count"],
                    "projected_local_duplicate_count": generation[
                        "reskin_projected_local_duplicate_count"
                    ],
                    "rejected_support_count": generation["reskin_rejected_support_count"],
                    "rejected_invalid_mask_count": generation["reskin_rejected_invalid_mask_count"],
                    "rejected_prior_skin_collision_count": generation[
                        "reskin_rejected_prior_skin_collision_count"
                    ],
                    "rejected_out_of_bounds_count": generation[
                        "reskin_rejected_out_of_bounds_count"
                    ],
                    "rejected_duplicate_world_index_count": generation[
                        "reskin_rejected_duplicate_world_index_count"
                    ],
                    "max_generated_chebyshev_distance_from_observed": generation[
                        "reskin_max_generated_chebyshev_distance_from_observed"
                    ],
                    "support_min": generation["reskin_support_min"],
                    "support_mean": generation["reskin_support_mean"],
                    "support_max": generation["reskin_support_max"],
                    "generated_likelihood_min": generation[
                        "reskin_final_likelihood_min_for_generated"
                    ],
                    "generated_likelihood_mean": generation[
                        "reskin_final_likelihood_mean_for_generated"
                    ],
                    "exact_precision": overlap["precision"],
                    "exact_recall": overlap["recall"],
                    "exact_f1": overlap["f1"],
                    "jaccard": overlap["jaccard"],
                    "buffered_precision": overlap["buffered_precision"],
                    "buffered_recall": overlap["buffered_recall"],
                    "buffered_f1": overlap["buffered_f1"],
                    "symmetric_chamfer_mean": distance["symmetric_chamfer_mean"],
                    "hausdorff_p95": distance["hausdorff_p95"],
                    "strike_error_mean": result["truth"]["orientation_error"]["strike_mean"],
                    "dip_error_mean": result["truth"]["orientation_error"]["dip_mean"],
                    "truth_hole_recovered_count": result["truth"]["truth_hole_recovered_count"],
                    "truth_buffer_outside_generated_cell_count": result["truth"][
                        "truth_buffer_outside_generated_cell_count"
                    ],
                    "reciprocal_link_violation_count": links["reciprocal_link_violation_count"],
                    "cross_skin_link_count": links["cross_skin_link_count"],
                    "self_link_count": links["self_link_count"],
                    "linked_component_count": links["linked_component_count"],
                    "isolated_cell_count": links["isolated_cell_count"],
                    "quad_closure_candidate_count": links["quad_closure_candidate_count"],
                    "quad_closure_match_count": links["quad_closure_match_count"],
                    "quad_closure_mismatch_count": links["quad_closure_mismatch_count"],
                    "duplicate_rounded_cell_index_count": result[
                        "duplicate_rounded_cell_index_count"
                    ],
                    "truth_fault_id_bridge_count": result["truth_fault_id_bridge_count"],
                    "generated_on_invalid_mask_count": result["generated_on_invalid_mask_count"],
                    "generated_on_prior_occupancy_count": result[
                        "generated_on_prior_occupancy_count"
                    ],
                    "out_of_volume_cell_count": result["out_of_volume_cell_count"],
                    "clamped_artifact_count": result["clamped_artifact_count"],
                }
            )
        writer.writerow(
            {
                "schema_version": gate["schema_version"],
                "row_type": "contrast",
                "case_id": case_result["case_id"],
                "policy": "candidate_minus_baseline",
                "reskin_policy": (
                    f"{RESKIN_POLICY_REFERENCE_DENSE_V1}_minus_{RESKIN_POLICY_EXISTING_CELLS_V1}"
                ),
                **case_result["contrast"],
            }
        )
    return stream.getvalue()


def _contains_nonfinite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int, np.integer)):
        return False
    if isinstance(value, (float, np.floating)):
        return not np.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, Sequence):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _nonfinite_to_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Mapping):
        return {key: _nonfinite_to_none(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_nonfinite_to_none(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rounded_index(world: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(np.floor(value + 0.5)) for value in world)


def _index_in_shape(index: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    i1, i2, i3 = index
    n3, n2, n1 = shape
    return 0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the small controlled gate and write its three evidence files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    gate = build_dense_reskin_promotion_gate()
    write_dense_reskin_evidence(gate, args.output_dir)
    print(args.output_dir)
    return 0


__all__ = [
    "DENSE_RESKIN_BASELINE",
    "DENSE_RESKIN_CANDIDATE",
    "DENSE_RESKIN_GATE_SCHEMA_VERSION",
    "DenseReskinCase",
    "DenseReskinSurface",
    "build_dense_reskin_promotion_gate",
    "controlled_dense_reskin_cases",
    "dense_reskin_gate_markdown",
    "evaluate_dense_reskin_case",
    "write_dense_reskin_evidence",
]


if __name__ == "__main__":
    raise SystemExit(main())

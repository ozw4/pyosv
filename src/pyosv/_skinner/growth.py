"""Candidate selection and reference-like skin growth."""

from __future__ import annotations

import heapq
import operator

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE
from pyosv._skinner.candidate_sampling import (
    _candidate_slice_numba,
    _candidate_slice_python,
)
from pyosv._skinner.candidate_path import (
    _pick_candidate_local_u_path_numba,
    _pick_candidate_local_u_path_python,
)
from pyosv._skinner.models import (
    _LocalTransformMap,
    _ReskinContext,
    _SkinCell,
    link_above_below,
    link_left_right,
)
from pyosv._skinner.occupancy import _SkinOccupancyMask
from pyosv._skinner.reskin import (
    RESKIN_POLICY_EXISTING_CELLS_V1,
    RESKIN_POLICY_REFERENCE_DENSE_V1,
    ReskinDiagnostics,
    ReskinPolicy,
    _reskin_reference_dense_v1,
    _reskin_reference,
    _validate_reskin_policy,
)
from pyosv._skinner.transforms import (
    _local_index_to_world,
    _sample_validated_volume_nearest_java_round,
    _update_transform_map,
    _validate_origin3,
    _validate_transform_index,
)
from pyosv._skinner.validation import (
    _validate_bool,
    _validate_matching_finite_arrays3_many,
    _validate_nonnegative_finite_float,
    _validate_nonnegative_int,
    _validate_optional_valid_mask,
)
from pyosv.cells import FaultCell, _java_round
from pyosv.skin import FaultSkin


def _candidate_slice_above_below(
    fv: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    ub: int,
    ue: int,
    vc: int,
    wc: int,
    direction: int,
    max_steps: int | None = None,
) -> np.ndarray:
    """Sample a candidate slice in the local v direction over the u range."""

    return _candidate_slice(
        fv=fv,
        transform_map=transform_map,
        origin=origin,
        ub=ub,
        ue=ue,
        vc=vc,
        wc=wc,
        direction=direction,
        axis="v",
        max_steps=max_steps,
    )


def _candidate_slice_left_right(
    fv: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    ub: int,
    ue: int,
    vc: int,
    wc: int,
    direction: int,
    max_steps: int | None = None,
) -> np.ndarray:
    """Sample a candidate slice in the local w direction over the u range."""

    return _candidate_slice(
        fv=fv,
        transform_map=transform_map,
        origin=origin,
        ub=ub,
        ue=ue,
        vc=vc,
        wc=wc,
        direction=direction,
        axis="w",
        max_steps=max_steps,
    )


def _pick_candidate_us(ub: int, candidate_slice: np.ndarray) -> np.ndarray:
    """Pick a continuous local-u likelihood path through a candidate slice."""

    u_start = _validate_nonnegative_int(ub, "ub")
    slice_array = np.asarray(candidate_slice, dtype=np.float32)
    if slice_array.ndim != 2:
        raise ValueError("candidate_slice must be a 2D array")
    if slice_array.shape[1] == 0:
        raise ValueError("candidate_slice must contain at least one u sample")
    if not np.isfinite(slice_array).all():
        raise ValueError("candidate_slice must contain only finite values")

    return _pick_candidate_us_validated(u_start, slice_array)


def _pick_candidate_us_validated(ub: int, candidate_slice: np.ndarray) -> np.ndarray:
    """Pick a path from a validated finite float32 candidate slice."""

    local_us = _pick_candidate_local_u_path(candidate_slice)
    return (ub + local_us).astype(np.int32, copy=False)


def _pick_candidate_local_u_path(
    candidate_slice: np.ndarray,
    *,
    max_jump: int = 2,
    jump_penalty: float = 0.1,
) -> np.ndarray:
    kernel = (
        _pick_candidate_local_u_path_numba
        if NUMBA_AVAILABLE
        else _pick_candidate_local_u_path_python
    )
    return kernel(candidate_slice, max_jump, jump_penalty)


def _best_candidate_u(scores: np.ndarray, nu: int) -> int:
    best_u = 0
    best_score = float(scores[0])
    for iu in range(1, nu):
        score = float(scores[iu])
        if score > best_score or (
            score == best_score and _candidate_u_tie_key(iu, nu) < _candidate_u_tie_key(best_u, nu)
        ):
            best_u = iu
            best_score = score
    return best_u


def _candidate_u_tie_key(iu: int, nu: int) -> tuple[int, int]:
    return (abs(2 * iu - (nu - 1)), iu)


def _grow_reference_skin(
    seed: FaultCell | _SkinCell,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    fmin: float,
    ru: int = 150,
    rv: int | None = None,
    rw: int | None = None,
    max_steps: int = 10,
    du: float = 5.0,
    max_delta_strike: float = 30.0,
    collision_grid: _SkinOccupancyMask | None = None,
    reskin: bool = True,
    reskin_policy: ReskinPolicy | str = RESKIN_POLICY_EXISTING_CELLS_V1,
    valid_mask: np.ndarray | None = None,
    _reskin_diagnostics: ReskinDiagnostics | None = None,
) -> FaultSkin:
    """Grow one skin in a seed-local fault-coordinate grid."""

    should_reskin = _validate_bool(reskin, "reskin")
    policy = _validate_reskin_policy(reskin_policy)
    validated_valid_mask = _validate_optional_valid_mask(valid_mask, fv)
    seed_cell = _validate_seed_cell(seed)
    threshold = _validate_nonnegative_finite_float(fmin, "fmin")
    radius_u = _validate_nonnegative_int(ru, "ru")
    step_limit = _validate_nonnegative_int(max_steps, "max_steps")
    max_delta_u = _validate_nonnegative_finite_float(du, "du")
    max_delta_fp = _validate_nonnegative_finite_float(max_delta_strike, "max_delta_strike")
    fv_array, vp_array, vt_array = _validate_matching_finite_arrays3_many(
        (fv, vp, vt),
        ("fv", "vp", "vt"),
    )
    n3, n2, _ = fv_array.shape
    radius_v = max(n2, n3) if rv is None else _validate_nonnegative_int(rv, "rv")
    radius_w = max(n2, n3) if rw is None else _validate_nonnegative_int(rw, "rw")
    if radius_u < 2:
        raise ValueError("ru must be at least 2")
    if radius_v < 2:
        raise ValueError("rv must be at least 2")
    if radius_w < 2:
        raise ValueError("rw must be at least 2")

    return _grow_reference_skin_validated(
        seed_cell,
        fv_array,
        vp_array,
        vt_array,
        fmin=threshold,
        ru=radius_u,
        rv=radius_v,
        rw=radius_w,
        max_steps=step_limit,
        du=max_delta_u,
        max_delta_strike=max_delta_fp,
        collision_grid=collision_grid,
        reskin=should_reskin,
        reskin_policy=policy,
        valid_mask=validated_valid_mask,
        _reskin_diagnostics=_reskin_diagnostics,
    )


def _grow_reference_skin_validated(
    seed: _SkinCell,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    fmin: float,
    ru: int,
    rv: int,
    rw: int,
    max_steps: int,
    du: float,
    max_delta_strike: float,
    collision_grid: _SkinOccupancyMask | None = None,
    reskin: bool,
    reskin_policy: ReskinPolicy = RESKIN_POLICY_EXISTING_CELLS_V1,
    valid_mask: np.ndarray | None = None,
    _reskin_diagnostics: ReskinDiagnostics | None = None,
) -> FaultSkin:
    """Grow from normalized inputs and native-float32, finite, matching 3D volumes."""

    seed_cell = seed
    fv_array, vp_array, vt_array = fv, vp, vt
    threshold = fmin
    radius_u, radius_v, radius_w = ru, rv, rw
    step_limit = max_steps
    max_delta_u = du
    max_delta_fp = max_delta_strike
    should_reskin = reskin
    origin = (seed_cell.x1, seed_cell.x2, seed_cell.x3)
    transform_map = _update_transform_map(
        radius_u,
        radius_v,
        radius_w,
        seed_cell.fault_normal(),
        seed_cell.fault_dip_vector(),
        seed_cell.fault_strike_vector(),
    )
    local_seed = _SkinCell(
        radius_u,
        radius_v,
        radius_w,
        seed_cell.fl,
        seed_cell.fp,
        seed_cell.ft,
    )
    local_cells: dict[tuple[int, int], _SkinCell] = {(radius_v, radius_w): local_seed}
    accepted: list[_SkinCell] = []
    accepted_world_indices: set[tuple[int, int, int]] = set()
    queue: list[tuple[float, int, _SkinCell]] = []
    sequence = 0
    heapq.heappush(queue, (-local_seed.fl, sequence, local_seed))

    while queue:
        _, _, cell = heapq.heappop(queue)
        if cell.skin_id is not None:
            continue

        world = _local_cell_to_world(cell, origin, transform_map)
        if not _is_world_interior(world, fv_array.shape):
            continue
        world_index = _world_index(world)
        if world_index in accepted_world_indices:
            continue
        if collision_grid is not None and collision_grid.any_in_box(
            world_index[0],
            world_index[1],
            world_index[2],
            2,
            2,
            2,
        ):
            continue

        cell.skin_id = 0
        accepted.append(cell)
        accepted_world_indices.add(world_index)

        if not _is_local_cell_expandable(cell, transform_map):
            continue

        for axis, direction in (("v", -1), ("v", 1), ("w", -1), ("w", 1)):
            sequence = _grow_reference_direction(
                cell=cell,
                axis=axis,
                direction=direction,
                local_cells=local_cells,
                accepted_world_indices=accepted_world_indices,
                queue=queue,
                sequence=sequence,
                fv=fv_array,
                vp=vp_array,
                vt=vt_array,
                transform_map=transform_map,
                origin=origin,
                fmin=threshold,
                du=max_delta_u,
                max_delta_strike=max_delta_fp,
                max_steps=step_limit,
                collision_grid=collision_grid,
            )

    grown_skin = FaultSkin.from_cells(
        _local_cell_to_fault_cell(cell, origin, transform_map, fv_array, vp_array, vt_array)
        for cell in accepted
    )
    if not should_reskin:
        if _reskin_diagnostics is not None:
            _reskin_diagnostics["processed_skin_count"] = 1
            _reskin_diagnostics["input_cell_count"] = len(grown_skin)
            _reskin_diagnostics["output_cell_count"] = len(grown_skin)
            _reskin_diagnostics["observed_output_cell_count"] = len(grown_skin)
        return grown_skin
    if not grown_skin:
        if _reskin_diagnostics is not None:
            _reskin_diagnostics["processed_skin_count"] = 1
        return grown_skin

    context = _ReskinContext(
        seed=seed_cell,
        origin=origin,
        transform_map=transform_map,
        accepted_cells=tuple(accepted),
        fv=fv_array,
        volume_shape=fv_array.shape,
        collision_grid=collision_grid,
        du=max_delta_u,
        valid_mask=valid_mask,
    )
    return _apply_reskin_policy(
        grown_skin,
        policy=reskin_policy,
        context=context,
        _diagnostics=_reskin_diagnostics,
    )


def _apply_reskin_policy(
    skin: FaultSkin,
    *,
    policy: ReskinPolicy,
    context: _ReskinContext,
    _diagnostics: ReskinDiagnostics | None = None,
) -> FaultSkin:
    """Dispatch a grown skin through its validated versioned policy."""

    if policy == RESKIN_POLICY_EXISTING_CELLS_V1:
        # This policy intentionally does not consume grow context. The explicit
        # boundary preserves it for policies that need local growth state.
        del context
        return _reskin_reference(skin, _diagnostics=_diagnostics)
    if policy == RESKIN_POLICY_REFERENCE_DENSE_V1:
        return _reskin_reference_dense_v1(
            skin,
            context=context,
            _diagnostics=_diagnostics,
        )

    raise ValueError(
        "reskin_policy must be 'existing_cells_v1' or 'reference_dense_v1'",
    )


def _grow_reference_direction(
    *,
    cell: _SkinCell,
    axis: str,
    direction: int,
    local_cells: dict[tuple[int, int], _SkinCell],
    accepted_world_indices: set[tuple[int, int, int]],
    queue: list[tuple[float, int, _SkinCell]],
    sequence: int,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    fmin: float,
    du: float,
    max_delta_strike: float,
    max_steps: int,
    collision_grid: _SkinOccupancyMask | None,
) -> int:
    if not _link_slot_is_empty(cell, axis, direction):
        return sequence

    next_v = cell.i2 + direction if axis == "v" else cell.i2
    next_w = cell.i3 + direction if axis == "w" else cell.i3
    neighbor = local_cells.get((next_v, next_w))
    if neighbor is not None:
        if _candidate_matches_delta(cell, neighbor, origin, transform_map, du, max_delta_strike):
            _link_cells_for_direction(cell, neighbor, axis, direction)
        return sequence

    ub = max(cell.i1 - 5, 0)
    ue = min(cell.i1 + 5, transform_map.us.shape[1] - 1)
    candidate_slice = _candidate_slice_validated(
        fv,
        transform_map,
        origin,
        ub,
        ue,
        cell.i2,
        cell.i3,
        direction,
        axis,
        max_steps,
    )
    picked_us = _pick_candidate_us_validated(ub, candidate_slice)

    previous = cell
    for row in range(1, len(picked_us)):
        iu = int(picked_us[row])
        likelihood = float(candidate_slice[row, iu - ub])
        if likelihood < fmin:
            break

        iv = cell.i2 + direction * row if axis == "v" else cell.i2
        iw = cell.i3 + direction * row if axis == "w" else cell.i3
        existing = local_cells.get((iv, iw))
        if existing is not None:
            if not _candidate_matches_delta(
                previous,
                existing,
                origin,
                transform_map,
                du,
                max_delta_strike,
            ):
                break
            _link_cells_for_direction(previous, existing, axis, direction)
            previous = existing
            continue

        candidate = _candidate_from_local_index(
            iu,
            iv,
            iw,
            likelihood,
            vp,
            vt,
            transform_map,
            origin,
        )
        world = _local_cell_to_world(candidate, origin, transform_map)
        if not _is_world_interior(world, fv.shape):
            break
        world_index = _world_index(world)
        if world_index in accepted_world_indices:
            break
        if collision_grid is not None and collision_grid.any_in_box(
            world_index[0],
            world_index[1],
            world_index[2],
            2,
            2,
            2,
        ):
            break
        if not _candidate_matches_delta(
            previous,
            candidate,
            origin,
            transform_map,
            du,
            max_delta_strike,
        ):
            break

        local_cells[(iv, iw)] = candidate
        _link_cells_for_direction(previous, candidate, axis, direction)
        sequence += 1
        heapq.heappush(queue, (-candidate.fl, sequence, candidate))
        previous = candidate

    return sequence


def _candidate_from_local_index(
    iu: int,
    iv: int,
    iw: int,
    likelihood: float,
    vp: np.ndarray,
    vt: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
) -> _SkinCell:
    world = _local_index_to_world(iu, iv, iw, origin, transform_map)
    fp = _sample_validated_volume_nearest_java_round(vp, world[0], world[1], world[2])
    ft = _sample_validated_volume_nearest_java_round(vt, world[0], world[1], world[2])
    return _SkinCell(iu, iv, iw, likelihood, fp, ft)


def _candidate_matches_delta(
    previous: _SkinCell,
    candidate: _SkinCell,
    origin: tuple[float, float, float],
    transform_map: _LocalTransformMap,
    du: float,
    max_delta_strike: float,
) -> bool:
    if abs(candidate.x1 - previous.x1) > du:
        return False
    if _angle_delta_degrees(candidate.fp, previous.fp) > max_delta_strike:
        return False

    previous_world = _local_cell_to_world(previous, origin, transform_map)
    candidate_world = _local_cell_to_world(candidate, origin, transform_map)
    return abs(float(candidate_world[0]) - float(previous_world[0])) <= du


def _angle_delta_degrees(first: float, second: float) -> float:
    delta = abs(float(first) - float(second)) % 360.0
    return min(delta, 360.0 - delta)


def _link_slot_is_empty(cell: _SkinCell, axis: str, direction: int) -> bool:
    if axis == "v" and direction < 0:
        return cell.ca is None
    if axis == "v" and direction > 0:
        return cell.cb is None
    if axis == "w" and direction < 0:
        return cell.cl is None
    if axis == "w" and direction > 0:
        return cell.cr is None
    raise ValueError("axis must be 'v' or 'w' and direction must be -1 or 1")


def _link_cells_for_direction(
    cell: _SkinCell,
    neighbor: _SkinCell,
    axis: str,
    direction: int,
) -> None:
    if axis == "v" and direction < 0:
        link_above_below(neighbor, cell)
    elif axis == "v" and direction > 0:
        link_above_below(cell, neighbor)
    elif axis == "w" and direction < 0:
        link_left_right(neighbor, cell)
    elif axis == "w" and direction > 0:
        link_left_right(cell, neighbor)
    else:
        raise ValueError("axis must be 'v' or 'w' and direction must be -1 or 1")


def _local_cell_to_fault_cell(
    cell: _SkinCell,
    origin: tuple[float, float, float],
    transform_map: _LocalTransformMap,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
) -> FaultCell:
    x1, x2, x3 = _local_cell_to_world(cell, origin, transform_map)
    fl = _sample_validated_volume_nearest_java_round(fv, x1, x2, x3)
    fp = _sample_validated_volume_nearest_java_round(vp, x1, x2, x3)
    ft = _sample_validated_volume_nearest_java_round(vt, x1, x2, x3)
    return FaultCell(x1, x2, x3, fl, fp, ft)


def _local_cell_to_world(
    cell: _SkinCell,
    origin: tuple[float, float, float],
    transform_map: _LocalTransformMap,
) -> tuple[np.float32, np.float32, np.float32]:
    return _local_index_to_world(cell.i1, cell.i2, cell.i3, origin, transform_map)


def _is_local_cell_expandable(cell: _SkinCell, transform_map: _LocalTransformMap) -> bool:
    return (
        1 < cell.i1 < transform_map.us.shape[1] - 2
        and 1 < cell.i2 < transform_map.vs.shape[1] - 2
        and 1 < cell.i3 < transform_map.ws.shape[1] - 2
    )


def _is_world_interior(
    world: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> bool:
    n3, n2, n1 = shape
    x1, x2, x3 = (float(world[0]), float(world[1]), float(world[2]))
    return 1.0 < x1 < n1 - 2 and 1.0 < x2 < n2 - 2 and 1.0 < x3 < n3 - 2


def _world_index(world: tuple[float, float, float]) -> tuple[int, int, int]:
    return (_java_round(world[0]), _java_round(world[1]), _java_round(world[2]))


def _validate_seed_cell(seed: FaultCell | _SkinCell) -> _SkinCell:
    if isinstance(seed, _SkinCell):
        return seed
    if isinstance(seed, FaultCell):
        return _SkinCell(seed.x1, seed.x2, seed.x3, seed.fl, seed.fp, seed.ft)

    raise TypeError("seed must be a FaultCell")


def _candidate_slice(
    fv: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    ub: int,
    ue: int,
    vc: int,
    wc: int,
    direction: int,
    axis: str,
    max_steps: int | None,
) -> np.ndarray:
    fv_array = _validate_matching_finite_arrays3_many((fv,), ("fv",))[0]
    u_start = _validate_nonnegative_int(ub, "ub")
    u_stop = _validate_nonnegative_int(ue, "ue")
    if u_stop < u_start:
        raise ValueError("ue must be greater than or equal to ub")
    _validate_transform_index(u_start, transform_map.us, "ub")
    _validate_transform_index(u_stop, transform_map.us, "ue")
    v_center = _validate_transform_index(vc, transform_map.vs, "vc")
    w_center = _validate_transform_index(wc, transform_map.ws, "wc")
    step_sign = _validate_direction(direction)
    step_limit = None if max_steps is None else _validate_nonnegative_int(max_steps, "max_steps")
    if axis not in {"v", "w"}:
        raise ValueError("axis must be 'v' or 'w'")
    o1, o2, o3 = _validate_origin3(origin)

    return _candidate_slice_validated(
        fv_array,
        transform_map,
        (o1, o2, o3),
        u_start,
        u_stop,
        v_center,
        w_center,
        step_sign,
        axis,
        step_limit,
    )


def _candidate_slice_validated(
    fv: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    ub: int,
    ue: int,
    vc: int,
    wc: int,
    direction: int,
    axis: str,
    max_steps: int | None,
) -> np.ndarray:
    """Sample a slice from validated finite float32 volume and normalized inputs."""

    if axis == "v":
        axis_code = 0
        distance_to_edge = vc if direction < 0 else transform_map.vs.shape[1] - 1 - vc
    elif axis == "w":
        axis_code = 1
        distance_to_edge = wc if direction < 0 else transform_map.ws.shape[1] - 1 - wc
    else:
        raise ValueError("axis must be 'v' or 'w'")

    row_count = distance_to_edge + 1
    if max_steps is not None:
        row_count = min(row_count, max_steps + 1)

    kernel = _candidate_slice_numba if NUMBA_AVAILABLE else _candidate_slice_python
    return kernel(
        fv,
        transform_map.us,
        transform_map.vs,
        transform_map.ws,
        origin[0],
        origin[1],
        origin[2],
        ub,
        ue,
        vc,
        wc,
        direction,
        axis_code,
        row_count,
    )


def _validate_direction(direction: int) -> int:
    try:
        direction_int = operator.index(direction)
    except TypeError as exc:
        raise ValueError("direction must be -1 or 1") from exc
    if direction_int not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")

    return direction_int

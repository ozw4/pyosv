"""Reference-skin surface projection, smoothing, orientation, and linking."""

from __future__ import annotations

import heapq
from collections.abc import MutableMapping
from typing import Literal, TypedDict

import numpy as np

from pyosv._skinner.models import (
    _ReskinContext,
    _SkinCell,
    link_above_below,
    link_left_right,
)
from pyosv._skinner.validation import _validate_nonnegative_finite_float
from pyosv.cells import (
    FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
    FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED,
    FaultCell,
    _java_round,
)
from pyosv.filters import smooth2d
from pyosv.geometry import strike_and_dip_from_local_surface_derivatives
from pyosv.skin import FaultSkin

RESKIN_POLICY_EXISTING_CELLS_V1 = "existing_cells_v1"
RESKIN_POLICY_REFERENCE_DENSE_V1 = "reference_dense_v1"
RESKIN_POLICIES = (
    RESKIN_POLICY_EXISTING_CELLS_V1,
    RESKIN_POLICY_REFERENCE_DENSE_V1,
)
ReskinPolicy = Literal["existing_cells_v1", "reference_dense_v1"]
RESKIN_DIAGNOSTICS_CONTRACT_VERSION = 2


class ReskinAttemptDiagnostics(TypedDict):
    """Counts for one or more grow/reskin items before final filtering."""

    reskin_applied: bool
    processed_skin_count: int
    input_cell_count: int
    output_cell_count: int
    observed_output_cell_count: int
    generated_cell_count: int
    dropped_input_cell_count: int
    projected_local_duplicate_count: int
    candidate_local_key_count: int
    rejected_support_count: int
    rejected_invalid_mask_count: int
    rejected_prior_skin_collision_count: int
    rejected_out_of_bounds_count: int
    rejected_duplicate_world_index_count: int
    max_generated_chebyshev_distance_from_observed: int


class ReskinDiagnostics(ReskinAttemptDiagnostics):
    """Versioned final diagnostics emitted through the dedicated reskin sink."""

    reskin_diagnostics_contract_version: int
    reskin_policy: ReskinPolicy
    attempted: ReskinAttemptDiagnostics


_COUNT_DIAGNOSTIC_KEYS = (
    "processed_skin_count",
    "input_cell_count",
    "output_cell_count",
    "observed_output_cell_count",
    "generated_cell_count",
    "dropped_input_cell_count",
    "projected_local_duplicate_count",
    "candidate_local_key_count",
    "rejected_support_count",
    "rejected_invalid_mask_count",
    "rejected_prior_skin_collision_count",
    "rejected_out_of_bounds_count",
    "rejected_duplicate_world_index_count",
)


def _empty_reskin_attempt_diagnostics() -> ReskinAttemptDiagnostics:
    return ReskinAttemptDiagnostics(
        reskin_applied=False,
        processed_skin_count=0,
        input_cell_count=0,
        output_cell_count=0,
        observed_output_cell_count=0,
        generated_cell_count=0,
        dropped_input_cell_count=0,
        projected_local_duplicate_count=0,
        candidate_local_key_count=0,
        rejected_support_count=0,
        rejected_invalid_mask_count=0,
        rejected_prior_skin_collision_count=0,
        rejected_out_of_bounds_count=0,
        rejected_duplicate_world_index_count=0,
        max_generated_chebyshev_distance_from_observed=0,
    )


def _empty_reskin_diagnostics(policy: ReskinPolicy) -> ReskinDiagnostics:
    return ReskinDiagnostics(
        reskin_diagnostics_contract_version=RESKIN_DIAGNOSTICS_CONTRACT_VERSION,
        reskin_policy=policy,
        **_empty_reskin_attempt_diagnostics(),
        attempted=_empty_reskin_attempt_diagnostics(),
    )


def _prepare_reskin_diagnostics_sink(
    sink: MutableMapping[str, object] | None,
    policy: ReskinPolicy,
) -> ReskinDiagnostics | None:
    if sink is None:
        return None
    if not isinstance(sink, MutableMapping):
        raise TypeError("reskin_diagnostics must be a mutable mapping")
    sink.clear()
    initial = _empty_reskin_diagnostics(policy)
    sink.update(initial)
    return initial


def _merge_reskin_diagnostics(
    aggregate: ReskinAttemptDiagnostics,
    item: ReskinAttemptDiagnostics,
) -> None:
    aggregate["reskin_applied"] = aggregate["reskin_applied"] or item["reskin_applied"]
    for key in _COUNT_DIAGNOSTIC_KEYS:
        aggregate[key] += item[key]  # type: ignore[literal-required]
    aggregate["max_generated_chebyshev_distance_from_observed"] = max(
        aggregate["max_generated_chebyshev_distance_from_observed"],
        item["max_generated_chebyshev_distance_from_observed"],
    )


def _replace_final_reskin_diagnostics_with_observed_skins(
    diagnostics: ReskinDiagnostics,
    *,
    skin_count: int,
    cell_count: int,
) -> None:
    """Replace final counts after a connected-component fallback wins."""

    final = _empty_reskin_attempt_diagnostics()
    final["processed_skin_count"] = skin_count
    final["input_cell_count"] = cell_count
    final["output_cell_count"] = cell_count
    final["observed_output_cell_count"] = cell_count
    diagnostics.update(final)


def _publish_reskin_diagnostics(
    sink: MutableMapping[str, object] | None,
    diagnostics: ReskinDiagnostics | None,
) -> None:
    if sink is not None and diagnostics is not None:
        sink.update(diagnostics)


def _validate_reskin_policy(policy: object) -> ReskinPolicy:
    if not isinstance(policy, str) or policy not in RESKIN_POLICIES:
        raise ValueError(
            "reskin_policy must be 'existing_cells_v1' or 'reference_dense_v1'",
        )

    return policy


def validate_reskin_policy(policy: object) -> ReskinPolicy:
    """Validate and normalize one public reskin-policy value."""

    return _validate_reskin_policy(policy)


def _reskin_reference(
    skin: FaultSkin,
    *,
    smoothing_sigma: float = 1.0,
    _diagnostics: ReskinDiagnostics | None = None,
) -> FaultSkin:
    """Compatibility wrapper for the ``existing_cells_v1`` implementation."""

    return _reskin_existing_cells_v1(
        skin,
        smoothing_sigma=smoothing_sigma,
        _diagnostics=_diagnostics,
    )


def _reskin_existing_cells_v1(
    skin: FaultSkin,
    *,
    smoothing_sigma: float = 1.0,
    _diagnostics: ReskinDiagnostics | None = None,
) -> FaultSkin:
    """Smooth and reorient a grown reference-like skin.

    This is an approximation of the reference weighted smoothing phase: cells
    are projected to a seed-local ``(v, w)`` grid, local ``u`` offsets are
    smoothed with likelihood weights, strike/dip are recomputed from surface
    derivatives, and neighbor links are rebuilt on the local grid.
    """

    if not isinstance(skin, FaultSkin):
        raise TypeError("skin must be a FaultSkin")

    sigma = _validate_nonnegative_finite_float(smoothing_sigma, "smoothing_sigma")
    cells = list(skin)
    if _diagnostics is not None:
        _diagnostics["processed_skin_count"] = 1
        _diagnostics["input_cell_count"] = len(cells)
    if len(cells) <= 1:
        if _diagnostics is not None:
            _diagnostics["output_cell_count"] = len(cells)
            _diagnostics["observed_output_cell_count"] = len(cells)
        return FaultSkin.from_cells(cells)

    seed = _highest_likelihood_cell(cells)
    origin = np.array([seed.x1, seed.x2, seed.x3], dtype=np.float32)
    normal = seed.fault_normal()
    dip = seed.fault_dip_vector()
    strike = seed.fault_strike_vector()
    entries = _project_cells_to_local_surface(cells, origin, normal, dip, strike)
    if not entries:
        return FaultSkin()

    v_min = min(entry[0] for entry in entries)
    v_max = max(entry[0] for entry in entries)
    w_min = min(entry[1] for entry in entries)
    w_max = max(entry[1] for entry in entries)
    nv = v_max - v_min + 1
    nw = w_max - w_min + 1
    surface = np.zeros((nw, nv), dtype=np.float32)
    weights = np.zeros((nw, nv), dtype=np.float32)
    cells_by_key: dict[tuple[int, int], FaultCell] = {}
    order_by_key: dict[tuple[int, int], int] = {}

    for order, (iv, iw, iu, cell) in enumerate(entries):
        row = iw - w_min
        col = iv - v_min
        key = (iv, iw)
        if key in cells_by_key and cell.fl <= cells_by_key[key].fl:
            continue
        weight = np.float32(max(float(cell.fl), 0.0))
        surface[row, col] = np.float32(iu)
        weights[row, col] = weight if weight > 0.0 else np.float32(1.0)
        cells_by_key[key] = cell
        order_by_key.setdefault(key, order)

    if _diagnostics is not None:
        _diagnostics["reskin_applied"] = True
        _diagnostics["projected_local_duplicate_count"] = len(cells) - len(cells_by_key)

    smoothed_surface = _smooth_weighted_surface(surface, weights, sigma)
    local_cells: dict[tuple[int, int], _SkinCell] = {}
    public_cells: dict[tuple[int, int], FaultCell] = {}
    for key, cell in cells_by_key.items():
        iv, iw = key
        row = iw - w_min
        col = iv - v_min
        iu = float(smoothed_surface[row, col])
        fp, ft = _local_surface_strike_and_dip(
            normal,
            dip,
            strike,
            smoothed_surface,
            row,
            col,
        )
        world = origin + iu * normal + np.float32(iv) * dip + np.float32(iw) * strike
        public_cells[key] = FaultCell(
            world[0],
            world[1],
            world[2],
            cell.fl,
            fp,
            ft,
            generation=FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED,
        )
        local_cells[key] = _SkinCell(iu, iv, iw, cell.fl, fp, ft)

    _link_local_surface_cells(local_cells)
    _link_public_surface_cells(public_cells)
    ordered_keys = sorted(public_cells, key=lambda key: order_by_key[key])
    if _diagnostics is not None:
        output_count = len(ordered_keys)
        _diagnostics["output_cell_count"] = output_count
        _diagnostics["observed_output_cell_count"] = output_count
        _diagnostics["dropped_input_cell_count"] = max(0, len(cells) - output_count)
    return FaultSkin.from_cells(public_cells[key] for key in ordered_keys)


def _reskin_reference_dense_v1(
    skin: FaultSkin,
    *,
    context: _ReskinContext,
    _diagnostics: ReskinDiagnostics | None = None,
) -> FaultSkin:
    """Regrow a smoothed seed-local surface into missing local grid positions."""

    if not isinstance(skin, FaultSkin):
        raise TypeError("skin must be a FaultSkin")

    cells = list(skin)
    if _diagnostics is not None:
        _diagnostics["processed_skin_count"] = 1
        _diagnostics["input_cell_count"] = len(cells)
    if len(cells) <= 1:
        if _diagnostics is not None:
            _diagnostics["output_cell_count"] = len(cells)
            _diagnostics["observed_output_cell_count"] = len(cells)
        return FaultSkin.from_cells(cells)

    observed = _dense_observed_cells(context.accepted_cells)
    if _diagnostics is not None:
        _diagnostics["reskin_applied"] = True
        _diagnostics["projected_local_duplicate_count"] = max(
            0,
            len(context.accepted_cells) - len(observed),
        )
    if not observed:
        return FaultSkin()

    v_min = min(key[0] for key in observed)
    v_max = max(key[0] for key in observed)
    w_min = min(key[1] for key in observed)
    w_max = max(key[1] for key in observed)
    shape = (w_max - w_min + 1, v_max - v_min + 1)
    observed_u = np.zeros(shape, dtype=np.float32)
    observed_likelihood = np.zeros(shape, dtype=np.float32)
    weights = np.zeros(shape, dtype=np.float32)

    for (iv, iw), (_, cell) in observed.items():
        row, col = iw - w_min, iv - v_min
        likelihood = np.float32(cell.fl)
        observed_u[row, col] = np.float32(cell.x1)
        observed_likelihood[row, col] = likelihood
        weights[row, col] = np.float32(max(float(likelihood), 0.0) ** 2)

    numerator = smooth2d(observed_u * weights, (4.0, 4.0))
    denominator = smooth2d(weights, (4.0, 4.0))
    smoothed_u = np.zeros(shape, dtype=np.float32)
    valid_surface = denominator > np.float32(1.0e-6)
    np.divide(numerator, denominator, out=smoothed_u, where=valid_surface)
    support = smooth2d(observed_likelihood, (8.0, 8.0)).astype(
        np.float32,
        copy=False,
    )

    center_u, center_v, center_w, normal, dip, strike = _dense_basis(context)
    seed_key = (center_v, center_w)
    if seed_key not in observed:
        if _diagnostics is not None:
            _diagnostics["output_cell_count"] = len(cells)
            _diagnostics["dropped_input_cell_count"] = len(cells)
        return FaultSkin.from_cells(cells)

    rejection_by_key: dict[tuple[int, int], str] | None = {} if _diagnostics is not None else None
    candidate_keys: set[tuple[int, int]] | None = {seed_key} if _diagnostics is not None else None
    accepted_keys = _dense_regrow_keys(
        seed_key=seed_key,
        bounds=(v_min, v_max, w_min, w_max),
        smoothed_u=smoothed_u,
        valid_surface=valid_surface,
        support=support,
        offsets=(v_min, w_min),
        context=context,
        centers=(center_u, center_v, center_w),
        basis=(normal, dip, strike),
        _candidate_keys=candidate_keys,
        _rejection_by_key=rejection_by_key,
    )
    candidates: dict[
        tuple[int, int],
        tuple[tuple[float, float, float], tuple[int, int, int]],
    ] = {}
    for key in accepted_keys:
        world = _dense_world(
            key,
            smoothed_u,
            offsets=(v_min, w_min),
            origin=context.origin,
            centers=(center_u, center_v, center_w),
            basis=(normal, dip, strike),
        )
        candidates[key] = (world, _rounded_world_index(world))

    winners: dict[tuple[int, int, int], tuple[int, int]] = {}
    for key in accepted_keys:
        world_index = candidates[key][1]
        current = winners.get(world_index)
        if current is None or _dense_duplicate_priority(
            key,
            observed,
            support,
            offsets=(v_min, w_min),
        ) < _dense_duplicate_priority(
            current,
            observed,
            support,
            offsets=(v_min, w_min),
        ):
            winners[world_index] = key

    kept_keys = [key for key in accepted_keys if winners[candidates[key][1]] == key]
    if rejection_by_key is not None:
        for key in accepted_keys:
            rejection_by_key.pop(key, None)
        duplicate_count = len(accepted_keys) - len(kept_keys)
        _diagnostics["candidate_local_key_count"] = len(candidate_keys or ())
        _diagnostics["rejected_support_count"] = sum(
            reason == "support" for reason in rejection_by_key.values()
        )
        _diagnostics["rejected_invalid_mask_count"] = sum(
            reason == "invalid_mask" for reason in rejection_by_key.values()
        )
        _diagnostics["rejected_prior_skin_collision_count"] = sum(
            reason == "prior_skin_collision" for reason in rejection_by_key.values()
        )
        _diagnostics["rejected_out_of_bounds_count"] = sum(
            reason == "out_of_bounds" for reason in rejection_by_key.values()
        )
        _diagnostics["rejected_duplicate_world_index_count"] = duplicate_count
    public_cells: dict[tuple[int, int], FaultCell] = {}
    for iv, iw in kept_keys:
        row, col = iw - w_min, iv - v_min
        world, (i1, i2, i3) = candidates[(iv, iw)]
        fp, ft = _local_surface_strike_and_dip(
            normal,
            dip,
            strike,
            smoothed_u,
            row,
            col,
        )
        fl = float(context.fv[i3, i2, i1])
        generation = (
            FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED
            if (iv, iw) in observed
            else FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED
        )
        cell_support = float(np.clip(support[row, col], 0.0, 1.0))
        public_cells[(iv, iw)] = FaultCell(
            *world,
            fl,
            fp,
            ft,
            generation=generation,
            reskin_support=cell_support,
        )

    _link_public_surface_cells(public_cells)
    if _diagnostics is not None:
        observed_output_count = sum(key in observed for key in kept_keys)
        generated_keys = [key for key in kept_keys if key not in observed]
        _diagnostics["output_cell_count"] = len(kept_keys)
        _diagnostics["observed_output_cell_count"] = observed_output_count
        _diagnostics["generated_cell_count"] = len(generated_keys)
        _diagnostics["dropped_input_cell_count"] = max(
            0,
            len(cells) - observed_output_count,
        )
        if generated_keys:
            _diagnostics["max_generated_chebyshev_distance_from_observed"] = max(
                min(
                    max(abs(key[0] - observed_key[0]), abs(key[1] - observed_key[1]))
                    for observed_key in observed
                )
                for key in generated_keys
            )
    return FaultSkin.from_cells(public_cells[key] for key in kept_keys)


def _dense_observed_cells(
    accepted_cells: tuple[_SkinCell, ...],
) -> dict[tuple[int, int], tuple[int, _SkinCell]]:
    """Choose one observed cell per local key by likelihood, then grow order."""

    observed: dict[tuple[int, int], tuple[int, _SkinCell]] = {}
    for order, cell in enumerate(accepted_cells):
        key = (cell.i2, cell.i3)
        current = observed.get(key)
        if current is None or cell.fl > current[1].fl:
            observed[key] = (order, cell)
    return observed


def _dense_basis(
    context: _ReskinContext,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    transform = context.transform_map
    center_u = transform.us.shape[1] // 2
    center_v = transform.vs.shape[1] // 2
    center_w = transform.ws.shape[1] // 2
    normal = transform.us[:, center_u + 1].astype(np.float32, copy=False)
    dip = transform.vs[:, center_v + 1].astype(np.float32, copy=False)
    strike = transform.ws[:, center_w + 1].astype(np.float32, copy=False)
    return center_u, center_v, center_w, normal, dip, strike


def _dense_regrow_keys(
    *,
    seed_key: tuple[int, int],
    bounds: tuple[int, int, int, int],
    smoothed_u: np.ndarray,
    valid_surface: np.ndarray,
    support: np.ndarray,
    offsets: tuple[int, int],
    context: _ReskinContext,
    centers: tuple[int, int, int],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    _candidate_keys: set[tuple[int, int]] | None = None,
    _rejection_by_key: dict[tuple[int, int], str] | None = None,
) -> list[tuple[int, int]]:
    """Traverse eligible local keys in deterministic support priority."""

    accepted = [seed_key]
    queued = {seed_key}
    queue: list[tuple[float, int, int]] = []

    def enqueue_neighbors(current: tuple[int, int]) -> None:
        iv, iw = current
        current_u = float(smoothed_u[iw - offsets[1], iv - offsets[0]])
        for candidate in ((iv - 1, iw), (iv + 1, iw), (iv, iw - 1), (iv, iw + 1)):
            if candidate in queued:
                continue
            candidate_iv, candidate_iw = candidate
            v_min, v_max, w_min, w_max = bounds
            if not (v_min <= candidate_iv <= v_max and w_min <= candidate_iw <= w_max):
                continue
            if _candidate_keys is not None:
                _candidate_keys.add(candidate)
            rejection = _dense_candidate_rejection(
                candidate,
                current_u=current_u,
                smoothed_u=smoothed_u,
                valid_surface=valid_surface,
                support=support,
                offsets=offsets,
                context=context,
                centers=centers,
                basis=basis,
            )
            if rejection is not None:
                if _rejection_by_key is not None:
                    _rejection_by_key.setdefault(candidate, rejection)
                continue
            queued.add(candidate)
            if _rejection_by_key is not None:
                _rejection_by_key.pop(candidate, None)
            candidate_support = float(
                support[candidate[1] - offsets[1], candidate[0] - offsets[0]],
            )
            heapq.heappush(queue, (-candidate_support, candidate[1], candidate[0]))

    enqueue_neighbors(seed_key)
    while queue:
        _, iw, iv = heapq.heappop(queue)
        key = (iv, iw)
        accepted.append(key)
        enqueue_neighbors(key)
    return accepted


def _dense_candidate_is_eligible(
    key: tuple[int, int],
    *,
    current_u: float,
    bounds: tuple[int, int, int, int],
    smoothed_u: np.ndarray,
    valid_surface: np.ndarray,
    support: np.ndarray,
    offsets: tuple[int, int],
    context: _ReskinContext,
    centers: tuple[int, int, int],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    iv, iw = key
    v_min, v_max, w_min, w_max = bounds
    if not (v_min <= iv <= v_max and w_min <= iw <= w_max):
        return False
    return (
        _dense_candidate_rejection(
            key,
            current_u=current_u,
            smoothed_u=smoothed_u,
            valid_surface=valid_surface,
            support=support,
            offsets=offsets,
            context=context,
            centers=centers,
            basis=basis,
        )
        is None
    )


def _dense_candidate_rejection(
    key: tuple[int, int],
    *,
    current_u: float,
    smoothed_u: np.ndarray,
    valid_surface: np.ndarray,
    support: np.ndarray,
    offsets: tuple[int, int],
    context: _ReskinContext,
    centers: tuple[int, int, int],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> str | None:
    """Return the first rejection reason in the versioned dense-v1 order."""

    iv, iw = key
    row, col = iw - offsets[1], iv - offsets[0]
    if not valid_surface[row, col] or not float(support[row, col]) > 0.2:
        return "support"
    if not abs(float(smoothed_u[row, col]) - current_u) < 5.0:
        return "local_u"

    world = _dense_world(
        key,
        smoothed_u,
        offsets=offsets,
        origin=context.origin,
        centers=centers,
        basis=basis,
    )
    if not _dense_world_is_in_volume(world, context.volume_shape):
        return "out_of_bounds"
    i1, i2, i3 = _rounded_world_index(world)
    if context.valid_mask is not None and not bool(context.valid_mask[i3, i2, i1]):
        return "invalid_mask"
    if context.collision_grid is not None and context.collision_grid.any_in_box(
        i1,
        i2,
        i3,
        2,
        2,
        2,
    ):
        return "prior_skin_collision"
    return None


def _dense_world(
    key: tuple[int, int],
    surface: np.ndarray,
    *,
    offsets: tuple[int, int],
    origin: tuple[float, float, float],
    centers: tuple[int, int, int],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float]:
    iv, iw = key
    row, col = iw - offsets[1], iv - offsets[0]
    center_u, center_v, center_w = centers
    normal, dip, strike = basis
    u = np.float32(surface[row, col] - center_u)
    v = np.float32(iv - center_v)
    w = np.float32(iw - center_w)
    world = np.asarray(origin, dtype=np.float32) + u * normal + v * dip + w * strike
    return float(world[0]), float(world[1]), float(world[2])


def _dense_world_is_in_volume(
    world: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> bool:
    if not np.isfinite(world).all():
        return False
    n3, n2, n1 = shape
    x1, x2, x3 = world
    if not (0.0 <= x1 <= n1 - 1 and 0.0 <= x2 <= n2 - 1 and 0.0 <= x3 <= n3 - 1):
        return False
    i1, i2, i3 = _rounded_world_index(world)
    return 0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3


def _rounded_world_index(world: tuple[float, float, float]) -> tuple[int, int, int]:
    return _java_round(world[0]), _java_round(world[1]), _java_round(world[2])


def _dense_duplicate_priority(
    key: tuple[int, int],
    observed: dict[tuple[int, int], tuple[int, _SkinCell]],
    support: np.ndarray,
    *,
    offsets: tuple[int, int],
) -> tuple[int, float, int, int]:
    iv, iw = key
    return (
        -int(key in observed),
        -float(support[iw - offsets[1], iv - offsets[0]]),
        iw,
        iv,
    )


def _highest_likelihood_cell(cells: list[FaultCell]) -> FaultCell:
    best_index = max(range(len(cells)), key=lambda index: (cells[index].fl, -index))
    return cells[best_index]


def _project_cells_to_local_surface(
    cells: list[FaultCell],
    origin: np.ndarray,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
) -> list[tuple[int, int, float, FaultCell]]:
    entries: list[tuple[int, int, float, FaultCell]] = []
    for cell in cells:
        offset = np.array([cell.x1, cell.x2, cell.x3], dtype=np.float32) - origin
        iu = float(np.dot(offset, normal))
        iv = _java_round(float(np.dot(offset, dip)))
        iw = _java_round(float(np.dot(offset, strike)))
        entries.append((iv, iw, iu, cell))

    return entries


def _smooth_weighted_surface(
    surface: np.ndarray,
    weights: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if sigma == 0.0 or surface.size <= 1:
        return surface.copy()

    numerator = smooth2d(surface * weights, sigma)
    denominator = smooth2d(weights, sigma)
    smoothed = surface.copy()
    np.divide(
        numerator,
        denominator,
        out=smoothed,
        where=denominator > np.float32(1.0e-6),
    )
    return smoothed.astype(np.float32, copy=False)


def _local_surface_strike_and_dip(
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    row: int,
    col: int,
) -> tuple[float, float]:
    du_dv = _surface_derivative(surface, row, col, axis=1)
    du_dw = _surface_derivative(surface, row, col, axis=0)
    return strike_and_dip_from_local_surface_derivatives(
        normal,
        dip,
        strike,
        du_dv,
        du_dw,
    )


def _surface_derivative(surface: np.ndarray, row: int, col: int, *, axis: int) -> float:
    if axis == 1:
        if surface.shape[1] == 1:
            return 0.0
        if 0 < col < surface.shape[1] - 1:
            return float(0.5 * (surface[row, col + 1] - surface[row, col - 1]))
        if col == 0:
            return float(surface[row, col + 1] - surface[row, col])
        return float(surface[row, col] - surface[row, col - 1])

    if axis == 0:
        if surface.shape[0] == 1:
            return 0.0
        if 0 < row < surface.shape[0] - 1:
            return float(0.5 * (surface[row + 1, col] - surface[row - 1, col]))
        if row == 0:
            return float(surface[row + 1, col] - surface[row, col])
        return float(surface[row, col] - surface[row - 1, col])

    raise ValueError("axis must be 0 or 1")


def _link_local_surface_cells(local_cells: dict[tuple[int, int], _SkinCell]) -> None:
    for (iv, iw), cell in local_cells.items():
        below = local_cells.get((iv + 1, iw))
        right = local_cells.get((iv, iw + 1))
        if below is not None:
            link_above_below(cell, below)
        if right is not None:
            link_left_right(cell, right)


def _link_public_surface_cells(public_cells: dict[tuple[int, int], FaultCell]) -> None:
    for (iv, iw), cell in public_cells.items():
        below = public_cells.get((iv + 1, iw))
        right = public_cells.get((iv, iw + 1))
        if below is not None:
            _link_fault_cells_above_below(cell, below)
        if right is not None:
            _link_fault_cells_left_right(cell, right)


def _link_fault_cells_above_below(a: FaultCell, b: FaultCell) -> None:
    object.__setattr__(a, "cb", b)
    object.__setattr__(b, "ca", a)


def _link_fault_cells_left_right(left: FaultCell, right: FaultCell) -> None:
    object.__setattr__(left, "cr", right)
    object.__setattr__(right, "cl", left)

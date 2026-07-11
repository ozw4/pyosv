"""Java-rounding cells and masked-surface feasibility recovery."""

from __future__ import annotations

import math

import numpy as np

from pyosv._accel import njit


def _surface_respects_masked_strain(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
) -> bool:
    """Return whether shape, finiteness, mask, and both strain limits hold."""

    surface_array = np.asarray(surface)
    if surface_array.ndim != 2 or surface_array.shape != valid_mask.shape[:2]:
        return False
    if not np.isfinite(surface_array).all():
        return False

    nw, nv = surface_array.shape
    nu = valid_mask.shape[2]
    if np.any(surface_array < lmin) or np.any(surface_array > lmin + nu - 1):
        return False
    for iw in range(nw):
        for iv in range(nv):
            lag_index = math.floor(float(surface_array[iw, iv] - lmin) + 0.5)
            if lag_index < 0 or lag_index >= nu or not valid_mask[iw, iv, lag_index]:
                return False

    tolerance = np.float32(1.0e-6)
    strain1 = np.float32(1.0 / bstrain1)
    strain2 = np.float32(1.0 / bstrain2)
    if nv > 1 and np.any(np.abs(np.diff(surface_array, axis=1)) > strain1 + tolerance):
        return False
    return not (nw > 1 and np.any(np.abs(np.diff(surface_array, axis=0)) > strain2 + tolerance))


def _recover_bidirectionally_feasible_surface(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
) -> np.ndarray | None:
    """Recover a deterministic globally feasible surface from valid cell runs."""

    target = np.asarray(surface, dtype=np.float32)
    nw, nv, _ = valid_mask.shape
    if target.shape != (nw, nv) or not np.isfinite(target).all():
        return None

    domains: list[list[tuple[float, float, int, int]]] = []
    for iw in range(nw):
        for iv in range(nv):
            runs = _valid_rounding_cell_runs(valid_mask[iw, iv], lmin)
            if not runs:
                return None
            domains.append(runs)

    search_budget = [max(50_000, 64 * nw * nv)]
    return _search_feasible_rounding_cell_runs(
        target,
        valid_mask,
        domains,
        lmin=lmin,
        bstrain1=bstrain1,
        bstrain2=bstrain2,
        search_budget=search_budget,
        search_depth=0,
    )


@njit(cache=True)
def _java_rounding_cell_bounds(
    lmin: int,
    lag_index: int,
    lag_count: int,
) -> tuple[np.float32, np.float32]:
    """Return closed float32 bounds selecting one Java-rounded lag."""

    lag = np.float32(lmin + lag_index)
    lower = lag if lag_index == 0 else np.float32(lag - np.float32(0.5))
    if lag_index == lag_count - 1:
        upper = lag
    else:
        upper_exclusive = np.float32(lag + np.float32(0.5))
        upper = np.nextafter(upper_exclusive, np.float32(-np.inf))
    return lower, upper


@njit(cache=True)
def _project_value_to_java_rounding_cell(
    value: np.float32,
    lmin: int,
    lag_index: int,
    lag_count: int,
) -> tuple[np.float32, float]:
    """Project one value to a lag cell and return its deterministic distance."""

    lower, upper = _java_rounding_cell_bounds(lmin, lag_index, lag_count)
    if value < lower:
        return lower, float(lower) - float(value)
    if value > upper:
        ordering_upper = upper
        if lag_index < lag_count - 1:
            ordering_upper = np.float32(lmin + lag_index) + np.float32(0.5)
        return upper, float(value) - float(ordering_upper)
    return value, 0.0


def _valid_rounding_cell_runs(
    valid_column: np.ndarray,
    lmin: int,
) -> list[tuple[float, float, int, int]]:
    """Return separate Java-rounding intervals for contiguous valid lag runs."""

    runs: list[tuple[float, float, int, int]] = []
    run_start = -1
    for lag_index in range(valid_column.size + 1):
        is_valid = lag_index < valid_column.size and bool(valid_column[lag_index])
        if is_valid and run_start < 0:
            run_start = lag_index
        if is_valid or run_start < 0:
            continue

        run_stop = lag_index - 1
        lower, _ = _java_rounding_cell_bounds(lmin, run_start, valid_column.size)
        _, upper = _java_rounding_cell_bounds(lmin, run_stop, valid_column.size)
        runs.append((float(lower), float(upper), run_start, run_stop))
        run_start = -1
    return runs


def _search_feasible_rounding_cell_runs(
    target: np.ndarray,
    valid_mask: np.ndarray,
    domains: list[list[tuple[float, float, int, int]]],
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
    search_budget: list[int],
    search_depth: int,
) -> np.ndarray | None:
    # Sampler-derived masks normally have one run per column. Bound recursive
    # branching so arbitrary synthetic hole patterns fail safely instead of
    # reaching Python's recursion limit.
    if search_budget[0] <= 0 or search_depth >= 512:
        return None
    search_budget[0] -= 1

    propagated = _propagate_rounding_cell_run_domains(
        domains,
        shape=target.shape,
        strain1=1.0 / bstrain1,
        strain2=1.0 / bstrain2,
    )
    if propagated is None:
        return None

    branch_index = -1
    branch_count = np.iinfo(np.int32).max
    for node_index, node_runs in enumerate(propagated):
        if 1 < len(node_runs) < branch_count:
            branch_index = node_index
            branch_count = len(node_runs)

    if branch_index < 0:
        return _surface_from_single_rounding_cell_runs(
            target,
            valid_mask,
            propagated,
            lmin=lmin,
            bstrain1=bstrain1,
            bstrain2=bstrain2,
        )

    iw, iv = divmod(branch_index, target.shape[1])
    target_value = float(target[iw, iv])
    ordered_runs = sorted(
        propagated[branch_index],
        key=lambda run: (_distance_to_closed_interval(target_value, run[0], run[1]), run[2]),
    )
    for selected_run in ordered_runs:
        selected_domains = [list(node_runs) for node_runs in propagated]
        selected_domains[branch_index] = [selected_run]
        recovered = _search_feasible_rounding_cell_runs(
            target,
            valid_mask,
            selected_domains,
            lmin=lmin,
            bstrain1=bstrain1,
            bstrain2=bstrain2,
            search_budget=search_budget,
            search_depth=search_depth + 1,
        )
        if recovered is not None:
            return recovered
    return None


def _propagate_rounding_cell_run_domains(
    domains: list[list[tuple[float, float, int, int]]],
    *,
    shape: tuple[int, int],
    strain1: float,
    strain2: float,
) -> list[list[tuple[float, float, int, int]]] | None:
    nw, nv = shape
    propagated = [list(node_runs) for node_runs in domains]
    for _ in range(max(1, 2 * nw * nv)):
        lower = np.array([node_runs[0][0] for node_runs in propagated], dtype=np.float64)
        upper = np.array([node_runs[-1][1] for node_runs in propagated], dtype=np.float64)
        lower = lower.reshape(shape)
        upper = upper.reshape(shape)

        for _ in range(max(1, nw + nv)):
            bounds_changed = False
            for iw in range(nw):
                for iv in range(nv - 1):
                    bounds_changed |= _tighten_strain_bounds_pair(
                        lower,
                        upper,
                        (iw, iv),
                        (iw, iv + 1),
                        strain1,
                    )
            for iw in range(nw - 1):
                for iv in range(nv):
                    bounds_changed |= _tighten_strain_bounds_pair(
                        lower,
                        upper,
                        (iw, iv),
                        (iw + 1, iv),
                        strain2,
                    )
            if np.any(lower > upper):
                return None
            if not bounds_changed:
                break

        domains_changed = False
        for node_index, node_runs in enumerate(propagated):
            iw, iv = divmod(node_index, nv)
            narrowed_runs: list[tuple[float, float, int, int]] = []
            for run_lower, run_upper, run_start, run_stop in node_runs:
                narrowed_lower = max(run_lower, float(lower[iw, iv]))
                narrowed_upper = min(run_upper, float(upper[iw, iv]))
                if narrowed_lower <= narrowed_upper:
                    narrowed_runs.append(
                        (narrowed_lower, narrowed_upper, run_start, run_stop),
                    )
            if not narrowed_runs:
                return None
            if narrowed_runs != node_runs:
                propagated[node_index] = narrowed_runs
                domains_changed = True
        if not domains_changed:
            return propagated
    return propagated


def _surface_from_single_rounding_cell_runs(
    target: np.ndarray,
    valid_mask: np.ndarray,
    domains: list[list[tuple[float, float, int, int]]],
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
) -> np.ndarray | None:
    shape = target.shape
    lower = np.array([node_runs[0][0] for node_runs in domains], dtype=np.float64).reshape(
        shape,
    )
    upper = np.array([node_runs[0][1] for node_runs in domains], dtype=np.float64).reshape(
        shape,
    )
    strain1 = 1.0 / bstrain1
    strain2 = 1.0 / bstrain2
    for _ in range(max(1, 2 * sum(shape))):
        changed = False
        for iw in range(shape[0]):
            for iv in range(shape[1] - 1):
                changed |= _tighten_strain_bounds_pair(
                    lower,
                    upper,
                    (iw, iv),
                    (iw, iv + 1),
                    strain1,
                )
        for iw in range(shape[0] - 1):
            for iv in range(shape[1]):
                changed |= _tighten_strain_bounds_pair(
                    lower,
                    upper,
                    (iw, iv),
                    (iw + 1, iv),
                    strain2,
                )
        if np.any(lower > upper):
            return None
        if not changed:
            break

    width = upper - lower
    denominator = float(np.sum(width * width))
    if denominator > 0.0:
        alpha = float(np.sum((target.astype(np.float64) - lower) * width) / denominator)
        alpha = min(max(alpha, 0.0), 1.0)
    else:
        alpha = 0.0
    candidates = (
        lower + alpha * width,
        lower,
        upper,
        0.5 * (lower + upper),
    )
    for candidate in candidates:
        candidate_float32 = candidate.astype(np.float32)
        if _surface_respects_masked_strain(
            candidate_float32,
            valid_mask,
            lmin=lmin,
            bstrain1=bstrain1,
            bstrain2=bstrain2,
        ):
            return candidate_float32
    return None


def _distance_to_closed_interval(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def _tighten_strain_bounds_pair(
    lower: np.ndarray,
    upper: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    strain: float,
) -> bool:
    first_lower = lower[first]
    second_lower = lower[second]
    first_upper = upper[first]
    second_upper = upper[second]
    new_first_lower = max(first_lower, second_lower - strain)
    new_second_lower = max(second_lower, first_lower - strain)
    new_first_upper = min(first_upper, second_upper + strain)
    new_second_upper = min(second_upper, first_upper + strain)
    changed = (
        new_first_lower != first_lower
        or new_second_lower != second_lower
        or new_first_upper != first_upper
        or new_second_upper != second_upper
    )
    lower[first] = new_first_lower
    lower[second] = new_second_lower
    upper[first] = new_first_upper
    upper[second] = new_second_upper
    return changed

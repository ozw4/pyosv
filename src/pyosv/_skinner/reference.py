"""Reference-like fault-skinning orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyosv._skinner.connected import (
    ConnectedComponentSkinner,
    find_connected_component_skins,
)
from pyosv._skinner.grid import _SkinCellGrid
from pyosv._skinner.growth import _grow_reference_skin
from pyosv._skinner.seeds import (
    _adaptive_skin_likelihood_threshold,
    _find_reference_seeds,
    _mark_occupied_skin,
)
from pyosv._skinner.validation import (
    _validate_bool,
    _validate_connectivity,
    _validate_matching_finite_arrays3_many,
    _validate_nonnegative_finite_float,
    _validate_nonnegative_int,
    _validate_optional_nonnegative_int,
    _validate_unit_interval_float,
)
from pyosv.cells import FaultCell
from pyosv.skin import FaultSkin

__all__ = [
    "ConnectedComponentSkinner",
    "FaultSkinner",
    "find_connected_component_skins",
    "find_skins",
]

_UNSET = object()
_REFERENCE_SEED_MIN_EP = 0.8
_QUALITY_SEED_MIN_EP = 0.5
_QUALITY_DEFAULT_GROW_MIN_LIKELIHOOD = 0.5


class FaultSkinner:
    """Default fault skinner facade.

    ``method="reference"`` uses reference-like seed selection and local
    geometry-aware growth. ``method="quality"`` reuses that backend with a
    looser seed gate and adaptive likelihood threshold when no threshold is
    configured. ``method="connected_component"`` explicitly selects the legacy
    connected-component fallback.
    """

    def __init__(
        self,
        min_likelihood: float | object = _UNSET,
        min_skin_size: int | None = None,
        connectivity: str = "corner",
        method: str = "reference",
    ) -> None:
        self._min_likelihood_configured = min_likelihood is not _UNSET
        fallback_min_likelihood = 0.0 if min_likelihood is _UNSET else min_likelihood
        self._fallback = ConnectedComponentSkinner(
            min_likelihood=fallback_min_likelihood,  # type: ignore[arg-type]
            min_skin_size=min_skin_size,
            connectivity=connectivity,
        )
        self._method = _validate_skinner_method(method)

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        self._method = _validate_skinner_method(value)

    @property
    def min_likelihood(self) -> float:
        return self._fallback.min_likelihood

    @min_likelihood.setter
    def min_likelihood(self, value: float) -> None:
        self._fallback.min_likelihood = _validate_nonnegative_finite_float(
            value,
            "min_likelihood",
        )
        self._min_likelihood_configured = True

    @property
    def min_skin_size(self) -> int | None:
        return self._fallback.min_skin_size

    @min_skin_size.setter
    def min_skin_size(self, value: int | None) -> None:
        self._fallback.min_skin_size = _validate_optional_nonnegative_int(
            value,
            "min_skin_size",
        )

    @property
    def connectivity(self) -> str:
        return self._fallback.connectivity

    @connectivity.setter
    def connectivity(self, value: str) -> None:
        self._fallback.connectivity = _validate_connectivity(value)

    def cells_from_votes(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        min_likelihood: float | None = None,
    ) -> list[FaultCell]:
        """Extract fallback cells from thresholded voting outputs."""

        return self._fallback.cells_from_votes(fv, vp, vt, min_likelihood=min_likelihood)

    def find_seeds(
        self,
        d: int,
        fm: float,
        ep: np.ndarray,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> list[FaultCell]:
        """Find reference-like seed cells from thinned likelihood volumes."""

        return [
            cell.to_fault_cell()
            for cell in _find_reference_seeds(
                d=d,
                fm=fm,
                ep=ep,
                ft=ft,
                pt=pt,
                tt=tt,
            )
        ]

    def find_skins(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        min_likelihood: float | None = None,
        *,
        ep: np.ndarray | None = None,
        ft: np.ndarray | None = None,
        pt: np.ndarray | None = None,
        tt: np.ndarray | None = None,
        d: int = 1,
        ru: int = 150,
        rv: int | None = None,
        rw: int | None = None,
        max_steps: int = 10,
        du: float = 5.0,
        max_delta_strike: float = 30.0,
        reskin: bool = True,
        accepted_occupancy_radius: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[FaultSkin]:
        """Find skins with the configured backend."""

        should_reskin = _validate_bool(reskin, "reskin")
        occupancy_radius = (
            5
            if accepted_occupancy_radius is None
            else _validate_nonnegative_int(
                accepted_occupancy_radius,
                "accepted_occupancy_radius",
            )
        )
        if self.method == "connected_component":
            return self._fallback.find_skins(fv, vp, vt, min_likelihood=min_likelihood)

        seed_min_ep = _REFERENCE_SEED_MIN_EP
        effective_ep = fv if ep is None else ep
        effective_ft = fv if ft is None else ft
        effective_pt = vp if pt is None else pt
        effective_tt = vt if tt is None else tt
        grow_threshold: float | None = None
        if self.method == "quality":
            seed_min_ep = _QUALITY_SEED_MIN_EP
            if min_likelihood is None and not self._min_likelihood_configured:
                threshold = _adaptive_skin_likelihood_threshold(effective_ft)
                grow_threshold = _QUALITY_DEFAULT_GROW_MIN_LIKELIHOOD
            else:
                threshold = (
                    self.min_likelihood
                    if min_likelihood is None
                    else _validate_nonnegative_finite_float(min_likelihood, "min_likelihood")
                )
        else:
            threshold = (
                self.min_likelihood
                if min_likelihood is None
                else _validate_nonnegative_finite_float(min_likelihood, "min_likelihood")
            )
        return _find_reference_skins(
            fv=fv,
            vp=vp,
            vt=vt,
            ep=effective_ep,
            ft=effective_ft,
            pt=effective_pt,
            tt=effective_tt,
            d=d,
            fm=threshold,
            grow_fmin=grow_threshold,
            seed_min_ep=seed_min_ep,
            min_skin_size=self.min_skin_size,
            ru=ru,
            rv=rv,
            rw=rw,
            max_steps=max_steps,
            du=du,
            max_delta_strike=max_delta_strike,
            reskin=should_reskin,
            accepted_occupancy_radius=occupancy_radius,
            diagnostics=diagnostics,
        )

    def find_skin(
        self,
        seed: FaultCell,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        *,
        min_likelihood: float | None = None,
        ru: int = 150,
        rv: int | None = None,
        rw: int | None = None,
        max_steps: int = 10,
        du: float = 5.0,
        max_delta_strike: float = 30.0,
        reskin: bool = True,
    ) -> FaultSkin:
        """Grow one reference-like skin from ``seed`` without changing defaults."""

        threshold = self.min_likelihood if min_likelihood is None else min_likelihood
        return _grow_reference_skin(
            seed,
            fv,
            vp,
            vt,
            fmin=threshold,
            ru=ru,
            rv=rv,
            rw=rw,
            max_steps=max_steps,
            du=du,
            max_delta_strike=max_delta_strike,
            reskin=reskin,
        )


def find_skins(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    min_likelihood: float | None = None,
    *,
    ep: np.ndarray | None = None,
    ft: np.ndarray | None = None,
    pt: np.ndarray | None = None,
    tt: np.ndarray | None = None,
    d: int = 1,
    ru: int = 150,
    rv: int | None = None,
    rw: int | None = None,
    max_steps: int = 10,
    du: float = 5.0,
    max_delta_strike: float = 30.0,
    reskin: bool = True,
    accepted_occupancy_radius: int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> list[FaultSkin]:
    """Find reference-like skins from 3D voting outputs.

    This module-level convenience API uses the same reference-like backend as
    ``FaultSkinner(method="reference")``. Use
    ``find_connected_component_skins`` for the legacy connected-component
    fallback.
    """

    return FaultSkinner().find_skins(
        fv,
        vp,
        vt,
        min_likelihood=min_likelihood,
        ep=ep,
        ft=ft,
        pt=pt,
        tt=tt,
        d=d,
        ru=ru,
        rv=rv,
        rw=rw,
        max_steps=max_steps,
        du=du,
        max_delta_strike=max_delta_strike,
        reskin=reskin,
        accepted_occupancy_radius=accepted_occupancy_radius,
        diagnostics=diagnostics,
    )


def _find_reference_skins(
    *,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    ep: np.ndarray,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    d: int,
    fm: float,
    min_skin_size: int | None,
    ru: int,
    rv: int | None,
    rw: int | None,
    max_steps: int,
    du: float,
    max_delta_strike: float,
    reskin: bool,
    grow_fmin: float | None = None,
    seed_min_ep: float = _REFERENCE_SEED_MIN_EP,
    accepted_occupancy_radius: int = 5,
    diagnostics: dict[str, Any] | None = None,
) -> list[FaultSkin]:
    should_reskin = _validate_bool(reskin, "reskin")
    seed_threshold = _validate_nonnegative_finite_float(fm, "fm")
    planarity_threshold = _validate_unit_interval_float(seed_min_ep, "seed_min_ep")
    grow_threshold = (
        seed_threshold
        if grow_fmin is None
        else _validate_nonnegative_finite_float(grow_fmin, "grow_fmin")
    )
    occupancy_radius = _validate_nonnegative_int(
        accepted_occupancy_radius,
        "accepted_occupancy_radius",
    )
    fv_array, vp_array, vt_array, ep_array, ft_array, pt_array, tt_array = (
        _validate_matching_finite_arrays3_many(
            (fv, vp, vt, ep, ft, pt, tt),
            ("fv", "vp", "vt", "ep", "ft", "pt", "tt"),
        )
    )
    seed_candidate_count_before_spacing = int(
        np.count_nonzero(
            (ep_array > np.float32(planarity_threshold)) & (ft_array > np.float32(seed_threshold))
        )
    )
    seeds = _find_reference_seeds(
        d=d,
        fm=seed_threshold,
        ep=ep_array,
        ft=ft_array,
        pt=pt_array,
        tt=tt_array,
        min_ep=planarity_threshold,
    )
    skin_size = _validate_optional_nonnegative_int(min_skin_size, "min_skin_size")
    occupied = _SkinCellGrid()
    skins: list[FaultSkin] = []
    seed_count_rejected_by_occupied = 0
    grow_attempt_count = 0
    grown_skin_count_before_min_size = 0
    discarded_empty_skin_count = 0
    discarded_small_skin_count = 0

    for seed in seeds:
        if occupied.find_cells_in_box(seed.i1, seed.i2, seed.i3, 2, 2, 2):
            seed_count_rejected_by_occupied += 1
            continue

        grow_attempt_count += 1
        skin = _grow_reference_skin(
            seed,
            fv_array,
            vp_array,
            vt_array,
            fmin=grow_threshold,
            ru=ru,
            rv=rv,
            rw=rw,
            max_steps=max_steps,
            du=du,
            max_delta_strike=max_delta_strike,
            collision_grid=occupied,
            reskin=should_reskin,
        )
        skin_cell_count = len(skin)
        if skin_cell_count == 0:
            if skin_size is not None:
                discarded_empty_skin_count += 1
                continue
        else:
            grown_skin_count_before_min_size += 1
        if skin_size is not None and skin_cell_count < skin_size:
            discarded_small_skin_count += 1
            continue

        skins.append(skin)
        _mark_occupied_skin(occupied, skin, radius=occupancy_radius)

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "seed_candidate_count_before_spacing": seed_candidate_count_before_spacing,
                "seed_count_after_spacing": int(len(seeds)),
                "seed_count_rejected_by_occupied": int(seed_count_rejected_by_occupied),
                "grow_attempt_count": int(grow_attempt_count),
                "grown_skin_count_before_min_size": int(grown_skin_count_before_min_size),
                "discarded_empty_skin_count": int(discarded_empty_skin_count),
                "discarded_small_skin_count": int(discarded_small_skin_count),
                "accepted_skin_count": int(len(skins)),
                "accepted_cell_count": int(sum(len(skin) for skin in skins)),
                "accepted_occupancy_radius": int(occupancy_radius),
                "seed_min_ep": float(planarity_threshold),
                "seed_threshold": float(seed_threshold),
                "grow_threshold": float(grow_threshold),
            }
        )

    return skins


def _validate_skinner_method(method: str) -> str:
    if not isinstance(method, str) or method not in {
        "reference",
        "quality",
        "connected_component",
    }:
        raise ValueError("method must be 'reference', 'quality', or 'connected_component'")

    return method

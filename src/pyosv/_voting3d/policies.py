"""Seed-level strategies for 3D optimal-surface voting."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from pyosv._voting3d.context import SurfaceVotingContext
from pyosv._voting3d.models import _SurfaceVotingDiagnostic, _TangentialRectangle


class SurfaceVotingPolicy(Protocol):
    """Strategy interface for processing one voting seed."""

    def vote(self, context: SurfaceVotingContext) -> _SurfaceVotingDiagnostic: ...


class ReferenceSurfaceVotingPolicy:
    """Reference clamped-sampling and reference accumulation strategy."""

    def vote(self, context: SurfaceVotingContext) -> _SurfaceVotingDiagnostic:
        config = context.config
        cell = context.cell
        c1, c2, c3 = cell.index
        samples = context.sample_reference_with_support(
            c1, c2, c3, context.normal, context.dip, context.strike, context.ft
        )
        full_column_count = int((2 * config.rw + 1) * (2 * config.rv + 1))
        surface = context.find_surface(
            samples.cost,
            lmin=config.lmin,
            bstrain1=config.bstrain1,
            bstrain2=config.bstrain2,
            attribute_smoothing=config.attribute_smoothing,
            surface_smoothing1=config.surface_smoothing1,
            surface_smoothing2=config.surface_smoothing2,
        )

        fa, valid_count = context.score_reference(
            c1,
            c2,
            c3,
            config.rv,
            config.rw,
            context.normal,
            context.dip,
            context.strike,
            surface,
            context.ft,
        )
        surface_size = int(surface.size)
        surface_center_lag = context.surface_center_lag(surface)
        if valid_count == 0:
            return _diagnostic(
                context,
                policy="reference",
                full_column_count=full_column_count,
                selected_column_count=surface_size,
                admissible_lag_count=samples.admissible_lag_count,
                in_bounds_lag_count=samples.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=surface_center_lag,
                selected_invalid_sample_count=surface_size,
                skip_reason="no_valid_surface_samples",
            )
        if surface_size == 0:
            return _diagnostic(
                context,
                policy="reference",
                full_column_count=full_column_count,
                selected_column_count=0,
                admissible_lag_count=samples.admissible_lag_count,
                in_bounds_lag_count=samples.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=None,
                skip_reason="no_feasible_surface",
            )
        support_fraction = float(valid_count) / float(surface_size)
        if support_fraction < config.surface_support_min_fraction:
            return _diagnostic(
                context,
                policy="reference",
                full_column_count=full_column_count,
                selected_column_count=surface_size,
                admissible_lag_count=samples.admissible_lag_count,
                in_bounds_lag_count=samples.in_bounds_lag_count,
                support_fraction=support_fraction,
                surface_center_lag=surface_center_lag,
                selected_invalid_sample_count=surface_size - valid_count,
                skip_reason="support_below_min_fraction",
            )
        if config.surface_support_exponent > 0.0:
            fa = np.float32(fa * np.float32(support_fraction**config.surface_support_exponent))

        strike_angle, dip_angle = context.surface_orientation(
            context.normal,
            context.dip,
            context.strike,
            surface,
            sigma=config.surface_orientation_smoothing,
            backend=config.surface_orientation_backend,
        )
        align_i3 = abs(context.normal[2]) > abs(context.normal[1])
        context.accumulate_reference(
            c1,
            c2,
            c3,
            config.rv,
            config.rw,
            fa,
            np.float32(strike_angle),
            np.float32(dip_angle),
            align_i3,
            context.normal,
            context.dip,
            context.strike,
            surface,
            context.fe,
            context.vp,
            context.vt,
            context.vm,
        )
        face_count = context.count_reference_face_votes(
            c1,
            c2,
            c3,
            config.rv,
            config.rw,
            context.normal,
            context.dip,
            context.strike,
            surface,
            context.ft.shape,
        )
        return _SurfaceVotingDiagnostic(
            seed_index=cell.index,
            policy="reference",
            full_tangential_column_count=full_column_count,
            selected_tangential_column_count=surface_size,
            admissible_lag_count=samples.admissible_lag_count,
            in_bounds_lag_count=samples.in_bounds_lag_count,
            support_fraction=support_fraction,
            surface_center_lag=surface_center_lag,
            surface_projection_count=0,
            selected_invalid_sample_count=surface_size - valid_count,
            center_vote_write_count=valid_count,
            face_center_vote_count=face_count,
            orientation_source="surface",
            skipped=False,
            skip_reason=None,
        )


class MaskedInBoundsSurfaceVotingPolicy:
    """Masked in-bounds sampling, DP, and all-face accumulation strategy."""

    def vote(self, context: SurfaceVotingContext) -> _SurfaceVotingDiagnostic:
        config = context.config
        cell = context.cell
        c1, c2, c3 = cell.index
        full_samples = context.sample_masked(
            c1, c2, c3, context.normal, context.dip, context.strike, context.ft
        )
        full_nw, full_nv = full_samples.full_tangential_shape
        full_column_count = full_nw * full_nv
        supported_columns = np.any(full_samples.valid_lag_mask, axis=2)
        rectangle = context.select_supported_rectangle(
            supported_columns, origin_w=config.rw, origin_v=config.rv
        )
        if rectangle is None:
            return _diagnostic(
                context,
                policy="masked_in_bounds",
                full_column_count=full_column_count,
                selected_column_count=0,
                admissible_lag_count=full_samples.admissible_lag_count,
                in_bounds_lag_count=full_samples.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=None,
                skip_reason="no_supported_origin",
            )

        samples = context.crop_masked_box(full_samples, rectangle)
        surface, projection_count = context.find_surface_masked(
            samples.costs,
            samples.valid_lag_mask,
            lmin=config.lmin,
            bstrain1=config.bstrain1,
            bstrain2=config.bstrain2,
            attribute_smoothing=config.attribute_smoothing,
            surface_smoothing1=config.surface_smoothing1,
            surface_smoothing2=config.surface_smoothing2,
        )
        if surface is None:
            return _diagnostic(
                context,
                policy="masked_in_bounds",
                full_column_count=full_column_count,
                selected_column_count=rectangle.size,
                admissible_lag_count=full_samples.admissible_lag_count,
                in_bounds_lag_count=full_samples.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=None,
                projection_count=projection_count,
                skip_reason="no_feasible_surface",
            )

        surface_center_lag = float(
            surface[config.rw - samples.w_offset, config.rv - samples.v_offset]
        )
        fa, valid_count, invalid_count = context.score_masked(
            c1,
            c2,
            c3,
            config.rv,
            config.rw,
            samples.w_offset,
            samples.v_offset,
            config.lmin,
            context.normal,
            context.dip,
            context.strike,
            surface,
            samples.valid_lag_mask,
            context.ft,
        )
        support_fraction = float(valid_count) / float(full_column_count)
        common = dict(
            context=context,
            policy="masked_in_bounds",
            full_column_count=full_column_count,
            selected_column_count=rectangle.size,
            admissible_lag_count=full_samples.admissible_lag_count,
            in_bounds_lag_count=full_samples.in_bounds_lag_count,
            support_fraction=support_fraction,
            surface_center_lag=surface_center_lag,
            projection_count=projection_count,
        )
        if invalid_count > 0:
            return _diagnostic(
                **common,
                selected_invalid_sample_count=invalid_count,
                skip_reason="invalid_selected_sample",
            )
        if valid_count == 0:
            return _diagnostic(
                **common,
                skip_reason="no_valid_surface_samples",
            )
        if support_fraction < config.surface_support_min_fraction:
            return _diagnostic(**common, skip_reason="support_below_min_fraction")
        if config.surface_support_exponent > 0.0:
            fa = np.float32(fa * np.float32(support_fraction**config.surface_support_exponent))

        full_box = rectangle == _TangentialRectangle(0, 0, full_nw, full_nv)
        if full_box and surface.shape[0] >= 3 and surface.shape[1] >= 3:
            strike_angle, dip_angle = context.surface_orientation(
                context.normal,
                context.dip,
                context.strike,
                surface,
                sigma=config.surface_orientation_smoothing,
                backend=config.surface_orientation_backend,
            )
            orientation_source = "surface"
        else:
            strike_angle, dip_angle = cell.fp, cell.ft
            orientation_source = (
                "seed_boundary_fallback" if not full_box else "seed_small_surface_fallback"
            )

        center_count, face_count, accumulation_invalid_count = context.accumulate_masked(
            c1,
            c2,
            c3,
            config.rv,
            config.rw,
            samples.w_offset,
            samples.v_offset,
            config.lmin,
            fa,
            np.float32(strike_angle),
            np.float32(dip_angle),
            abs(context.normal[2]) > abs(context.normal[1]),
            context.normal,
            context.dip,
            context.strike,
            surface,
            samples.valid_lag_mask,
            context.fe,
            context.vp,
            context.vt,
            context.vm,
        )
        if accumulation_invalid_count > 0:
            return _diagnostic(
                **common,
                selected_invalid_sample_count=accumulation_invalid_count,
                orientation_source=orientation_source,
                skip_reason="invalid_selected_sample",
            )
        return _SurfaceVotingDiagnostic(
            seed_index=cell.index,
            policy="masked_in_bounds",
            full_tangential_column_count=full_column_count,
            selected_tangential_column_count=rectangle.size,
            admissible_lag_count=full_samples.admissible_lag_count,
            in_bounds_lag_count=full_samples.in_bounds_lag_count,
            support_fraction=support_fraction,
            surface_center_lag=surface_center_lag,
            surface_projection_count=projection_count,
            selected_invalid_sample_count=0,
            center_vote_write_count=center_count,
            face_center_vote_count=face_count,
            orientation_source=orientation_source,
            skipped=False,
            skip_reason=None,
        )


def _diagnostic(
    context: SurfaceVotingContext,
    *,
    policy: str,
    full_column_count: int,
    selected_column_count: int,
    admissible_lag_count: int,
    in_bounds_lag_count: int,
    support_fraction: float,
    surface_center_lag: float | None,
    projection_count: int = 0,
    selected_invalid_sample_count: int = 0,
    orientation_source: str | None = None,
    skip_reason: str,
) -> _SurfaceVotingDiagnostic:
    return _SurfaceVotingDiagnostic(
        seed_index=context.cell.index,
        policy=policy,
        full_tangential_column_count=full_column_count,
        selected_tangential_column_count=selected_column_count,
        admissible_lag_count=admissible_lag_count,
        in_bounds_lag_count=in_bounds_lag_count,
        support_fraction=support_fraction,
        surface_center_lag=surface_center_lag,
        surface_projection_count=projection_count,
        selected_invalid_sample_count=selected_invalid_sample_count,
        center_vote_write_count=0,
        face_center_vote_count=0,
        orientation_source=orientation_source,
        skipped=True,
        skip_reason=skip_reason,
    )


SURFACE_VOTING_POLICY_REGISTRY: Mapping[str, SurfaceVotingPolicy] = MappingProxyType(
    {
        "reference": ReferenceSurfaceVotingPolicy(),
        "masked_in_bounds": MaskedInBoundsSurfaceVotingPolicy(),
    }
)

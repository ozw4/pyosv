"""Truth-independent execution of the prepared-attribute 3D workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from pyosv._dp.path2d import strain_to_bstrain
from pyosv._voting3d.orientation import _SURFACE_ORIENTATION_BACKENDS
from pyosv._voting3d.policies import SURFACE_VOTING_POLICY_REGISTRY
from pyosv._voting3d.validation import (
    _validate_fraction_float,
    _validate_nonnegative_float,
)
from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticSkinningConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.models import OrientationField3D
from pyosv.evaluation.synthetic_quality.stage_cache import (
    AttributeStageKey,
    FinalThinningStageKey,
    FinalThinningStageResult,
    PipelineStageBuildTimer,
    PipelineStageCache,
    PrimarySkinningStageKey,
    PrimarySkinningStageResult,
    SeedStageKey,
    SeedStageResult,
    ThinningStageKey,
    ThinningStageResult,
    VotingStageKey,
    VotingStageResult,
    diagnostic_items,
    freeze_scalar_evidence,
)
from pyosv.evaluation.synthetic_quality.stage_keys import (
    DEFAULT_PRIMARY_SKINNER_IDENTITY,
    THIN_HYBRID_ORIENTATION_GRADIENT_THRESHOLD,
    THIN_HYBRID_V2_EDGE_MARGIN,
    THIN_PLATEAU_TOLERANCE,
    build_external_attribute_stage_key,
    build_final_thinning_stage_key,
    build_primary_skinning_stage_key,
    build_seed_stage_key,
    build_thinning_stage_key,
    build_voting_stage_key,
    resolve_stage_target_source,
)
from pyosv.evaluation.synthetic_quality.variants import (
    VariantSpec,
    effective_thin_mode,
)
from pyosv.experimental.boundary_seed_selection import select_boundary_seed_retention_v1
from pyosv.experimental.boundary_skinning import (
    apply_boundary_skinner_fallback,
    find_synthetic_skins,
)
from pyosv.experimental.boundary_thinning import (
    FVT_RECENTER_MAX_SHIFT,
    apply_boundary_edge_thin_v1,
    fvt_recenter_target_distance_diagnostics,
    recenter_edge_fvt_to_target,
)
from pyosv.experimental.skin_diagnostics import add_primary_skin_diagnostics
from pyosv.synthetic_metrics import skin_mask_from_skins
from pyosv.voting3d import (
    OptimalSurfaceVoter,
    _DEFAULT_FINAL_NORMALIZATION_SMOOTHING,
    _DEFAULT_SURFACE_ORIENTATION_BACKEND,
    _DEFAULT_SURFACE_SMOOTHING1,
    _DEFAULT_SURFACE_SMOOTHING2,
    _DEFAULT_SURFACE_VOTING_BOUNDARY_POLICY,
)

EDGE_FALSE_POSITIVE_MARGIN = quality_metrics.EDGE_FALSE_POSITIVE_MARGIN

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class PreparedAttributeIdentity:
    """Semantic identity for externally prepared ``ft/pt/tt`` attributes."""

    dataset_fingerprint: str
    stage_fingerprint: str
    shape: tuple[int, int, int]
    backend: str
    scanner_thin_mode: str
    edge_policy: str | bool

    def __post_init__(self) -> None:
        for name in ("dataset_fingerprint", "stage_fingerprint", "backend", "scanner_thin_mode"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 3
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1
                for size in self.shape
            )
        ):
            raise ValueError("shape must contain three positive integers")
        if not isinstance(self.edge_policy, (str, bool)):
            raise ValueError("edge_policy must be a string or bool")

    @property
    def stage_key(self) -> AttributeStageKey:
        """Return the cache key corresponding exactly to this identity."""

        return build_external_attribute_stage_key(
            dataset_fingerprint=self.dataset_fingerprint,
            stage_fingerprint=self.stage_fingerprint,
            shape=self.shape,
            backend=self.backend,
            scanner_thin_mode=self.scanner_thin_mode,
            edge_policy=self.edge_policy,
        )


@dataclass(frozen=True, slots=True)
class VolumeVotingControls:
    """Effective controls applied to one :class:`OptimalSurfaceVoter`."""

    strain_max1: float = 0.25
    strain_max2: float = 0.25
    surface_smoothing1: float = _DEFAULT_SURFACE_SMOOTHING1
    surface_smoothing2: float = _DEFAULT_SURFACE_SMOOTHING2
    boundary_policy: str = _DEFAULT_SURFACE_VOTING_BOUNDARY_POLICY
    support_min_fraction: float = 0.0
    support_exponent: float = 0.0
    orientation_smoothing: float = 0.0
    orientation_backend: str = _DEFAULT_SURFACE_ORIENTATION_BACKEND
    final_normalization_smoothing: float = _DEFAULT_FINAL_NORMALIZATION_SMOOTHING

    def __post_init__(self) -> None:
        for name in ("strain_max1", "strain_max2"):
            try:
                strain_to_bstrain(getattr(self, name))
            except ValueError as error:
                raise ValueError(f"{name} must satisfy 0 < {name} <= 1") from error
        for name in (
            "surface_smoothing1",
            "surface_smoothing2",
            "support_exponent",
            "orientation_smoothing",
            "final_normalization_smoothing",
        ):
            _validate_nonnegative_float(getattr(self, name), name)
        _validate_fraction_float(self.support_min_fraction, "support_min_fraction")
        if self.boundary_policy not in SURFACE_VOTING_POLICY_REGISTRY:
            allowed = ", ".join(repr(value) for value in SURFACE_VOTING_POLICY_REGISTRY)
            raise ValueError(f"boundary_policy must be one of: {allowed}")
        if self.orientation_backend not in _SURFACE_ORIENTATION_BACKENDS:
            raise ValueError(
                "orientation_backend must be one of "
                f"{_SURFACE_ORIENTATION_BACKENDS}, got {self.orientation_backend!r}"
            )

    @property
    def bstrain1(self) -> int:
        return strain_to_bstrain(self.strain_max1)

    @property
    def bstrain2(self) -> int:
        return strain_to_bstrain(self.strain_max2)

    @classmethod
    def resolve(
        cls,
        voting_settings: SyntheticVotingConfig,
        variant_spec: VariantSpec,
    ) -> VolumeVotingControls:
        """Resolve established defaults plus a synthetic variant patch."""

        patch = variant_spec.voting
        return cls(
            boundary_policy=(
                _DEFAULT_SURFACE_VOTING_BOUNDARY_POLICY
                if patch.boundary_policy is None
                else patch.boundary_policy
            ),
            support_min_fraction=(
                voting_settings.surface_support_min_fraction
                if patch.support_min_fraction is None
                else patch.support_min_fraction
            ),
            support_exponent=(
                voting_settings.surface_support_exponent
                if patch.support_exponent is None
                else patch.support_exponent
            ),
            orientation_smoothing=(
                float(max(voting_settings.rv, voting_settings.rw))
                if patch.orientation_smoothing is None
                else patch.orientation_smoothing
            ),
            final_normalization_smoothing=(
                _DEFAULT_FINAL_NORMALIZATION_SMOOTHING
                if patch.final_normalization_smoothing is None
                else patch.final_normalization_smoothing
            ),
        )


@dataclass(frozen=True, slots=True)
class Workflow3DStageKeys:
    attribute: AttributeStageKey | None
    seed: SeedStageKey | None
    voting: VotingStageKey | None
    thinning: ThinningStageKey | None
    final_thinning: FinalThinningStageKey | None
    primary_skinning: PrimarySkinningStageKey | None


@dataclass(frozen=True, slots=True)
class Workflow3DEffectiveSettings:
    voting: SyntheticVotingConfig
    controls: VolumeVotingControls
    skinning: SyntheticSkinningConfig
    variant: VariantSpec
    thin_mode: str


@dataclass(frozen=True, slots=True)
class Workflow3DDiagnostics:
    seed: Mapping[str, Any] | None
    voting: Mapping[str, Any]
    thinning: Mapping[str, Any]
    skinning: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Workflow3DSkinResult:
    enabled: bool
    skins: tuple[Any, ...]
    primary_mask: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Workflow3DResult:
    """Owned numerical result of one truth-independent workflow execution."""

    fv: np.ndarray
    vp: np.ndarray
    vt: np.ndarray
    fvt: np.ndarray
    skin: Workflow3DSkinResult
    diagnostics: Workflow3DDiagnostics
    effective_settings: Workflow3DEffectiveSettings
    stage_keys: Workflow3DStageKeys
    seed_indices: tuple[tuple[int, int, int], ...]
    voter: OptimalSurfaceVoter

    @property
    def skins(self) -> tuple[Any, ...]:
        return self.skin.skins


def _time_build(
    *,
    stage: str,
    semantic_key: Any,
    builder: Callable[[], _T],
    timer: PipelineStageBuildTimer | None,
) -> _T:
    return builder() if timer is None else timer(stage, semantic_key, builder)


def _cache_timer(
    stage_cache: PipelineStageCache | None,
    stage_timer: PipelineStageBuildTimer | None,
    *,
    cache_enabled: bool,
) -> PipelineStageBuildTimer | None:
    if stage_timer is not None:
        return stage_timer
    if cache_enabled and stage_cache is not None:
        return stage_cache.build_timer
    return None


def _clone_array(value: np.ndarray) -> np.ndarray:
    return np.array(value, dtype=np.float32, copy=True, order="C")


def _readonly_clone(value: np.ndarray) -> np.ndarray:
    result = _clone_array(value)
    result.flags.writeable = False
    return result


def _readonly_bool_clone(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=bool, copy=True, order="C")
    result.flags.writeable = False
    return result


def _resolve_primary_skinner_identity(
    primary_skinner: Callable[..., list[Any]],
    explicit_identity: str | None,
) -> str | None:
    if explicit_identity is not None:
        if not isinstance(explicit_identity, str) or not explicit_identity.strip():
            raise ValueError("primary_skinner_identity must be a non-empty string")
        return explicit_identity
    if primary_skinner is find_synthetic_skins:
        return DEFAULT_PRIMARY_SKINNER_IDENTITY
    return None


def execute_workflow3d(
    *,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    attribute_identity: PreparedAttributeIdentity | AttributeStageKey | None,
    voting_settings: SyntheticVotingConfig,
    voting_controls: VolumeVotingControls,
    skinning_settings: SyntheticSkinningConfig,
    variant_spec: VariantSpec = VariantSpec("generic", experimental=False),
    stage_cache: PipelineStageCache | None = None,
    stage_timer: PipelineStageBuildTimer | None = None,
    scanner_target_positive_mask: np.ndarray | None = None,
    fvt_recenter_target: np.ndarray | None = None,
    fvt_recenter_target_source: str | None = None,
    recenter_distance_diagnostic_runner: Callable[
        ..., dict[str, float | None]
    ] = fvt_recenter_target_distance_diagnostics,
    primary_skinner: Callable[..., list[Any]] = find_synthetic_skins,
    primary_skinner_identity: str | None = None,
    boundary_fallback_runner: Callable[..., None] = apply_boundary_skinner_fallback,
) -> Workflow3DResult:
    """Execute seed selection through skinning without truth or reference data."""

    field = OrientationField3D(ft=ft, pt=pt, tt=tt)
    if not isinstance(voting_settings, SyntheticVotingConfig):
        raise TypeError("voting_settings must be a SyntheticVotingConfig")
    if not isinstance(voting_controls, VolumeVotingControls):
        raise TypeError("voting_controls must be VolumeVotingControls")
    if not isinstance(skinning_settings, SyntheticSkinningConfig):
        raise TypeError("skinning_settings must be a SyntheticSkinningConfig")
    if not isinstance(variant_spec, VariantSpec):
        raise TypeError("variant_spec must be a VariantSpec")

    if isinstance(attribute_identity, PreparedAttributeIdentity):
        if attribute_identity.shape != field.ft.shape:
            raise ValueError(
                f"attribute identity shape {attribute_identity.shape} does not match "
                f"attribute shape {field.ft.shape}"
            )
        attribute_key = attribute_identity.stage_key
    elif isinstance(attribute_identity, AttributeStageKey):
        if attribute_identity.shape != field.ft.shape:
            raise ValueError(
                f"attribute identity shape {attribute_identity.shape} does not match "
                f"attribute shape {field.ft.shape}"
            )
        attribute_key = attribute_identity
    elif attribute_identity is None:
        attribute_key = None
    else:
        raise TypeError(
            "attribute_identity must be PreparedAttributeIdentity, AttributeStageKey, or None"
        )

    voter = OptimalSurfaceVoter(
        ru=voting_settings.ru,
        rv=voting_settings.rv,
        rw=voting_settings.rw,
    )
    voter.set_strain_max(voting_controls.strain_max1, voting_controls.strain_max2)
    voter.set_attribute_smoothing(voting_settings.attribute_smoothing)
    voter.set_surface_smoothing(
        voting_controls.surface_smoothing1,
        voting_controls.surface_smoothing2,
    )
    voter.set_surface_support_policy(
        min_fraction=voting_controls.support_min_fraction,
        exponent=voting_controls.support_exponent,
    )
    voter.set_surface_voting_boundary_policy(voting_controls.boundary_policy)
    voter.set_surface_orientation_smoothing(voting_controls.orientation_smoothing)
    voter.set_surface_orientation_backend(voting_controls.orientation_backend)
    voter.set_final_normalization_smoothing(voting_controls.final_normalization_smoothing)

    target_is_external = fvt_recenter_target is not None and fvt_recenter_target is not ft
    seed_key_safe = attribute_key is not None and not (
        variant_spec.seed_policy == "boundary_seed_retention_v1" and target_is_external
    )
    cache_enabled = stage_cache is not None and seed_key_safe
    final_thinning_key_safe = seed_key_safe and not (
        variant_spec.post_thinning_policy != "none" and target_is_external
    )
    timer = _cache_timer(stage_cache, stage_timer, cache_enabled=cache_enabled)

    seed_key = build_seed_stage_key(
        attribute_key=attribute_key if seed_key_safe else None,
        voting_config=voting_settings,
        variant_spec=variant_spec,
        target_source=fvt_recenter_target_source,
    )
    voting_key = build_voting_stage_key(
        seed_key=seed_key,
        voting_config=voting_settings,
        variant_spec=variant_spec,
        voting_controls=voting_controls,
    )
    boundary_target_source = (
        resolve_stage_target_source(fvt_recenter_target_source)
        if variant_spec.seed_policy == "boundary_seed_retention_v1"
        else None
    )

    def build_seed_result() -> SeedStageResult:
        if variant_spec.seed_policy == "boundary_seed_retention_v1":
            selected = select_boundary_seed_retention_v1(
                voting_config=voting_settings,
                ft=ft,
                pt=pt,
                tt=tt,
                target=ft if fvt_recenter_target is None else fvt_recenter_target,
                target_source=boundary_target_source,
                edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            )
            return SeedStageResult(
                seeds=tuple(selected.selected_seeds),
                diagnostic_items=diagnostic_items(selected.diagnostics),
            )
        return SeedStageResult(
            seeds=tuple(
                voter.pick_seeds(
                    d=voting_settings.seed_distance,
                    fm=voting_settings.seed_threshold,
                    ft=ft,
                    pt=pt,
                    tt=tt,
                )
            )
        )

    voting_result = (
        stage_cache.get_voting(voting_key)
        if cache_enabled
        and stage_cache is not None
        and seed_key is not None
        and voting_key is not None
        else None
    )
    seed_result: SeedStageResult | None = None
    if voting_result is None:
        if cache_enabled and stage_cache is not None and seed_key is not None:
            seed_result = stage_cache.get_seed(seed_key)
            if seed_result is None:
                seed_result = _time_build(
                    stage="seed_selection",
                    semantic_key=seed_key,
                    builder=build_seed_result,
                    timer=timer,
                )
                stage_cache.put_seed(seed_key, seed_result)
        else:
            seed_result = _time_build(
                stage="seed_selection",
                semantic_key=seed_key,
                builder=build_seed_result,
                timer=stage_timer,
            )
        assert seed_result is not None

        def build_voting_result() -> VotingStageResult:
            built_fv, built_vp, built_vt = voter.apply_voting_from_seeds(
                seed_result.seeds,
                ft=ft,
                pt=pt,
                tt=tt,
            )
            return VotingStageResult(
                fv=built_fv,
                vp=built_vp,
                vt=built_vt,
                diagnostic_items=diagnostic_items(voter.surface_voting_diagnostic_summary()),
            )

        if cache_enabled and stage_cache is not None and voting_key is not None:
            voting_result = _time_build(
                stage="voting_volume",
                semantic_key=voting_key,
                builder=build_voting_result,
                timer=timer,
            )
            stage_cache.put_voting(voting_key, voting_result)
        else:
            voting_result = _time_build(
                stage="voting_volume",
                semantic_key=voting_key,
                builder=build_voting_result,
                timer=stage_timer,
            )
    elif stage_cache is not None and seed_key is not None:
        # A persisted voting artifact does not contain seed coordinates. Reuse
        # an in-memory seed result when present, but never rebuild it solely for
        # downstream stages.
        seed_result = stage_cache.get_seed(seed_key)
    fv = voting_result.fv
    vp = voting_result.vp
    vt = voting_result.vt
    voting_diagnostics = voting_result.diagnostics()

    thin_mode = effective_thin_mode(variant_spec, voting_settings)
    thinning_key = build_thinning_stage_key(
        voting_key=voting_key,
        voting_config=voting_settings,
        variant_spec=variant_spec,
    )
    final_thinning_key = build_final_thinning_stage_key(
        thinning_key=thinning_key if final_thinning_key_safe else None,
        variant_spec=variant_spec,
        target_source=fvt_recenter_target_source,
    )
    final_thinning_result = (
        None
        if stage_cache is None or final_thinning_key is None
        else stage_cache.get_final_thinning(final_thinning_key)
    )

    def build_thinning_result() -> ThinningStageResult:
        return ThinningStageResult(
            fvt=voter.thin(
                fv,
                vp,
                vt,
                mode=thin_mode,
                reference_sigma=voting_settings.reference_thin_sigma,
                hybrid_orientation_gradient_threshold=(THIN_HYBRID_ORIENTATION_GRADIENT_THRESHOLD),
                hybrid_v2_edge_margin=THIN_HYBRID_V2_EDGE_MARGIN,
                plateau_tie_breaker=(ft if thin_mode in {"hybrid_v2", "normal_plateau"} else None),
                plateau_tolerance=THIN_PLATEAU_TOLERANCE,
            )
        )

    if final_thinning_result is not None:
        fvt = _clone_array(final_thinning_result.fvt)
        final_diagnostics = final_thinning_result.diagnostics()
        recenter_diagnostic = final_diagnostics.get("recenter")
        boundary_thin_diagnostic = final_diagnostics.get("boundary_edge_thin")
    else:
        if cache_enabled and stage_cache is not None and thinning_key is not None:
            thinning_result = stage_cache.get_thinning(thinning_key)
            if thinning_result is None:
                thinning_result = _time_build(
                    stage="base_thinning",
                    semantic_key=thinning_key,
                    builder=build_thinning_result,
                    timer=timer,
                )
                stage_cache.put_thinning(thinning_key, thinning_result)
        else:
            thinning_result = _time_build(
                stage="base_thinning",
                semantic_key=thinning_key,
                builder=build_thinning_result,
                timer=stage_timer,
            )
        fvt = _clone_array(thinning_result.fvt)

        recenter_diagnostic = None
        boundary_thin_diagnostic = None
        if variant_spec.post_thinning_policy == "recenter_scanner_target":
            recenter_before = quality_metrics.positive_candidate_mask(fvt)
            recenter_target = ft if fvt_recenter_target is None else fvt_recenter_target
            recenter_result = recenter_edge_fvt_to_target(
                fvt,
                vp,
                vt,
                target=recenter_target,
                target_source=resolve_stage_target_source(fvt_recenter_target_source),
                max_shift=FVT_RECENTER_MAX_SHIFT,
                edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            )
            fvt = recenter_result.output
            recenter_diagnostic = recenter_result.diagnostics
            recenter_diagnostic.update(
                recenter_distance_diagnostic_runner(
                    before=recenter_before,
                    after=quality_metrics.positive_candidate_mask(fvt),
                    target=quality_metrics.positive_candidate_mask(recenter_target),
                )
            )
        elif variant_spec.post_thinning_policy == "boundary_edge_thin_v1":
            boundary_result = apply_boundary_edge_thin_v1(
                fvt,
                fv,
                vp,
                vt,
                voter=voter,
                target=ft if fvt_recenter_target is None else fvt_recenter_target,
                target_source=resolve_stage_target_source(fvt_recenter_target_source),
                edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            )
            fvt = boundary_result.output
            boundary_thin_diagnostic = boundary_result.diagnostics
        final_thinning_diagnostics = {
            "recenter": recenter_diagnostic,
            "boundary_edge_thin": boundary_thin_diagnostic,
        }
        if cache_enabled and stage_cache is not None and final_thinning_key is not None:
            stage_cache.put_final_thinning(
                final_thinning_key,
                FinalThinningStageResult(
                    fvt=_clone_array(fvt),
                    diagnostic_items=diagnostic_items(final_thinning_diagnostics),
                ),
            )
    resolved_primary_skinner_identity = _resolve_primary_skinner_identity(
        primary_skinner,
        primary_skinner_identity,
    )
    primary_key = (
        None
        if resolved_primary_skinner_identity is None
        else build_primary_skinning_stage_key(
            thinning_key=thinning_key if final_thinning_key_safe else None,
            skinning_config=skinning_settings,
            variant_spec=variant_spec,
            target_source=fvt_recenter_target_source,
            skinner_identity=resolved_primary_skinner_identity,
        )
    )
    primary_cache_enabled = stage_cache is not None and primary_key is not None
    if skinning_settings.enabled:
        primary_diagnostic_candidate_count = int(
            np.count_nonzero(quality_metrics.positive_candidate_mask(fvt))
        )

        def build_primary_result() -> PrimarySkinningStageResult:
            built_diagnostics: dict[str, Any] = {}
            built_skins = primary_skinner(
                fv,
                fvt,
                vp,
                vt,
                skinning_config=skinning_settings,
                diagnostics=built_diagnostics,
            )
            return PrimarySkinningStageResult.from_skins(
                built_skins,
                built_diagnostics,
            )

        if primary_cache_enabled and stage_cache is not None and primary_key is not None:
            primary_result = stage_cache.get_primary_skinning(primary_key)
            if primary_result is None:
                primary_result = _time_build(
                    stage="primary_skinning",
                    semantic_key=primary_key,
                    builder=build_primary_result,
                    timer=timer,
                )
                stage_cache.put_primary_skinning(primary_key, primary_result)
        else:
            primary_result = _time_build(
                stage="primary_skinning",
                semantic_key=primary_key,
                builder=build_primary_result,
                timer=stage_timer,
            )
        skins, skin_diagnostics = primary_result.clone()
        add_primary_skin_diagnostics(
            skin_diagnostics,
            skins,
            shape=field.ft.shape,
            fvt_positive_candidate_count=primary_diagnostic_candidate_count,
            small_skin_size=skinning_settings.small_skin_size,
        )
        primary_mask = skin_mask_from_skins(skins, field.ft.shape)
        boundary_fallback_runner(
            skins,
            fvt,
            vp,
            vt,
            skinning_config=skinning_settings,
            variant_spec=variant_spec,
            diagnostics=skin_diagnostics,
            scanner_target_positive_mask=scanner_target_positive_mask,
        )
    else:
        skins = []
        skin_diagnostics = {}
        primary_mask = np.zeros(field.ft.shape, dtype=bool)

    frozen_seed = None
    if seed_result is not None and seed_result.diagnostics() is not None:
        frozen_seed = freeze_scalar_evidence(seed_result.diagnostics(), "seed")
    frozen_voting = freeze_scalar_evidence(voting_diagnostics, "voting")
    frozen_thinning = freeze_scalar_evidence(
        {
            "recenter": recenter_diagnostic,
            "boundary_edge_thin": boundary_thin_diagnostic,
        },
        "thinning",
    )
    frozen_skinning = freeze_scalar_evidence(skin_diagnostics, "skinning")
    return Workflow3DResult(
        fv=_readonly_clone(fv),
        vp=_readonly_clone(vp),
        vt=_readonly_clone(vt),
        fvt=_readonly_clone(fvt),
        skin=Workflow3DSkinResult(
            enabled=skinning_settings.enabled,
            skins=tuple(skins),
            primary_mask=_readonly_bool_clone(primary_mask),
            diagnostics=frozen_skinning,
        ),
        diagnostics=Workflow3DDiagnostics(
            seed=frozen_seed,
            voting=frozen_voting,
            thinning=frozen_thinning,
            skinning=frozen_skinning,
        ),
        effective_settings=Workflow3DEffectiveSettings(
            voting=voting_settings,
            controls=voting_controls,
            skinning=skinning_settings,
            variant=variant_spec,
            thin_mode=thin_mode,
        ),
        stage_keys=Workflow3DStageKeys(
            attribute=attribute_key,
            seed=seed_key,
            voting=voting_key,
            thinning=thinning_key,
            final_thinning=final_thinning_key,
            primary_skinning=primary_key,
        ),
        seed_indices=(
            ()
            if seed_result is None
            else tuple((int(seed.i3), int(seed.i2), int(seed.i1)) for seed in seed_result.seeds)
        ),
        voter=voter,
    )


run_workflow3d = execute_workflow3d

__all__ = [
    "PreparedAttributeIdentity",
    "VolumeVotingControls",
    "Workflow3DDiagnostics",
    "Workflow3DEffectiveSettings",
    "Workflow3DResult",
    "Workflow3DSkinResult",
    "Workflow3DStageKeys",
    "build_external_attribute_stage_key",
    "execute_workflow3d",
    "run_workflow3d",
]

"""Pure construction of semantic synthetic-quality pipeline stage keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.experimental.boundary_thinning import FVT_RECENTER_MAX_SHIFT

from .config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from .stage_cache import (
    AttributeStageKey,
    FinalThinningStageKey,
    PrimarySkinningStageKey,
    SCALAR_EVIDENCE_CONTRACT_VERSION,
    SeedStageKey,
    ThinningScalarEvidenceKey,
    ThinningStageKey,
    VotingScalarEvidenceKey,
    VotingStageKey,
)
from .variants import VariantSpec, effective_thin_mode

if TYPE_CHECKING:
    from pyosv.evaluation.workflow3d import VolumeVotingControls

THIN_HYBRID_ORIENTATION_GRADIENT_THRESHOLD = 8.0
THIN_HYBRID_V2_EDGE_MARGIN = 2
THIN_PLATEAU_TOLERANCE = 1.0e-6
DEFAULT_PRIMARY_SKINNER_IDENTITY = "pyosv.experimental.boundary_skinning.find_synthetic_skins"


@dataclass(frozen=True, slots=True)
class PipelineStageKeys:
    """Semantic keys looked up by one downstream pipeline execution."""

    attribute: AttributeStageKey | None
    seed: SeedStageKey | None
    voting: VotingStageKey | None
    thinning: ThinningStageKey | None
    primary_skinning: PrimarySkinningStageKey | None


def resolve_stage_target_source(target_source: str | None) -> str:
    """Resolve the semantic name used for a post-attribute target."""

    return "ft_input" if target_source is None else target_source


def build_oracle_attribute_stage_key(
    *, case_id: str, shape: tuple[int, int, int]
) -> AttributeStageKey:
    """Build the key for case-owned oracle attributes."""

    return AttributeStageKey(case_id=case_id, shape=shape, source="oracle")


def build_scanner_attribute_stage_key(
    *,
    case_id: str,
    shape: tuple[int, int, int],
    scanner_config: SyntheticScannerConfig,
) -> AttributeStageKey:
    """Build the key for one scanner backend's prepared attributes."""

    input_config = scanner_config.input_config
    return AttributeStageKey(
        case_id=case_id,
        shape=shape,
        source="scanner",
        settings=(
            ("backend", scanner_config.backend),
            ("phi_min", float(scanner_config.phi_min)),
            ("phi_max", float(scanner_config.phi_max)),
            ("theta_min", float(scanner_config.theta_min)),
            ("theta_max", float(scanner_config.theta_max)),
            ("sigma1", float(scanner_config.sigma1)),
            ("sigma2", float(scanner_config.sigma2)),
            ("refinement_factor", int(scanner_config.refinement_factor)),
            ("scanner_thin_mode", scanner_config.scanner_thin_mode),
            ("remove_edge_effects", scanner_config.remove_edge_effects),
            ("input_background", float(input_config.background)),
            ("input_fault_contrast", float(input_config.fault_contrast)),
            ("input_noise_sigma", float(input_config.noise_sigma)),
            ("input_seed", int(input_config.seed)),
            ("input_clip_min", float(input_config.clip_min)),
            ("input_clip_max", float(input_config.clip_max)),
        ),
    )


def build_external_attribute_stage_key(
    *,
    dataset_fingerprint: str,
    stage_fingerprint: str,
    shape: tuple[int, int, int],
    backend: str,
    scanner_thin_mode: str,
    edge_policy: str | bool,
) -> AttributeStageKey:
    """Build an attribute key without inventing synthetic-case configuration."""

    return AttributeStageKey(
        case_id=dataset_fingerprint,
        shape=shape,
        source="external",
        settings=(
            ("stage_fingerprint", stage_fingerprint),
            ("backend", backend),
            ("scanner_thin_mode", scanner_thin_mode),
            ("edge_policy", edge_policy),
        ),
    )


def build_seed_stage_key(
    *,
    attribute_key: AttributeStageKey | None,
    voting_config: SyntheticVotingConfig,
    variant_spec: VariantSpec,
    target_source: str | None,
) -> SeedStageKey | None:
    """Build the key for the effective seed-selection inputs."""

    if attribute_key is None:
        return None
    boundary_target_source = None
    boundary_edge_margin = None
    if variant_spec.seed_policy == "boundary_seed_retention_v1":
        boundary_target_source = resolve_stage_target_source(target_source)
        boundary_edge_margin = quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
    return SeedStageKey(
        attributes=attribute_key,
        seed_policy=variant_spec.seed_policy,
        seed_distance=int(voting_config.seed_distance),
        seed_threshold=float(voting_config.seed_threshold),
        ru=int(voting_config.ru),
        rv=int(voting_config.rv),
        rw=int(voting_config.rw),
        boundary_target_source=boundary_target_source,
        boundary_edge_margin=boundary_edge_margin,
    )


def build_voting_stage_key(
    *,
    seed_key: SeedStageKey | None,
    voting_config: SyntheticVotingConfig,
    variant_spec: VariantSpec,
    voting_controls: VolumeVotingControls,
) -> VotingStageKey | None:
    """Build the key for effective surface-voting settings."""

    if seed_key is None:
        return None
    return VotingStageKey(
        seed=seed_key,
        ru=int(voting_config.ru),
        rv=int(voting_config.rv),
        rw=int(voting_config.rw),
        bstrain1=int(voting_controls.bstrain1),
        bstrain2=int(voting_controls.bstrain2),
        attribute_smoothing=int(voting_config.attribute_smoothing),
        surface_smoothing1=float(voting_controls.surface_smoothing1),
        surface_smoothing2=float(voting_controls.surface_smoothing2),
        boundary_policy=voting_controls.boundary_policy,
        support_min_fraction=float(voting_controls.support_min_fraction),
        support_exponent=float(voting_controls.support_exponent),
        orientation_smoothing=float(voting_controls.orientation_smoothing),
        orientation_backend=voting_controls.orientation_backend,
        final_normalization_smoothing=float(voting_controls.final_normalization_smoothing),
    )


def build_thinning_stage_key(
    *,
    voting_key: VotingStageKey | None,
    voting_config: SyntheticVotingConfig,
    variant_spec: VariantSpec,
) -> ThinningStageKey | None:
    """Build the key for effective base-thinning inputs."""

    if voting_key is None:
        return None
    thin_mode = effective_thin_mode(variant_spec, voting_config)
    return ThinningStageKey(
        voting=voting_key,
        thin_mode=thin_mode,
        reference_sigma=float(voting_config.reference_thin_sigma),
        hybrid_orientation_gradient_threshold=THIN_HYBRID_ORIENTATION_GRADIENT_THRESHOLD,
        hybrid_v2_edge_margin=THIN_HYBRID_V2_EDGE_MARGIN,
        orientation_source="voting_vp_vt",
        tie_break_policy=(
            "attribute_ft" if thin_mode in {"hybrid_v2", "normal_plateau"} else "voting_fv"
        ),
        plateau_tolerance=THIN_PLATEAU_TOLERANCE,
    )


def build_final_thinning_stage_key(
    *,
    thinning_key: ThinningStageKey | None,
    variant_spec: VariantSpec,
    target_source: str | None,
) -> FinalThinningStageKey | None:
    """Build the identity of final ``fvt`` including post-thinning semantics."""

    if thinning_key is None:
        return None
    post_thinning_target_source = None
    if variant_spec.post_thinning_policy != "none":
        post_thinning_target_source = resolve_stage_target_source(target_source)
    return FinalThinningStageKey(
        thinning=thinning_key,
        post_thinning_policy=variant_spec.post_thinning_policy,
        post_thinning_target_source=post_thinning_target_source,
        post_thinning_max_shift=(
            FVT_RECENTER_MAX_SHIFT
            if variant_spec.post_thinning_policy == "recenter_scanner_target"
            else None
        ),
        post_thinning_edge_margin=(
            quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
            if variant_spec.post_thinning_policy != "none"
            else None
        ),
    )


def build_voting_scalar_evidence_key(
    *,
    case_id: str,
    case_token: int,
    shape: tuple[int, int, int],
    voting_key: VotingStageKey | None,
    truth_metric_config: SyntheticTruthMetricConfig,
    contract_version: int = SCALAR_EVIDENCE_CONTRACT_VERSION,
) -> VotingScalarEvidenceKey | None:
    """Build the complete identity for reusable voting scalar evidence."""

    if voting_key is None:
        return None
    return VotingScalarEvidenceKey(
        case_id=case_id,
        case_token=case_token,
        shape=shape,
        voting=voting_key,
        truth_metric_config=truth_metric_config,
        contract_version=contract_version,
    )


def build_thinning_scalar_evidence_key(
    *,
    case_id: str,
    case_token: int,
    shape: tuple[int, int, int],
    final_thinning_key: FinalThinningStageKey | None,
    truth_metric_config: SyntheticTruthMetricConfig,
    contract_version: int = SCALAR_EVIDENCE_CONTRACT_VERSION,
) -> ThinningScalarEvidenceKey | None:
    """Build the complete identity for reusable final-thinning evidence."""

    if final_thinning_key is None:
        return None
    return ThinningScalarEvidenceKey(
        case_id=case_id,
        case_token=case_token,
        shape=shape,
        thinning=final_thinning_key,
        truth_metric_config=truth_metric_config,
        contract_version=contract_version,
    )


def build_primary_skinning_stage_key(
    *,
    thinning_key: ThinningStageKey | None,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    target_source: str | None,
    skinner_identity: str = DEFAULT_PRIMARY_SKINNER_IDENTITY,
) -> PrimarySkinningStageKey | None:
    """Build the key for effective primary skin-growth inputs."""

    if thinning_key is None or not skinning_config.enabled:
        return None
    post_thinning_target_source = None
    if variant_spec.post_thinning_policy != "none":
        post_thinning_target_source = resolve_stage_target_source(target_source)
    return PrimarySkinningStageKey(
        thinning=thinning_key,
        skinner_identity=skinner_identity,
        post_thinning_policy=variant_spec.post_thinning_policy,
        post_thinning_target_source=post_thinning_target_source,
        post_thinning_max_shift=(
            FVT_RECENTER_MAX_SHIFT
            if variant_spec.post_thinning_policy == "recenter_scanner_target"
            else None
        ),
        post_thinning_edge_margin=(
            quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
            if variant_spec.post_thinning_policy != "none"
            else None
        ),
        method=skinning_config.method,
        growth_source=skinning_config.growth_source,
        min_likelihood=skinning_config.min_likelihood,
        min_skin_size=skinning_config.min_skin_size,
        d=skinning_config.d,
        ru=skinning_config.ru,
        rv=skinning_config.rv,
        rw=skinning_config.rw,
        max_steps=skinning_config.max_steps,
        du=skinning_config.du,
        max_delta_strike=skinning_config.max_delta_strike,
        reskin=skinning_config.reskin,
        reskin_policy=skinning_config.reskin_policy,
        accepted_occupancy_radius=skinning_config.accepted_occupancy_radius,
        small_skin_size=skinning_config.small_skin_size,
    )

"""Case-local cache keys and values for synthetic-quality pipeline stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np

from pyosv.cells import FaultCell
from pyosv.skin import FaultSkin

if TYPE_CHECKING:
    from pyosv.evaluation.synthetic_quality.config import SyntheticTruthMetricConfig
    from pyosv.synthetic3d import Synthetic3DCase


ScalarStageSetting = str | int | float | bool
DiagnosticItems = tuple[tuple[str, Any], ...]
SCALAR_EVIDENCE_CONTRACT_VERSION = 5


class ImmutableScalarMapping(dict[str, Any]):
    """JSON-serializable mapping that cannot be changed after construction."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("downstream scalar evidence is immutable")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class ImmutableScalarList(list[Any]):
    """JSON-serializable list that cannot be changed after construction."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("downstream scalar evidence is immutable")

    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    __setitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def freeze_scalar_evidence(value: Any, context: str) -> Any:
    """Validate and recursively freeze one JSON-safe scalar payload."""

    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} keys must be strings")
            frozen[key] = freeze_scalar_evidence(item, f"{context}.{key}")
        return ImmutableScalarMapping(frozen)
    if isinstance(value, tuple):
        return tuple(
            freeze_scalar_evidence(item, f"{context}[{index}]") for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return ImmutableScalarList(
            freeze_scalar_evidence(item, f"{context}[{index}]") for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ValueError(f"{context} must contain only finite scalars")
        return normalized
    raise ValueError(f"{context} must contain scalar-only report values")


@dataclass(frozen=True, slots=True)
class _FrozenDiagnosticMapping:
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _FrozenDiagnosticList:
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _FrozenDiagnosticTuple:
    items: tuple[Any, ...]


def _freeze_diagnostic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDiagnosticMapping(
            tuple((key, _freeze_diagnostic_value(item)) for key, item in value.items())
        )
    if isinstance(value, list):
        return _FrozenDiagnosticList(tuple(_freeze_diagnostic_value(item) for item in value))
    if isinstance(value, tuple):
        return _FrozenDiagnosticTuple(tuple(_freeze_diagnostic_value(item) for item in value))
    return deepcopy(value)


def _thaw_diagnostic_value(value: Any) -> Any:
    if isinstance(value, _FrozenDiagnosticMapping):
        return {key: _thaw_diagnostic_value(item) for key, item in value.items}
    if isinstance(value, _FrozenDiagnosticList):
        return [_thaw_diagnostic_value(item) for item in value.items]
    if isinstance(value, _FrozenDiagnosticTuple):
        return tuple(_thaw_diagnostic_value(item) for item in value.items)
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class AttributeStageKey:
    """Semantic identity of one case-local orientation attribute source."""

    case_id: str
    shape: tuple[int, int, int]
    source: str
    settings: tuple[tuple[str, ScalarStageSetting], ...] = ()


@dataclass(frozen=True, slots=True)
class SeedStageKey:
    """All effective settings that can change the selected seed sequence."""

    attributes: AttributeStageKey
    seed_policy: str
    seed_distance: int
    seed_threshold: float
    ru: int
    rv: int
    rw: int
    boundary_target_source: str | None
    boundary_edge_margin: int | None


@dataclass(frozen=True, slots=True)
class VotingStageKey:
    """All effective settings that can change completed ``fv/vp/vt`` arrays."""

    seed: SeedStageKey
    ru: int
    rv: int
    rw: int
    bstrain1: int
    bstrain2: int
    attribute_smoothing: int
    surface_smoothing1: float
    surface_smoothing2: float
    boundary_policy: str
    support_min_fraction: float
    support_exponent: float
    orientation_smoothing: float
    orientation_backend: str
    final_normalization_smoothing: float


@dataclass(frozen=True, slots=True)
class ThinningStageKey:
    """All effective inputs to base thinning, before variant post-processing."""

    voting: VotingStageKey
    thin_mode: str
    reference_sigma: float
    hybrid_orientation_gradient_threshold: float
    hybrid_v2_edge_margin: int
    orientation_source: str
    tie_break_policy: str
    plateau_tolerance: float


@dataclass(frozen=True, slots=True)
class FinalThinningStageKey:
    """Semantic identity of final ``fvt`` after variant post-processing."""

    thinning: ThinningStageKey
    post_thinning_policy: str
    post_thinning_target_source: str | None
    post_thinning_max_shift: int | None
    post_thinning_edge_margin: int | None


@dataclass(frozen=True, slots=True)
class VotingScalarEvidenceKey:
    """Case, truth configuration, and voting identity for scalar evidence."""

    case_id: str
    case_token: int
    shape: tuple[int, int, int]
    voting: VotingStageKey
    truth_metric_config: SyntheticTruthMetricConfig
    contract_version: int


@dataclass(frozen=True, slots=True)
class ThinningScalarEvidenceKey:
    """Case, truth configuration, and final-thinning scalar identity."""

    case_id: str
    case_token: int
    shape: tuple[int, int, int]
    thinning: FinalThinningStageKey
    truth_metric_config: SyntheticTruthMetricConfig
    contract_version: int


@dataclass(frozen=True, slots=True)
class DownstreamScalarEvidence:
    """Immutable scalar-only report fragments for one array stage."""

    array_summary: Mapping[str, Any]
    top_truth_count: Mapping[str, Any]
    positive_top_truth_count: Mapping[str, Any]
    edge_top_truth_count: Mapping[str, Any]
    edge_positive_top_truth_count: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "array_summary",
            "top_truth_count",
            "positive_top_truth_count",
            "edge_top_truth_count",
            "edge_positive_top_truth_count",
        ):
            object.__setattr__(
                self,
                name,
                freeze_scalar_evidence(getattr(self, name), name),
            )


@dataclass(frozen=True, slots=True)
class PrimarySkinningStageKey:
    """All effective array stages and settings used by primary skin growth."""

    thinning: ThinningStageKey
    skinner_identity: str
    post_thinning_policy: str
    post_thinning_target_source: str | None
    post_thinning_max_shift: int | None
    post_thinning_edge_margin: int | None
    method: str
    growth_source: str
    min_likelihood: float | None
    min_skin_size: int | None
    d: int
    ru: int
    rv: int | None
    rw: int | None
    max_steps: int
    du: float
    max_delta_strike: float
    reskin: bool
    accepted_occupancy_radius: int | None
    small_skin_size: int


@dataclass(frozen=True, slots=True)
class SeedStageResult:
    seeds: tuple[FaultCell, ...]
    diagnostic_items: DiagnosticItems | None = None

    def diagnostics(self) -> dict[str, Any] | None:
        return None if self.diagnostic_items is None else dict(self.diagnostic_items)


@dataclass(frozen=True, slots=True)
class VotingStageResult:
    fv: np.ndarray
    vp: np.ndarray
    vt: np.ndarray
    diagnostic_items: DiagnosticItems

    def __post_init__(self) -> None:
        for array in (self.fv, self.vp, self.vt):
            array.flags.writeable = False

    def diagnostics(self) -> dict[str, Any]:
        return dict(self.diagnostic_items)


@dataclass(frozen=True, slots=True)
class ThinningStageResult:
    fvt: np.ndarray

    def __post_init__(self) -> None:
        self.fvt.flags.writeable = False


@dataclass(frozen=True, slots=True)
class FinalThinningStageResult:
    """Final ``fvt`` and diagnostics after variant post-processing."""

    fvt: np.ndarray
    diagnostic_items: DiagnosticItems

    def __post_init__(self) -> None:
        self.fvt.flags.writeable = False

    def diagnostics(self) -> dict[str, Any]:
        return dict(self.diagnostic_items)


@dataclass(frozen=True, slots=True)
class _FaultCellSnapshot:
    x1: float
    x2: float
    x3: float
    fl: float
    fp: float
    ft: float
    ca: int | None
    cb: int | None
    cl: int | None
    cr: int | None


@dataclass(frozen=True, slots=True)
class PrimarySkinningStageResult:
    """Immutable cell/link snapshot that yields independent variant-local skins."""

    cells: tuple[_FaultCellSnapshot, ...]
    skins: tuple[tuple[int, ...], ...]
    diagnostic_items: DiagnosticItems

    @classmethod
    def from_skins(
        cls,
        skins: tuple[FaultSkin, ...] | list[FaultSkin],
        diagnostics: dict[str, Any],
    ) -> PrimarySkinningStageResult:
        ordered_cells: list[FaultCell] = []
        cell_indices: dict[int, int] = {}

        def add_cell(cell: FaultCell) -> int:
            identity = id(cell)
            index = cell_indices.get(identity)
            if index is not None:
                return index
            index = len(ordered_cells)
            cell_indices[identity] = index
            ordered_cells.append(cell)
            return index

        skin_indices = tuple(tuple(add_cell(cell) for cell in skin) for skin in skins)
        # Preserve link targets even if an implementation keeps a linked cell
        # outside the returned skin membership.
        cursor = 0
        while cursor < len(ordered_cells):
            cell = ordered_cells[cursor]
            for linked in (cell.ca, cell.cb, cell.cl, cell.cr):
                if linked is not None:
                    add_cell(linked)
            cursor += 1

        def link_index(cell: FaultCell | None) -> int | None:
            return None if cell is None else cell_indices[id(cell)]

        snapshots = tuple(
            _FaultCellSnapshot(
                x1=float(cell.x1),
                x2=float(cell.x2),
                x3=float(cell.x3),
                fl=float(cell.fl),
                fp=float(cell.fp),
                ft=float(cell.ft),
                ca=link_index(cell.ca),
                cb=link_index(cell.cb),
                cl=link_index(cell.cl),
                cr=link_index(cell.cr),
            )
            for cell in ordered_cells
        )
        return cls(
            cells=snapshots,
            skins=skin_indices,
            diagnostic_items=tuple(
                (key, _freeze_diagnostic_value(value)) for key, value in diagnostics.items()
            ),
        )

    def clone(self) -> tuple[list[FaultSkin], dict[str, Any]]:
        cells = [
            FaultCell(item.x1, item.x2, item.x3, item.fl, item.fp, item.ft) for item in self.cells
        ]
        for cell, item in zip(cells, self.cells):
            for name in ("ca", "cb", "cl", "cr"):
                linked_index = getattr(item, name)
                object.__setattr__(
                    cell,
                    name,
                    None if linked_index is None else cells[linked_index],
                )
        return (
            [FaultSkin.from_cells(cells[index] for index in indices) for indices in self.skins],
            {key: _thaw_diagnostic_value(value) for key, value in self.diagnostic_items},
        )


_PipelineStageResultT = TypeVar("_PipelineStageResultT")


class PipelineStageBuildTimer(Protocol):
    """Synchronous wrapper around one cache-miss stage build."""

    def __call__(
        self,
        stage: str,
        semantic_key: Any,
        operation: Callable[[], _PipelineStageResultT],
    ) -> _PipelineStageResultT: ...


@dataclass(frozen=True, slots=True)
class PipelineStageCacheStats:
    seed_hits: int
    seed_misses: int
    voting_hits: int
    voting_misses: int
    thinning_hits: int
    thinning_misses: int
    primary_skinning_hits: int
    primary_skinning_misses: int


@dataclass(slots=True)
class PipelineStageCache:
    """Cache whose owner limits its lifetime to one synthetic case."""

    case: InitVar[Synthetic3DCase | None] = None
    build_timer: PipelineStageBuildTimer | None = field(default=None, repr=False)
    _case: Synthetic3DCase | None = field(default=None, init=False, repr=False)
    _seeds: dict[SeedStageKey, SeedStageResult] = field(default_factory=dict, init=False)
    _voting: dict[VotingStageKey, VotingStageResult] = field(default_factory=dict, init=False)
    _thinning: dict[ThinningStageKey, ThinningStageResult] = field(default_factory=dict, init=False)
    _final_thinning: dict[FinalThinningStageKey, FinalThinningStageResult] = field(
        default_factory=dict, init=False
    )
    _primary_skinning: dict[PrimarySkinningStageKey, PrimarySkinningStageResult] = field(
        default_factory=dict, init=False
    )
    seed_hits: int = field(default=0, init=False)
    seed_misses: int = field(default=0, init=False)
    voting_hits: int = field(default=0, init=False)
    voting_misses: int = field(default=0, init=False)
    thinning_hits: int = field(default=0, init=False)
    thinning_misses: int = field(default=0, init=False)
    primary_skinning_hits: int = field(default=0, init=False)
    primary_skinning_misses: int = field(default=0, init=False)

    def __post_init__(self, case: Synthetic3DCase | None) -> None:
        self._case = case

    def bind_case(self, case: Synthetic3DCase) -> None:
        """Bind an unowned cache or reject reuse for a different case instance."""

        if self._case is None:
            self._case = case
        elif self._case is not case:
            raise ValueError("pipeline stage cache must not be reused across synthetic cases")

    @property
    def stats(self) -> PipelineStageCacheStats:
        """Return an immutable snapshot for tests and internal diagnostics."""

        return PipelineStageCacheStats(
            seed_hits=self.seed_hits,
            seed_misses=self.seed_misses,
            voting_hits=self.voting_hits,
            voting_misses=self.voting_misses,
            thinning_hits=self.thinning_hits,
            thinning_misses=self.thinning_misses,
            primary_skinning_hits=self.primary_skinning_hits,
            primary_skinning_misses=self.primary_skinning_misses,
        )

    def get_seed(self, key: SeedStageKey) -> SeedStageResult | None:
        result = self._seeds.get(key)
        if result is None:
            self.seed_misses += 1
        else:
            self.seed_hits += 1
        return result

    def put_seed(self, key: SeedStageKey, result: SeedStageResult) -> None:
        self._seeds[key] = result

    def get_or_build_seed(
        self,
        key: SeedStageKey,
        builder: Callable[[], SeedStageResult],
    ) -> SeedStageResult:
        """Return a cached seed result or time and store one completed build."""

        result = self.get_seed(key)
        if result is not None:
            return result
        result = (
            builder()
            if self.build_timer is None
            else self.build_timer("seed_selection", key, builder)
        )
        self.put_seed(key, result)
        return result

    def get_voting(self, key: VotingStageKey) -> VotingStageResult | None:
        result = self._voting.get(key)
        if result is None:
            self.voting_misses += 1
        else:
            self.voting_hits += 1
        return result

    def put_voting(self, key: VotingStageKey, result: VotingStageResult) -> None:
        self._voting[key] = result

    def get_or_build_voting(
        self,
        key: VotingStageKey,
        builder: Callable[[], VotingStageResult],
    ) -> VotingStageResult:
        """Return a cached voting result or time and store one completed build."""

        result = self.get_voting(key)
        if result is not None:
            return result
        result = (
            builder()
            if self.build_timer is None
            else self.build_timer("voting_volume", key, builder)
        )
        self.put_voting(key, result)
        return result

    def get_thinning(self, key: ThinningStageKey) -> ThinningStageResult | None:
        result = self._thinning.get(key)
        if result is None:
            self.thinning_misses += 1
        else:
            self.thinning_hits += 1
        return result

    def put_thinning(self, key: ThinningStageKey, result: ThinningStageResult) -> None:
        self._thinning[key] = result

    def get_final_thinning(self, key: FinalThinningStageKey) -> FinalThinningStageResult | None:
        return self._final_thinning.get(key)

    def put_final_thinning(
        self,
        key: FinalThinningStageKey,
        result: FinalThinningStageResult,
    ) -> None:
        self._final_thinning[key] = result

    def get_or_build_thinning(
        self,
        key: ThinningStageKey,
        builder: Callable[[], ThinningStageResult],
    ) -> ThinningStageResult:
        """Return a cached thinning result or time and store one completed build."""

        result = self.get_thinning(key)
        if result is not None:
            return result
        result = (
            builder()
            if self.build_timer is None
            else self.build_timer("base_thinning", key, builder)
        )
        self.put_thinning(key, result)
        return result

    def get_primary_skinning(
        self, key: PrimarySkinningStageKey
    ) -> PrimarySkinningStageResult | None:
        result = self._primary_skinning.get(key)
        if result is None:
            self.primary_skinning_misses += 1
        else:
            self.primary_skinning_hits += 1
        return result

    def put_primary_skinning(
        self, key: PrimarySkinningStageKey, result: PrimarySkinningStageResult
    ) -> None:
        self._primary_skinning[key] = result

    def get_or_build_primary_skinning(
        self,
        key: PrimarySkinningStageKey,
        builder: Callable[[], PrimarySkinningStageResult],
    ) -> PrimarySkinningStageResult:
        """Return cached primary skins or time and store one completed build."""

        result = self.get_primary_skinning(key)
        if result is not None:
            return result
        result = (
            builder()
            if self.build_timer is None
            else self.build_timer("primary_skinning", key, builder)
        )
        self.put_primary_skinning(key, result)
        return result

    def clear(self) -> None:
        """Release all case-local stage values and end this cache scope."""

        self._seeds.clear()
        self._voting.clear()
        self._thinning.clear()
        self._final_thinning.clear()
        self._primary_skinning.clear()
        self._case = None


@dataclass(frozen=True, slots=True)
class DownstreamScalarEvidenceCacheStats:
    voting_builds: int
    voting_reuses: int
    thinning_builds: int
    thinning_reuses: int


@dataclass(slots=True)
class DownstreamScalarEvidenceCache:
    """Process-local scalar cache owned by one synthetic trial."""

    case: InitVar[Synthetic3DCase | None] = None
    contract_version: int = SCALAR_EVIDENCE_CONTRACT_VERSION
    build_timer: (
        Callable[
            [str, Any, Callable[[], DownstreamScalarEvidence]],
            DownstreamScalarEvidence,
        ]
        | None
    ) = field(default=None, repr=False)
    _case: Synthetic3DCase | None = field(default=None, init=False, repr=False)
    _voting: dict[VotingScalarEvidenceKey, DownstreamScalarEvidence] = field(
        default_factory=dict, init=False
    )
    _thinning: dict[ThinningScalarEvidenceKey, DownstreamScalarEvidence] = field(
        default_factory=dict, init=False
    )
    voting_builds: int = field(default=0, init=False)
    voting_reuses: int = field(default=0, init=False)
    thinning_builds: int = field(default=0, init=False)
    thinning_reuses: int = field(default=0, init=False)

    def __post_init__(self, case: Synthetic3DCase | None) -> None:
        if isinstance(self.contract_version, bool) or not isinstance(self.contract_version, int):
            raise ValueError("scalar evidence contract_version must be an integer")
        if self.contract_version < 1:
            raise ValueError("scalar evidence contract_version must be positive")
        self._case = case

    def bind_case(self, case: Synthetic3DCase) -> None:
        if self._case is None:
            self._case = case
        elif self._case is not case:
            raise ValueError("downstream scalar evidence cache must not be reused across cases")

    @property
    def stats(self) -> DownstreamScalarEvidenceCacheStats:
        return DownstreamScalarEvidenceCacheStats(
            voting_builds=self.voting_builds,
            voting_reuses=self.voting_reuses,
            thinning_builds=self.thinning_builds,
            thinning_reuses=self.thinning_reuses,
        )

    def _validate_key(
        self,
        key: VotingScalarEvidenceKey | ThinningScalarEvidenceKey,
    ) -> None:
        if self._case is None:
            raise ValueError("downstream scalar evidence cache is not bound to a case")
        if key.contract_version != self.contract_version:
            raise ValueError("scalar evidence key uses a different contract version")
        if (
            key.case_token != id(self._case)
            or key.case_id != self._case.case_id
            or key.shape != self._case.shape
        ):
            raise ValueError("scalar evidence key does not match the bound case")

    def get_voting(self, key: VotingScalarEvidenceKey) -> DownstreamScalarEvidence | None:
        if not isinstance(key, VotingScalarEvidenceKey):
            raise ValueError("voting scalar evidence key has the wrong type")
        self._validate_key(key)
        result = self._voting.get(key)
        if result is not None:
            self.voting_reuses += 1
        return result

    def put_voting(self, key: VotingScalarEvidenceKey, result: DownstreamScalarEvidence) -> None:
        if not isinstance(key, VotingScalarEvidenceKey):
            raise ValueError("voting scalar evidence key has the wrong type")
        self._validate_key(key)
        if not isinstance(result, DownstreamScalarEvidence):
            raise ValueError("voting scalar evidence has the wrong type")
        self._voting[key] = result
        self.voting_builds += 1

    def get_or_build_voting(
        self,
        key: VotingScalarEvidenceKey,
        builder: Callable[[], DownstreamScalarEvidence],
    ) -> DownstreamScalarEvidence:
        """Return cached voting evidence or time and store one unique build."""

        result = self.get_voting(key)
        if result is not None:
            return result
        result = (
            builder()
            if self.build_timer is None
            else self.build_timer("voting_scalar_evidence", key, builder)
        )
        self.put_voting(key, result)
        return result

    def get_thinning(self, key: ThinningScalarEvidenceKey) -> DownstreamScalarEvidence | None:
        if not isinstance(key, ThinningScalarEvidenceKey):
            raise ValueError("thinning scalar evidence key has the wrong type")
        self._validate_key(key)
        result = self._thinning.get(key)
        if result is not None:
            self.thinning_reuses += 1
        return result

    def put_thinning(
        self, key: ThinningScalarEvidenceKey, result: DownstreamScalarEvidence
    ) -> None:
        if not isinstance(key, ThinningScalarEvidenceKey):
            raise ValueError("thinning scalar evidence key has the wrong type")
        self._validate_key(key)
        if not isinstance(result, DownstreamScalarEvidence):
            raise ValueError("thinning scalar evidence has the wrong type")
        self._thinning[key] = result
        self.thinning_builds += 1

    def get_or_build_thinning(
        self,
        key: ThinningScalarEvidenceKey,
        builder: Callable[[], DownstreamScalarEvidence],
    ) -> DownstreamScalarEvidence:
        """Return cached thinning evidence or time and store one unique build."""

        result = self.get_thinning(key)
        if result is not None:
            return result
        result = (
            builder()
            if self.build_timer is None
            else self.build_timer("thinning_scalar_evidence", key, builder)
        )
        self.put_thinning(key, result)
        return result

    def clear(self) -> None:
        self._voting.clear()
        self._thinning.clear()
        self._case = None


def diagnostic_items(diagnostics: dict[str, Any]) -> DiagnosticItems:
    """Freeze the scalar, JSON-safe stage summaries used by this cache."""

    return tuple(diagnostics.items())

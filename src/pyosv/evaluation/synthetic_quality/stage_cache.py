"""Case-local cache keys and values for synthetic-quality pipeline stages."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pyosv.cells import FaultCell

if TYPE_CHECKING:
    from pyosv.synthetic3d import Synthetic3DCase


ScalarStageSetting = str | int | float | bool
DiagnosticItems = tuple[tuple[str, ScalarStageSetting], ...]


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
class PipelineStageCacheStats:
    seed_hits: int
    seed_misses: int
    voting_hits: int
    voting_misses: int


@dataclass(slots=True)
class PipelineStageCache:
    """Cache whose owner limits its lifetime to one synthetic case."""

    case: InitVar[Synthetic3DCase | None] = None
    _case: Synthetic3DCase | None = field(default=None, init=False, repr=False)
    _seeds: dict[SeedStageKey, SeedStageResult] = field(default_factory=dict, init=False)
    _voting: dict[VotingStageKey, VotingStageResult] = field(default_factory=dict, init=False)
    seed_hits: int = field(default=0, init=False)
    seed_misses: int = field(default=0, init=False)
    voting_hits: int = field(default=0, init=False)
    voting_misses: int = field(default=0, init=False)

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

    def get_voting(self, key: VotingStageKey) -> VotingStageResult | None:
        result = self._voting.get(key)
        if result is None:
            self.voting_misses += 1
        else:
            self.voting_hits += 1
        return result

    def put_voting(self, key: VotingStageKey, result: VotingStageResult) -> None:
        self._voting[key] = result


def diagnostic_items(diagnostics: dict[str, Any]) -> DiagnosticItems:
    """Freeze the scalar, JSON-safe stage summaries used by this cache."""

    return tuple(diagnostics.items())

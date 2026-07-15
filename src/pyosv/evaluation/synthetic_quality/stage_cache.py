"""Case-local cache keys and values for synthetic-quality pipeline stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pyosv.cells import FaultCell
from pyosv.skin import FaultSkin

if TYPE_CHECKING:
    from pyosv.synthetic3d import Synthetic3DCase


ScalarStageSetting = str | int | float | bool
DiagnosticItems = tuple[tuple[str, Any], ...]


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
class PrimarySkinningStageKey:
    """All effective array stages and settings used by primary skin growth."""

    thinning: ThinningStageKey
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
    _case: Synthetic3DCase | None = field(default=None, init=False, repr=False)
    _seeds: dict[SeedStageKey, SeedStageResult] = field(default_factory=dict, init=False)
    _voting: dict[VotingStageKey, VotingStageResult] = field(default_factory=dict, init=False)
    _thinning: dict[ThinningStageKey, ThinningStageResult] = field(default_factory=dict, init=False)
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

    def get_voting(self, key: VotingStageKey) -> VotingStageResult | None:
        result = self._voting.get(key)
        if result is None:
            self.voting_misses += 1
        else:
            self.voting_hits += 1
        return result

    def put_voting(self, key: VotingStageKey, result: VotingStageResult) -> None:
        self._voting[key] = result

    def get_thinning(self, key: ThinningStageKey) -> ThinningStageResult | None:
        result = self._thinning.get(key)
        if result is None:
            self.thinning_misses += 1
        else:
            self.thinning_hits += 1
        return result

    def put_thinning(self, key: ThinningStageKey, result: ThinningStageResult) -> None:
        self._thinning[key] = result

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

    def clear(self) -> None:
        """Release all case-local stage values and end this cache scope."""

        self._seeds.clear()
        self._voting.clear()
        self._thinning.clear()
        self._primary_skinning.clear()
        self._case = None


def diagnostic_items(diagnostics: dict[str, Any]) -> DiagnosticItems:
    """Freeze the scalar, JSON-safe stage summaries used by this cache."""

    return tuple(diagnostics.items())

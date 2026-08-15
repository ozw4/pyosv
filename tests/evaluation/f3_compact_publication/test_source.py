from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyosv.evaluation.f3_compact_publication.config import (
    AMPLITUDE_DTYPE,
    AMPLITUDE_FILENAME,
    AMPLITUDE_ROLE,
    DISPLAY_CELL,
    SECTION_GROUPS,
    SECTION_SELECTION_POLICY,
    SECTIONS_PER_AXIS,
    STAGE_ORDER,
)
from pyosv.evaluation.f3_compact_publication import source as source_module
from pyosv.evaluation.f3_compact_publication.source import load_compact_source
from pyosv.evaluation.f3d_mode_comparison import (
    F3_BUFFERED_PERCENTILE,
    F3_BUFFER_RADIUS,
    F3_METRIC_SCHEMA_VERSION,
    F3_REFERENCE_STAGE_FILES,
    F3CellReference,
    F3CellStageFingerprints,
    F3DatasetSpec,
    MetricEvidence,
)
from pyosv.evaluation.f3d_mode_comparison.metrics import F3_REFERENCE_STAGE_ROLES
from pyosv.evaluation.mode_comparison_publication.models import F3SourceBundle

_SHAPE = (10, 4, 10)
_DTYPE = np.dtype(">f4")
_REFERENCE_THRESHOLDS = {"ft": 0.8, "fv": 0.7, "fvt": 0.6}
_CELLS = ("RL-REF", "RL-QUAL", "Q-REF", "Q-QUAL")
_FINGERPRINTS = {
    "scanner": "1" * 64,
    "voting": "2" * 64,
    "thinning": "3" * 64,
    "skinning": "4" * 64,
}


@dataclass(frozen=True)
class _Fixture:
    source: F3SourceBundle
    bundle: Path
    data_root: Path
    q_qual: F3CellReference


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_volume(path: Path, values: np.ndarray | None = None) -> None:
    array = np.zeros(_SHAPE, dtype=np.float32) if values is None else values
    np.asarray(array, dtype=_DTYPE).tofile(path)


def _metric_evidence() -> tuple[MetricEvidence, ...]:
    return tuple(
        MetricEvidence(
            schema_version=F3_METRIC_SCHEMA_VERSION,
            dataset_id="compact-fixture",
            cell_label=cell,
            stage=stage,
            region="full",
            selection="positive_p99_radius2",
            reference_file=F3_REFERENCE_STAGE_FILES[stage],
            source_stage_fingerprint=_FINGERPRINTS[
                {"ft": "scanner", "fv": "voting", "fvt": "thinning"}[stage]
            ],
            reference_sha256="a" * 64,
            shape=_SHAPE,
            thresholds=(
                ("percentile", F3_BUFFERED_PERCENTILE),
                ("radius", F3_BUFFER_RADIUS),
                ("reference_threshold", _REFERENCE_THRESHOLDS[stage]),
                ("candidate_threshold", _REFERENCE_THRESHOLDS[stage] - 0.1),
            ),
        )
        for stage in STAGE_ORDER
        for cell in _CELLS
    )


@pytest.fixture
def compact_fixture(tmp_path: Path) -> _Fixture:
    data_root = tmp_path / "f3-data"
    data_root.mkdir()
    public_fvt = np.zeros(_SHAPE, dtype=np.float32)
    for index in (1, 3, 5, 7, 9):
        public_fvt[:, 0, index] = 0.8
        public_fvt[index, 1, :] = 0.9
    for filename in ("ep.dat", "fl.dat", "fv.dat"):
        _write_volume(data_root / filename)
    _write_volume(data_root / "fvt.dat", public_fvt)
    amplitude = np.arange(np.prod(_SHAPE), dtype=np.float32).reshape(_SHAPE)
    _write_volume(data_root / AMPLITUDE_FILENAME, amplitude)

    bundle = tmp_path / "f3-bundle"
    bundle.mkdir()
    for stage, kind in zip(STAGE_ORDER, ("scanner", "voting", "thinning")):
        stage_root = bundle / "stages" / kind / _FINGERPRINTS[kind]
        stage_root.mkdir(parents=True)
        _write_volume(stage_root / f"{stage}.dat")

    stages = F3CellStageFingerprints(**_FINGERPRINTS)
    q_qual = F3CellReference(
        label=DISPLAY_CELL,
        backend="quality",
        workflow="quality",
        resolved_config={},
        stages=stages,
        skinning_enabled=True,
        path=bundle,
        reused=True,
    )
    other_cells = tuple(SimpleNamespace(label=label) for label in _CELLS[:-1])
    result = SimpleNamespace(
        volume_shape=_SHAPE,
        storage_dtype=_DTYPE.str,
        cells=(*other_cells, q_qual),
    )
    files = []
    for stage in STAGE_ORDER:
        role = F3_REFERENCE_STAGE_ROLES[stage]
        path = data_root / F3_REFERENCE_STAGE_FILES[stage]
        files.append(
            {
                "role": role,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
                "shape": list(_SHAPE),
                "storage_dtype": _DTYPE.str,
            }
        )
    spec = F3DatasetSpec(
        dataset_id="compact-fixture",
        shape=_SHAPE,
        storage_dtype=_DTYPE.str,
        files=(
            ("input", "ep.dat"),
            ("reference_fault_likelihood", "fl.dat"),
            ("reference_fault_votes", "fv.dat"),
            ("reference_thinned_fault_votes", "fvt.dat"),
        ),
        expected_bytes=int(np.prod(_SHAPE)) * _DTYPE.itemsize,
    )
    source = F3SourceBundle(
        path=bundle,
        data_root=data_root,
        dataset_spec=spec,
        run_manifest={},
        completion_sha256="b" * 64,
        result=result,
        metric_evidence=_metric_evidence(),
        dataset_identity={"dataset_id": spec.dataset_id, "files": files},
    )
    return _Fixture(source=source, bundle=bundle, data_root=data_root, q_qual=q_qual)


def _load(
    fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    source: F3SourceBundle | None = None,
):
    selected = fixture.source if source is None else source
    monkeypatch.setattr(source_module, "load_f3_source", lambda *_: selected)
    return load_compact_source(fixture.bundle, fixture.data_root)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_load_compact_source_resolves_ordered_immutable_context(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_bundle = _snapshot(compact_fixture.bundle)
    before_data = _snapshot(compact_fixture.data_root)

    context = _load(compact_fixture, monkeypatch)

    assert tuple(item.stage for item in context.stage_sources) == STAGE_ORDER
    assert context.q_qual_cell is compact_fixture.q_qual
    assert context.amplitude.role == AMPLITUDE_ROLE
    assert context.amplitude.filename == AMPLITUDE_FILENAME
    assert context.amplitude.storage_dtype == AMPLITUDE_DTYPE
    assert context.amplitude.sha256 == _sha256(compact_fixture.data_root / AMPLITUDE_FILENAME)
    assert len(context.selected_sections) == 2 * SECTIONS_PER_AXIS
    assert tuple(
        (item.section_group, item.axis, item.bin_index) for item in context.selected_sections
    ) == tuple(
        (section_group, axis, bin_index)
        for section_group, axis in SECTION_GROUPS
        for bin_index in range(SECTIONS_PER_AXIS)
    )
    assert tuple(item.index for item in context.selected_sections[:5]) == (1, 3, 5, 7, 9)
    assert tuple(item.index for item in context.selected_sections[5:]) == (1, 3, 5, 7, 9)
    assert all(item.policy == SECTION_SELECTION_POLICY for item in context.selected_sections)
    assert tuple(item.stage for item in context.ridge_threshold_contract.stages) == STAGE_ORDER
    assert _snapshot(compact_fixture.bundle) == before_bundle
    assert _snapshot(compact_fixture.data_root) == before_data


@pytest.mark.parametrize("cells", [(), ("duplicate",)])
def test_q_qual_must_appear_exactly_once(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    cells: tuple[str, ...],
) -> None:
    if cells:
        replacement_cells = (compact_fixture.q_qual, compact_fixture.q_qual)
    else:
        replacement_cells = (SimpleNamespace(label="RL-REF"),)
    result = SimpleNamespace(
        volume_shape=_SHAPE,
        storage_dtype=_DTYPE.str,
        cells=replacement_cells,
    )
    source = replace(compact_fixture.source, result=result)

    with pytest.raises(ValueError, match="exactly one 'Q-QUAL'"):
        _load(compact_fixture, monkeypatch, source)


def test_xs_missing_is_rejected(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (compact_fixture.data_root / AMPLITUDE_FILENAME).unlink()
    with pytest.raises(ValueError, match="amplitude input is missing"):
        _load(compact_fixture, monkeypatch)


def test_xs_symlink_is_rejected(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    amplitude = compact_fixture.data_root / AMPLITUDE_FILENAME
    payload = amplitude.read_bytes()
    amplitude.unlink()
    backing = tmp_path / "amplitude.dat"
    backing.write_bytes(payload)
    amplitude.symlink_to(backing)

    with pytest.raises(ValueError, match="regular non-symlink"):
        _load(compact_fixture, monkeypatch)


def test_xs_wrong_size_is_rejected(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (compact_fixture.data_root / AMPLITUDE_FILENAME).write_bytes(b"short")
    with pytest.raises(ValueError, match="amplitude input size mismatch"):
        _load(compact_fixture, monkeypatch)


@pytest.mark.parametrize("failure", ["missing", "wrong-size"])
def test_stage_artifact_missing_or_wrong_size_is_rejected(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    path = compact_fixture.bundle / "stages" / "voting" / _FINGERPRINTS["voting"] / "fv.dat"
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes(b"short")
    with pytest.raises(ValueError, match="Q-QUAL fv artifact"):
        _load(compact_fixture, monkeypatch)


def test_public_fvt_peak_tie_and_zero_bins_use_smallest_index(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_fvt = np.zeros(_SHAPE, dtype=np.float32)
    public_fvt[:, 0, 0] = 0.8
    public_fvt[:, 0, 1] = 0.8
    public_fvt[0, 1, :] = 0.9
    public_fvt[1, 1, :] = 0.9
    _write_volume(compact_fixture.data_root / "fvt.dat", public_fvt)

    context = _load(compact_fixture, monkeypatch)

    assert tuple(item.index for item in context.selected_sections[:5]) == (0, 2, 4, 6, 8)
    assert tuple(item.index for item in context.selected_sections[5:]) == (0, 2, 4, 6, 8)
    assert context.selected_sections[0].ridge_count_score > 0


def test_all_zero_bins_use_smallest_index(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_volume(compact_fixture.data_root / "fvt.dat")

    context = _load(compact_fixture, monkeypatch)

    assert tuple(item.index for item in context.selected_sections[:5]) == (0, 2, 4, 6, 8)
    assert tuple(item.index for item in context.selected_sections[5:]) == (0, 2, 4, 6, 8)
    assert all(item.ridge_count_score == 0 for item in context.selected_sections)


def test_candidate_volume_does_not_affect_selected_sections(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _load(compact_fixture, monkeypatch).selected_sections
    candidate = np.full(_SHAPE, 100.0, dtype=np.float32)
    path = compact_fixture.bundle / "stages" / "thinning" / _FINGERPRINTS["thinning"] / "fvt.dat"
    _write_volume(path, candidate)

    after = _load(compact_fixture, monkeypatch).selected_sections

    assert after == before


@pytest.mark.parametrize("shape", [(4, 5, 5), (5, 5, 4)])
def test_section_axis_shorter_than_five_is_rejected(
    compact_fixture: _Fixture,
    shape: tuple[int, int, int],
) -> None:
    axis = "i3" if shape[0] < 5 else "i1"

    with pytest.raises(ValueError, match="length must be at least 5"):
        source_module._select_public_fvt_sections(
            compact_fixture.data_root / "fvt.dat",
            shape,
            _DTYPE.str,
            0.6,
            section_group="test",
            axis=axis,
            count=5,
        )


def test_selection_memmap_is_closed(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_memmap = np.memmap
    opened: list[np.memmap] = []

    class TrackedMemmap(original_memmap):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: object, **kwargs: object) -> np.memmap:
            volume = super().__new__(cls, *args, **kwargs)
            opened.append(volume)
            return volume

    monkeypatch.setattr(source_module.np, "memmap", TrackedMemmap)

    _load(compact_fixture, monkeypatch)

    assert len(opened) == 2
    assert all(volume._mmap.closed for volume in opened)


def test_stage_sources_keep_q_qual_fingerprints(
    compact_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _load(compact_fixture, monkeypatch)

    assert {
        item.stage: (item.candidate_source_kind, item.candidate_fingerprint)
        for item in context.stage_sources
    } == {
        "ft": ("scanner", _FINGERPRINTS["scanner"]),
        "fv": ("voting", _FINGERPRINTS["voting"]),
        "fvt": ("thinning", _FINGERPRINTS["thinning"]),
    }

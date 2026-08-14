from __future__ import annotations

import csv
import hashlib
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyosv.evaluation.f3_compact_publication import (
    AMPLITUDE_DTYPE,
    AMPLITUDE_FILENAME,
    AMPLITUDE_ROLE,
    DISPLAY_CELL,
    FIGURE_DATA_HEADER,
    PUBLIC_REFERENCE_LABEL,
    SLICE_AXIS,
    SLICE_POLICY,
    STAGE_ORDER,
    AmplitudeIdentity,
    CompactSourceContext,
    RidgeStageThresholds,
    SelectedSlice,
    SourceRidgeThresholdContract,
    StageSource,
    generate_figures,
)
from pyosv.evaluation.f3_compact_publication import figures as figures_module
from pyosv.evaluation.f3d_mode_comparison import (
    F3_METRIC_SCHEMA_VERSION,
    F3_REFERENCE_STAGE_FILES,
    F3DatasetSpec,
    METRIC_REGISTRY,
    MetricRow,
)
from pyosv.evaluation.f3d_mode_comparison.metrics import F3_REFERENCE_STAGE_ROLES
from pyosv.evaluation.mode_comparison_publication.models import F3SourceBundle

_SHAPE = (3, 4, 5)
_DTYPE = np.dtype(AMPLITUDE_DTYPE)
_SIZE = int(np.prod(_SHAPE)) * _DTYPE.itemsize
_INDEX = 2
_DEFINITIONS = {(item.stage, item.selection, item.metric): item for item in METRIC_REGISTRY}


@dataclass(frozen=True)
class _FigureFixture:
    context: CompactSourceContext
    data_root: Path
    bundle: Path


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_volume(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=_DTYPE).tofile(path)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _scale_row(stage: str, metric: str, value: float) -> MetricRow:
    definition = _DEFINITIONS[(stage, "all", metric)]
    return MetricRow(
        schema_version=F3_METRIC_SCHEMA_VERSION,
        dataset_id="compact-figure-fixture",
        cell_label=DISPLAY_CELL,
        scanner_backend="quality",
        workflow_mode="quality",
        stage=stage,
        region="full",
        selection="all",
        reference_file=F3_REFERENCE_STAGE_FILES[stage],
        metric=metric,
        value=value,
        unit=definition.unit,
        direction=definition.direction,
        contrast_eligible=definition.contrast_eligible,
    )


@pytest.fixture
def figure_fixture(tmp_path: Path) -> _FigureFixture:
    data_root = tmp_path / "source" / "data"
    bundle = tmp_path / "source" / "bundle"
    data_root.mkdir(parents=True)
    bundle.mkdir(parents=True)

    amplitude = np.linspace(-2.0, 2.0, num=np.prod(_SHAPE), dtype=np.float32).reshape(_SHAPE)
    _write_volume(data_root / AMPLITUDE_FILENAME, amplitude)

    stage_sources = []
    threshold_rows = []
    metric_rows = []
    identities = []
    for stage_index, (stage, kind) in enumerate(
        zip(STAGE_ORDER, ("scanner", "voting", "thinning"), strict=True)
    ):
        reference = np.zeros(_SHAPE, dtype=np.float32)
        candidate = np.zeros(_SHAPE, dtype=np.float32)
        reference[0, _INDEX, 1] = 0.7 + stage_index * 0.1
        reference[2, _INDEX, 3] = 1.0 + stage_index * 0.1
        candidate[0, _INDEX, 1] = 0.9 + stage_index * 0.1
        candidate[1, _INDEX, 2] = 1.2 + stage_index * 0.1
        public_filename = F3_REFERENCE_STAGE_FILES[stage]
        public_path = data_root / public_filename
        _write_volume(public_path, reference)
        fingerprint = _digest(kind)
        candidate_path = bundle / "stages" / kind / fingerprint / f"{stage}.dat"
        _write_volume(candidate_path, candidate)
        public_sha256 = _sha256(public_path)
        stage_sources.append(
            StageSource(
                stage=stage,
                public_reference_role=F3_REFERENCE_STAGE_ROLES[stage],
                public_reference_filename=public_filename,
                public_reference_path=public_path,
                public_reference_sha256=public_sha256,
                candidate_source_kind=kind,
                candidate_fingerprint=fingerprint,
                candidate_filename=f"{stage}.dat",
                candidate_path=candidate_path,
            )
        )
        threshold_rows.append(
            RidgeStageThresholds(
                stage=stage,
                public_reference_threshold=0.6 + stage_index * 0.1,
                q_qual_threshold=0.8 + stage_index * 0.1,
            )
        )
        metric_rows.extend(
            (
                _scale_row(stage, "reference_max", float(np.max(reference))),
                _scale_row(stage, "candidate_max", float(np.max(candidate))),
            )
        )
        identities.append(
            {
                "role": F3_REFERENCE_STAGE_ROLES[stage],
                "size": _SIZE,
                "sha256": public_sha256,
                "shape": list(_SHAPE),
                "storage_dtype": AMPLITUDE_DTYPE,
            }
        )

    spec = F3DatasetSpec(
        dataset_id="compact-figure-fixture",
        shape=_SHAPE,
        storage_dtype=AMPLITUDE_DTYPE,
        files=(
            ("input", "ep.dat"),
            ("reference_fault_likelihood", "fl.dat"),
            ("reference_fault_votes", "fv.dat"),
            ("reference_thinned_fault_votes", "fvt.dat"),
        ),
        expected_bytes=_SIZE,
    )
    result = SimpleNamespace(
        dataset_id=spec.dataset_id,
        volume_shape=_SHAPE,
        storage_dtype=AMPLITUDE_DTYPE,
        metric_rows=tuple(metric_rows),
    )
    source = F3SourceBundle(
        path=bundle,
        data_root=data_root,
        dataset_spec=spec,
        run_manifest={},
        completion_sha256=_digest("completion"),
        result=result,
        metric_evidence=(),
        dataset_identity={"dataset_id": spec.dataset_id, "files": identities},
    )
    context = CompactSourceContext(
        f3=source,
        amplitude=AmplitudeIdentity(
            role=AMPLITUDE_ROLE,
            filename=AMPLITUDE_FILENAME,
            resolved_path=data_root / AMPLITUDE_FILENAME,
            shape=_SHAPE,
            storage_dtype=AMPLITUDE_DTYPE,
            size=_SIZE,
            sha256=_sha256(data_root / AMPLITUDE_FILENAME),
        ),
        q_qual_cell=SimpleNamespace(label=DISPLAY_CELL),
        stage_sources=tuple(stage_sources),
        ridge_threshold_contract=SourceRidgeThresholdContract(
            selection="positive_p99_radius2",
            percentile=99.0,
            buffer_radius=2.0,
            stages=tuple(threshold_rows),
        ),
        selected_slice=SelectedSlice(
            axis=SLICE_AXIS,
            index=_INDEX,
            policy=SLICE_POLICY,
            public_fvt_reference_threshold=threshold_rows[-1].public_reference_threshold,
            ridge_count_score=2,
        ),
    )
    return _FigureFixture(context=context, data_root=data_root, bundle=bundle)


def _csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == FIGURE_DATA_HEADER
        return tuple(reader)


def test_generate_figures_writes_fixed_ordered_artifacts_and_metadata(
    figure_fixture: _FigureFixture,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    before_data = _snapshot(figure_fixture.data_root)
    before_bundle = _snapshot(figure_fixture.bundle)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = generate_figures(figure_fixture.context, output)

    assert caught == []
    assert tuple(record["stage"] for record in records) == STAGE_ORDER
    expected_ids = tuple(
        f"f3_{stage}_public_ref_vs_q_qual_{SLICE_AXIS}_{_INDEX}" for stage in STAGE_ORDER
    )
    assert tuple(record["figure_id"] for record in records) == expected_ids
    assert {path.name for path in (output / "figures").glob("*.png")} == {
        f"{figure_id}.png" for figure_id in expected_ids
    }
    assert {path.name for path in (output / "figure_data").glob("*.csv")} == {
        f"{figure_id}.csv" for figure_id in expected_ids
    }

    amplitude_limits = set()
    for stage, figure_id, source, thresholds, record in zip(
        STAGE_ORDER,
        expected_ids,
        figure_fixture.context.stage_sources,
        figure_fixture.context.ridge_threshold_contract.stages,
        records,
        strict=True,
    ):
        assert record["relative_path"] == f"figures/{figure_id}.png"
        assert record["figure_data_csv"] == f"figure_data/{figure_id}.csv"
        rows = _csv_rows(output / str(record["figure_data_csv"]))
        assert tuple(row["panel_label"] for row in rows) == (
            PUBLIC_REFERENCE_LABEL,
            DISPLAY_CELL,
            "difference",
        )
        assert {row["axis"] for row in rows} == {SLICE_AXIS}
        assert {row["slice_index"] for row in rows} == {str(_INDEX)}
        assert rows[0]["source_sha256"] == source.public_reference_sha256
        assert rows[0]["source_stage_fingerprint"] == ""
        assert float(rows[0]["selection_threshold"]) == thresholds.public_reference_threshold
        assert rows[1]["source_sha256"] == ""
        assert rows[1]["source_stage_fingerprint"] == source.candidate_fingerprint
        assert float(rows[1]["selection_threshold"]) == thresholds.q_qual_threshold
        assert rows[2]["source_label"] == "Q-QUAL - PUBLIC-REF"
        assert rows[2]["source_file"] == ""
        expected_vmax = max(
            float(metric.value)
            for metric in figure_fixture.context.f3.result.metric_rows
            if metric.stage == stage and metric.metric in {"reference_max", "candidate_max"}
        )
        assert {float(rows[index]["overlay_vmin"]) for index in (0, 1)} == {0.0}
        assert {float(rows[index]["overlay_vmax"]) for index in (0, 1)} == {expected_vmax}
        assert {row["amplitude_file"] for row in rows} == {AMPLITUDE_FILENAME}
        assert {row["amplitude_sha256"] for row in rows} == {
            figure_fixture.context.amplitude.sha256
        }
        amplitude_limits.add(rows[0]["amplitude_limit"])

        png = output / str(record["relative_path"])
        assert png.stat().st_size > 0
        from pyosv.viz import require_matplotlib

        image = require_matplotlib().imread(png)
        assert image.ndim == 3
        assert image.size > 0

    assert len(amplitude_limits) == 1
    text_output = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((output / "figure_data").glob("*.csv"))
    ) + "\n".join(str(value) for record in records for value in record.values())
    assert all(label not in text_output for label in ("RL-REF", "RL-QUAL", "Q-REF", "Synthetic"))
    assert _snapshot(figure_fixture.data_root) == before_data
    assert _snapshot(figure_fixture.bundle) == before_bundle


def test_degenerate_amplitude_and_difference_use_finite_floor(
    figure_fixture: _FigureFixture,
    tmp_path: Path,
) -> None:
    zero = np.zeros(_SHAPE, dtype=np.float32)
    _write_volume(figure_fixture.context.amplitude.resolved_path, zero)
    for source in figure_fixture.context.stage_sources:
        _write_volume(source.public_reference_path, zero)
        _write_volume(source.candidate_path, zero)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = generate_figures(figure_fixture.context, tmp_path / "degenerate")

    assert caught == []
    for record in records:
        rows = _csv_rows(tmp_path / "degenerate" / str(record["figure_data_csv"]))
        assert {float(row["amplitude_limit"]) for row in rows} == {1.0e-6}
        difference = rows[2]
        assert float(difference["difference_limit"]) == 1.0e-6
        assert np.isfinite(float(difference["overlay_vmin"]))
        assert np.isfinite(float(difference["overlay_vmax"]))


@pytest.mark.parametrize("source_kind", ["amplitude", "public", "candidate"])
def test_nonfinite_selected_slice_is_rejected(
    figure_fixture: _FigureFixture,
    tmp_path: Path,
    source_kind: str,
) -> None:
    values = np.zeros(_SHAPE, dtype=np.float32)
    values[0, _INDEX, 0] = np.nan
    if source_kind == "amplitude":
        path = figure_fixture.context.amplitude.resolved_path
    elif source_kind == "public":
        path = figure_fixture.context.stage_sources[0].public_reference_path
    else:
        path = figure_fixture.context.stage_sources[0].candidate_path
    _write_volume(path, values)

    with pytest.raises(ValueError, match="selected slice must contain only finite"):
        generate_figures(figure_fixture.context, tmp_path / source_kind)


def test_all_read_only_memmaps_are_closed(
    figure_fixture: _FigureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_memmap = np.memmap
    opened: list[np.memmap] = []

    class TrackedMemmap(original_memmap):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: object, **kwargs: object) -> np.memmap:
            volume = super().__new__(cls, *args, **kwargs)
            opened.append(volume)
            return volume

    monkeypatch.setattr(figures_module.np, "memmap", TrackedMemmap)

    generate_figures(figure_fixture.context, tmp_path / "tracked")

    assert len(opened) == 7
    assert all(volume._mmap.closed for volume in opened)


@pytest.mark.parametrize("failure", ["duplicate", "wrong-unit"])
def test_display_scale_metric_must_be_unique_and_registry_compatible(
    figure_fixture: _FigureFixture,
    tmp_path: Path,
    failure: str,
) -> None:
    rows = figure_fixture.context.f3.result.metric_rows
    changed = (
        (rows[0], *rows) if failure == "duplicate" else (replace(rows[0], unit="wrong"), *rows[1:])
    )
    result = SimpleNamespace(
        dataset_id=figure_fixture.context.f3.result.dataset_id,
        volume_shape=_SHAPE,
        storage_dtype=AMPLITUDE_DTYPE,
        metric_rows=changed,
    )
    context = replace(
        figure_fixture.context,
        f3=replace(figure_fixture.context.f3, result=result),
    )

    message = "exactly one display-scale metric row" if failure == "duplicate" else "semantics"
    with pytest.raises(ValueError, match=message):
        generate_figures(context, tmp_path / "duplicate")

from __future__ import annotations

import csv
import hashlib
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyosv.evaluation.f3_compact_publication import figures as figures_module
from pyosv.evaluation.f3_compact_publication.config import (
    AMPLITUDE_DTYPE,
    AMPLITUDE_FILENAME,
    AMPLITUDE_ROLE,
    ATTRIBUTE_ALPHA_GAMMA,
    ATTRIBUTE_ALPHA_MAX,
    ATTRIBUTE_ALPHA_MIN,
    ATTRIBUTE_COLORMAP,
    ATTRIBUTE_DISPLAY_THRESHOLD_RATIO,
    ATTRIBUTE_HALO_ALPHA,
    ATTRIBUTE_HALO_ENABLED,
    ATTRIBUTE_HALO_RADIUS_PIXELS,
    ATTRIBUTE_HALO_STRUCTURE,
    DISPLAY_CELL,
    FIGURE_DATA_HEADER,
    IMAGE_INTERPOLATION,
    PUBLIC_REFERENCE_LABEL,
    SECTION_GROUPS,
    SECTION_SELECTION_POLICY,
    SECTIONS_PER_AXIS,
    STAGE_ORDER,
)
from pyosv.evaluation.f3_compact_publication.figures import generate_figures
from pyosv.evaluation.f3_compact_publication.models import (
    AmplitudeIdentity,
    CompactSourceContext,
    RidgeStageThresholds,
    SelectedSection,
    SourceRidgeThresholdContract,
    StageSource,
)
from pyosv.evaluation.f3d_mode_comparison import (
    F3_METRIC_SCHEMA_VERSION,
    F3_REFERENCE_STAGE_FILES,
    F3DatasetSpec,
    METRIC_REGISTRY,
    MetricRow,
)
from pyosv.evaluation.f3d_mode_comparison.metrics import F3_REFERENCE_STAGE_ROLES
from pyosv.evaluation.mode_comparison_publication.models import F3SourceBundle

_SHAPE = (5, 4, 5)
_DTYPE = np.dtype(AMPLITUDE_DTYPE)
_SIZE = int(np.prod(_SHAPE)) * _DTYPE.itemsize
_DEFINITIONS = {(item.stage, item.selection, item.metric): item for item in METRIC_REGISTRY}
_EXPECTED_ORDER = tuple(
    (stage, section_group) for stage in STAGE_ORDER for section_group, _axis in SECTION_GROUPS
)


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
        for index in range(SECTIONS_PER_AXIS):
            reference[index, 1, index] = 0.7 + stage_index * 0.1
            reference[index, 2, index] = 1.0 + stage_index * 0.1
            candidate[index, 1, index] = 0.9 + stage_index * 0.1
            candidate[index, 3, index] = 1.2 + stage_index * 0.1
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
    selected_sections = tuple(
        SelectedSection(
            section_group=section_group,
            axis=axis,
            bin_index=index,
            index=index,
            policy=SECTION_SELECTION_POLICY,
            ridge_count_score=index + 1,
        )
        for section_group, axis in SECTION_GROUPS
        for index in range(SECTIONS_PER_AXIS)
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
        selected_sections=selected_sections,
    )
    return _FigureFixture(context=context, data_root=data_root, bundle=bundle)


def _csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == FIGURE_DATA_HEADER
        return tuple(reader)


def test_generate_figures_writes_fixed_ordered_atlases_and_metadata(
    figure_fixture: _FigureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    before_data = _snapshot(figure_fixture.data_root)
    before_bundle = _snapshot(figure_fixture.bundle)
    from matplotlib.axes import Axes

    imshow_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_imshow = Axes.imshow

    def tracked_imshow(self: Axes, *args: object, **kwargs: object) -> object:
        imshow_calls.append((args, kwargs))
        return original_imshow(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", tracked_imshow)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = generate_figures(figure_fixture.context, output)

    assert caught == []
    assert tuple((record["stage"], record["section_group"]) for record in records) == (
        _EXPECTED_ORDER
    )
    expected_ids = tuple(f"f3_{stage}_{group}" for stage, group in _EXPECTED_ORDER)
    assert tuple(record["figure_id"] for record in records) == expected_ids
    assert {path.name for path in (output / "figures").glob("*.png")} == {
        f"{figure_id}.png" for figure_id in expected_ids
    }
    assert {path.name for path in (output / "figure_data").glob("*.csv")} == {
        f"{figure_id}.csv" for figure_id in expected_ids
    }

    group_indices: dict[str, set[tuple[str, ...]]] = {group: set() for group, _ in SECTION_GROUPS}
    amplitude_limits: dict[str, set[str]] = {group: set() for group, _ in SECTION_GROUPS}
    for record in records:
        stage = str(record["stage"])
        group = str(record["section_group"])
        source = next(item for item in figure_fixture.context.stage_sources if item.stage == stage)
        thresholds = next(
            item
            for item in figure_fixture.context.ridge_threshold_contract.stages
            if item.stage == stage
        )
        rows = _csv_rows(output / str(record["figure_data_csv"]))
        row_count = 3 * SECTIONS_PER_AXIS
        assert len(rows) == row_count
        assert (
            tuple(row["panel_label"] for row in rows)
            == (
                PUBLIC_REFERENCE_LABEL,
                DISPLAY_CELL,
                "difference",
            )
            * SECTIONS_PER_AXIS
        )
        axis = dict(SECTION_GROUPS)[group]
        assert {row["axis"] for row in rows} == {axis}
        assert {row["section_group"] for row in rows} == {group}
        indices = tuple(rows[index]["section_index"] for index in range(0, row_count, 3))
        assert len(set(indices)) == SECTIONS_PER_AXIS
        group_indices[group].add(indices)
        assert tuple(int(rows[index]["bin_index"]) for index in range(0, row_count, 3)) == tuple(
            range(SECTIONS_PER_AXIS)
        )
        for offset in range(0, row_count, 3):
            public, candidate, difference = rows[offset : offset + 3]
            assert public["source_sha256"] == source.public_reference_sha256
            assert public["source_stage_fingerprint"] == ""
            assert float(public["selection_threshold"]) == thresholds.public_reference_threshold
            assert float(public["display_threshold"]) == pytest.approx(
                thresholds.public_reference_threshold * ATTRIBUTE_DISPLAY_THRESHOLD_RATIO
            )
            assert candidate["source_sha256"] == ""
            assert candidate["source_stage_fingerprint"] == source.candidate_fingerprint
            assert float(candidate["selection_threshold"]) == thresholds.q_qual_threshold
            assert float(candidate["display_threshold"]) == pytest.approx(
                thresholds.q_qual_threshold * ATTRIBUTE_DISPLAY_THRESHOLD_RATIO
            )
            assert difference["source_label"] == "Q-QUAL - PUBLIC-REF"
            assert difference["source_file"] == ""
            assert difference["selection_threshold"] == ""
            assert difference["display_threshold"] == ""
            assert {row["interpolation"] for row in (public, candidate, difference)} == {
                IMAGE_INTERPOLATION
            }
            for attribute in (public, candidate):
                assert attribute["colormap"] == ATTRIBUTE_COLORMAP
                assert float(attribute["alpha_min"]) == ATTRIBUTE_ALPHA_MIN
                assert float(attribute["alpha_max"]) == ATTRIBUTE_ALPHA_MAX
                assert float(attribute["alpha_gamma"]) == ATTRIBUTE_ALPHA_GAMMA
                assert attribute["halo_enabled"] == "true"
                assert int(attribute["halo_radius_pixels"]) == ATTRIBUTE_HALO_RADIUS_PIXELS
                assert float(attribute["halo_alpha"]) == ATTRIBUTE_HALO_ALPHA
                assert attribute["halo_structure"] == ATTRIBUTE_HALO_STRUCTURE
            assert difference["halo_enabled"] == "false"
            assert difference["halo_radius_pixels"] == ""
            assert difference["halo_alpha"] == ""
            assert difference["halo_structure"] == ""
            assert difference["alpha_min"] == ""
            assert difference["alpha_gamma"] == ""
        expected_vmax = max(
            float(metric.value)
            for metric in figure_fixture.context.f3.result.metric_rows
            if metric.stage == stage and metric.metric in {"reference_max", "candidate_max"}
        )
        assert {float(row["overlay_vmax"]) for row in rows[0::3]} == {expected_vmax}
        assert {float(row["overlay_vmax"]) for row in rows[1::3]} == {expected_vmax}
        assert len({row["difference_limit"] for row in rows[2::3]}) == 1
        assert {row["amplitude_file"] for row in rows} == {AMPLITUDE_FILENAME}
        amplitude_limits[group].add(rows[0]["amplitude_limit"])

        png = output / str(record["relative_path"])
        assert png.stat().st_size > 0
        from pyosv.viz import require_matplotlib

        image = require_matplotlib().imread(png)
        assert image.ndim == 3
        assert image.size > 0

    assert all(len(values) == 1 for values in group_indices.values())
    assert all(len(values) == 1 for values in amplitude_limits.values())
    assert len(imshow_calls) == len(records) * SECTIONS_PER_AXIS * 8
    assert {kwargs.get("interpolation") for _args, kwargs in imshow_calls} == {IMAGE_INTERPOLATION}
    halo_calls = [
        (args, kwargs)
        for args, kwargs in imshow_calls
        if args
        and isinstance(args[0], np.ndarray)
        and np.issubdtype(args[0].dtype, np.floating)
        and np.isnan(args[0]).any()
    ]
    assert len(halo_calls) == len(records) * SECTIONS_PER_AXIS * 2
    assert {kwargs["cmap"] for _args, kwargs in halo_calls} == {ATTRIBUTE_COLORMAP}
    for _args, kwargs in halo_calls:
        alpha = np.asarray(kwargs["alpha"])
        assert np.all(alpha[alpha == 0.0] == 0.0)
        assert np.allclose(alpha[alpha > 0.0], ATTRIBUTE_HALO_ALPHA)
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((output / "figure_data").glob("*.csv"))
    ) + "\n".join(str(value) for record in records for value in record.values())
    assert all(label not in public_text for label in ("RL-REF", "RL-QUAL", "Q-REF", "Synthetic"))
    assert _snapshot(figure_fixture.data_root) == before_data
    assert _snapshot(figure_fixture.bundle) == before_bundle


def test_attribute_alpha_keeps_threshold_visible_and_emphasizes_high_values() -> None:
    source_threshold = 0.8
    display_threshold = figures_module._display_threshold(source_threshold)
    values = np.asarray([[0.39, 0.4, 0.6, 1.0]], dtype=np.float32)
    alpha = figures_module._ridge_alpha(
        values,
        display_threshold=display_threshold,
        vmax=1.0,
    )

    assert display_threshold == pytest.approx(0.4)
    assert alpha[0, 0] == 0.0
    assert alpha[0, 1] == pytest.approx(ATTRIBUTE_ALPHA_MIN)
    assert ATTRIBUTE_ALPHA_MIN < alpha[0, 2] < alpha[0, 3]
    assert alpha[0, 3] == pytest.approx(ATTRIBUTE_ALPHA_MAX)
    assert values[0, 2] < source_threshold
    assert alpha[0, 2] > 0.0
    assert IMAGE_INTERPOLATION == "nearest"


def test_section_orientation_uses_crossline_as_horizontal_axis() -> None:
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    time_slice = figures_module._section(volume, "i1", 2)
    inline = figures_module._section(volume, "i3", 1)

    assert time_slice.shape == (2, 3)
    assert np.array_equal(time_slice, volume[:, :, 2])
    assert inline.shape == (4, 3)
    assert np.array_equal(inline, volume[1, :, :].T)


def test_halo_is_one_pixel_cross_outer_ring_with_fixed_alpha() -> None:
    values = np.zeros((5, 5), dtype=np.float32)
    values[2, 2] = 0.8
    display_mask = figures_module._display_mask(values, display_threshold=0.4)
    halo = figures_module._halo_mask(display_mask)

    expected = np.zeros((5, 5), dtype=bool)
    expected[1, 2] = True
    expected[2, 1] = True
    expected[2, 3] = True
    expected[3, 2] = True
    assert np.array_equal(halo, expected)
    assert not np.any(halo & display_mask)
    alpha = figures_module._halo_alpha(halo)
    assert np.all(alpha[halo] == ATTRIBUTE_HALO_ALPHA)
    assert np.all(alpha[~halo] == 0.0)
    assert ATTRIBUTE_HALO_ENABLED is True
    assert ATTRIBUTE_HALO_RADIUS_PIXELS == 1
    assert ATTRIBUTE_HALO_STRUCTURE == "cross"


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
        assert {float(row["difference_limit"]) for row in rows[2::3]} == {1.0e-6}


@pytest.mark.parametrize("source_kind", ["amplitude", "public", "candidate"])
def test_nonfinite_selected_section_is_rejected(
    figure_fixture: _FigureFixture,
    tmp_path: Path,
    source_kind: str,
) -> None:
    values = np.zeros(_SHAPE, dtype=np.float32)
    values[0, 0, 0] = np.nan
    if source_kind == "amplitude":
        path = figure_fixture.context.amplitude.resolved_path
    elif source_kind == "public":
        path = figure_fixture.context.stage_sources[0].public_reference_path
    else:
        path = figure_fixture.context.stage_sources[0].candidate_path
    _write_volume(path, values)

    with pytest.raises(ValueError, match="selected sections must contain only finite"):
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

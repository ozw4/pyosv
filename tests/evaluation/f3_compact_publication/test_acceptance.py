from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyosv.cli import f3_compact_publication as compact_cli
from pyosv.evaluation.f3_compact_publication import source as source_module
from pyosv.evaluation.f3_compact_publication.config import STAGE_ORDER
from pyosv.evaluation.f3d_mode_comparison import (
    F3_BUFFERED_PERCENTILE,
    F3_BUFFER_RADIUS,
    F3_METRIC_SCHEMA_VERSION,
    F3_REFERENCE_STAGE_FILES,
    F3CellReference,
    F3CellStageFingerprints,
    F3DatasetSpec,
    METRIC_REGISTRY,
    MetricEvidence,
    MetricRow,
)
from pyosv.evaluation.mode_comparison_publication.models import F3SourceBundle
from tests.evaluation.f3d_mode_comparison.test_integration import (
    _run_fixture as _run_completed_f3_fixture,
)
from tests.evaluation.f3d_mode_comparison.test_integration import (
    _write_fixture as _write_f3_fixture,
)

_SHAPE = (3, 4, 5)
_DTYPE = np.dtype(">f4")
_SIZE = int(np.prod(_SHAPE)) * _DTYPE.itemsize
_CELLS = ("RL-REF", "RL-QUAL", "Q-REF", "Q-QUAL")
_KINDS = ("scanner", "voting", "thinning")
_FINGERPRINTS = {
    "scanner": "1" * 64,
    "voting": "2" * 64,
    "thinning": "3" * 64,
    "skinning": "4" * 64,
}
_REFERENCE_THRESHOLDS = {"ft": 0.50, "fv": 0.55, "fvt": 0.60}
_CANDIDATE_THRESHOLDS = {"ft": 0.45, "fv": 0.50, "fvt": 0.55}
_CONTROLS = {name: "1" for name in compact_cli._ENVIRONMENT_CONTROL_NAMES}
_CODE = {"repository": "ozw4/pyosv", "git_commit": "a" * 40, "dirty": False}
_DEFINITIONS = {
    (definition.stage, definition.selection, definition.metric): definition
    for definition in METRIC_REGISTRY
}


@dataclass(frozen=True)
class _AcceptanceFixture:
    source: F3SourceBundle
    source_root: Path
    data_root: Path
    lock: Path


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


def _metric_row(
    stage: str,
    selection: str,
    metric: str,
    value: float,
) -> MetricRow:
    definition = _DEFINITIONS[(stage, selection, metric)]
    return MetricRow(
        schema_version=F3_METRIC_SCHEMA_VERSION,
        dataset_id="compact-acceptance-fixture",
        cell_label="Q-QUAL",
        scanner_backend="quality",
        workflow_mode="quality",
        stage=stage,
        region="full",
        selection=selection,
        reference_file=F3_REFERENCE_STAGE_FILES[stage],
        metric=metric,
        value=value,
        unit=definition.unit,
        direction=definition.direction,
        contrast_eligible=definition.contrast_eligible,
    )


def _metric_rows(
    references: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
) -> tuple[MetricRow, ...]:
    rows = []
    for index, stage in enumerate(STAGE_ORDER):
        rows.extend(
            (
                _metric_row(stage, "all", "reference_max", float(references[stage].max())),
                _metric_row(stage, "all", "candidate_max", float(candidates[stage].max())),
                _metric_row(stage, "all", "normalized_correlation", 0.90 - 0.05 * index),
                _metric_row(stage, "all", "mean_absolute_difference", 0.10 + 0.02 * index),
                _metric_row(stage, "all", "nonzero_fraction_ratio", 1.0 + 0.1 * index),
                _metric_row(stage, "positive_p99_radius2", "buffered_f1", 0.8 - 0.1 * index),
                _metric_row(
                    stage,
                    "positive_p99_distance",
                    "candidate_to_reference_p95",
                    1.0 + index,
                ),
                _metric_row(
                    stage,
                    "positive_p99_distance",
                    "reference_to_candidate_p95",
                    1.5 + index,
                ),
            )
        )
    return tuple(rows)


def _metric_evidence(reference_hashes: dict[str, str]) -> tuple[MetricEvidence, ...]:
    return tuple(
        MetricEvidence(
            schema_version=F3_METRIC_SCHEMA_VERSION,
            dataset_id="compact-acceptance-fixture",
            cell_label=cell,
            stage=stage,
            region="full",
            selection="positive_p99_radius2",
            reference_file=F3_REFERENCE_STAGE_FILES[stage],
            source_stage_fingerprint=_FINGERPRINTS[_KINDS[STAGE_ORDER.index(stage)]],
            reference_sha256=reference_hashes[stage],
            shape=_SHAPE,
            thresholds=(
                ("percentile", F3_BUFFERED_PERCENTILE),
                ("radius", F3_BUFFER_RADIUS),
                ("reference_threshold", _REFERENCE_THRESHOLDS[stage]),
                ("candidate_threshold", _CANDIDATE_THRESHOLDS[stage]),
            ),
        )
        for stage in STAGE_ORDER
        for cell in _CELLS
    )


@pytest.fixture
def acceptance_fixture(tmp_path: Path) -> _AcceptanceFixture:
    source_root = tmp_path / "completed-f3"
    data_root = tmp_path / "f3-data"
    source_root.mkdir()
    data_root.mkdir()
    (source_root / "completion.json").write_text('{"state":"complete"}\n', encoding="utf-8")

    ep = np.linspace(0.0, 1.0, num=np.prod(_SHAPE), dtype=np.float32).reshape(_SHAPE)
    amplitude = np.linspace(-2.0, 2.0, num=np.prod(_SHAPE), dtype=np.float32).reshape(_SHAPE)
    _write_volume(data_root / "ep.dat", ep)
    _write_volume(data_root / "xs.dat", amplitude)

    references: dict[str, np.ndarray] = {}
    candidates: dict[str, np.ndarray] = {}
    for stage_index, (stage, kind) in enumerate(zip(STAGE_ORDER, _KINDS, strict=True)):
        reference = np.zeros(_SHAPE, dtype=np.float32)
        candidate = np.zeros(_SHAPE, dtype=np.float32)
        reference[0, 1, 1] = 0.7 + 0.05 * stage_index
        reference[0, 2, 1] = 0.8 + 0.05 * stage_index
        reference[1, 2, 2] = 0.9 + 0.05 * stage_index
        reference[2, 2, 3] = 1.0 + 0.05 * stage_index
        candidate[0, 2, 1] = 0.85 + 0.05 * stage_index
        candidate[1, 2, 2] = 1.05 + 0.05 * stage_index
        candidate[2, 2, 4] = 1.15 + 0.05 * stage_index
        references[stage] = reference
        candidates[stage] = candidate
        _write_volume(data_root / F3_REFERENCE_STAGE_FILES[stage], reference)
        _write_volume(
            source_root / "stages" / kind / _FINGERPRINTS[kind] / f"{stage}.dat",
            candidate,
        )

    spec = F3DatasetSpec(
        dataset_id="compact-acceptance-fixture",
        shape=_SHAPE,
        storage_dtype=_DTYPE.str,
        files=(
            ("input", "ep.dat"),
            ("reference_fault_likelihood", "fl.dat"),
            ("reference_fault_votes", "fv.dat"),
            ("reference_thinned_fault_votes", "fvt.dat"),
        ),
        expected_bytes=_SIZE,
    )
    identities = [
        {
            "role": role,
            "size": _SIZE,
            "sha256": _sha256(data_root / filename),
            "shape": list(_SHAPE),
            "storage_dtype": _DTYPE.str,
        }
        for role, filename in spec.files
    ]
    reference_hashes = {
        stage: _sha256(data_root / F3_REFERENCE_STAGE_FILES[stage]) for stage in STAGE_ORDER
    }
    stages = F3CellStageFingerprints(**_FINGERPRINTS)
    q_qual = F3CellReference(
        label="Q-QUAL",
        backend="quality",
        workflow="quality",
        resolved_config={},
        stages=stages,
        skinning_enabled=True,
        path=source_root,
        reused=True,
    )
    cells = (
        SimpleNamespace(label="RL-REF"),
        SimpleNamespace(label="RL-QUAL"),
        SimpleNamespace(label="Q-REF"),
        q_qual,
    )
    result = SimpleNamespace(
        dataset_id=spec.dataset_id,
        volume_shape=_SHAPE,
        storage_dtype=_DTYPE.str,
        cells=cells,
        metric_rows=_metric_rows(references, candidates),
    )
    source = F3SourceBundle(
        path=source_root,
        data_root=data_root,
        dataset_spec=spec,
        run_manifest={},
        completion_sha256=_sha256(source_root / "completion.json"),
        result=result,
        metric_evidence=_metric_evidence(reference_hashes),
        dataset_identity={"dataset_id": spec.dataset_id, "files": identities},
    )
    lock = tmp_path / "uv.lock"
    lock.write_text("lock-version = 1\n", encoding="utf-8")
    return _AcceptanceFixture(
        source=source, source_root=source_root, data_root=data_root, lock=lock
    )


def test_cli_generation_validation_and_archive_acceptance(
    acceptance_fixture: _AcceptanceFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = acceptance_fixture
    output = tmp_path / "publication"
    before_source = _snapshot(fixture.source_root)
    before_data = _snapshot(fixture.data_root)
    monkeypatch.setattr(source_module, "load_f3_source", lambda *_args: fixture.source)
    monkeypatch.setattr(compact_cli, "_collect_code_identity", lambda: _CODE)
    monkeypatch.setattr(compact_cli, "_collect_environment_controls", lambda: _CONTROLS)

    generation = compact_cli.main(
        [
            "--f3-bundle",
            str(fixture.source_root),
            "--f3-data-root",
            str(fixture.data_root),
            "--environment-lock",
            str(fixture.lock),
            "--output-dir",
            str(output),
            "--pretty",
        ]
    )
    validation = compact_cli.main(["--validate-only", "--output-dir", str(output)])

    assert generation == 0
    assert validation == 0
    expected_csv = {
        f"figure_data/f3_{stage}_public_ref_vs_q_qual_i2_2.csv" for stage in STAGE_ORDER
    }
    expected_png = {f"figures/f3_{stage}_public_ref_vs_q_qual_i2_2.png" for stage in STAGE_ORDER}
    files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert files == {
        "publication_manifest.json",
        "experiment.json",
        "uv.lock",
        "f3_q_qual_vs_public_ref_summary.csv",
        "report.md",
        *expected_csv,
        *expected_png,
    }
    assert len(tuple((output / "figures").glob("*.png"))) == 3
    assert len(tuple((output / "figure_data").glob("*.csv"))) == 3

    with (output / "f3_q_qual_vs_public_ref_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        summary = tuple(csv.DictReader(stream))
    assert tuple(row["stage"] for row in summary) == STAGE_ORDER

    figure_slices = set()
    for path in sorted((output / "figure_data").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            rows = tuple(csv.DictReader(stream))
        assert len(rows) == 3
        figure_slices.add((rows[0]["axis"], rows[0]["slice_index"]))
        assert [row["source_label"] for row in rows] == [
            "PUBLIC-REF",
            "Q-QUAL",
            "Q-QUAL - PUBLIC-REF",
        ]
    assert figure_slices == {("i2", "2")}

    experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert experiment["display"] == {
        "candidate_label": "Q-QUAL",
        "public_reference_label": "PUBLIC-REF",
        "stage_order": ["ft", "fv", "fvt"],
    }
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output.rglob("*"))
        if path.suffix in {".csv", ".json", ".md"}
    )
    for excluded in ("RL-REF", "RL-QUAL", "Q-REF", "Synthetic"):
        assert excluded not in public_text
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "not geological truth" in report

    manifest = json.loads((output / "publication_manifest.json").read_text(encoding="utf-8"))
    amplitude = next(
        item for item in manifest["dataset"]["files"] if item["role"] == "seismic_amplitude"
    )
    assert amplitude == {
        "filename": "xs.dat",
        "role": "seismic_amplitude",
        "sha256": _sha256(fixture.data_root / "xs.dat"),
        "size": _SIZE,
    }
    assert _snapshot(fixture.source_root) == before_source
    assert _snapshot(fixture.data_root) == before_data

    archive = tmp_path / "publication.tar"
    with tarfile.open(archive, "w") as stream:
        stream.add(output, arcname="publication")
    extracted_root = tmp_path / "extracted"
    extracted_root.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(extracted_root)
    extracted = extracted_root / "publication"

    assert compact_cli.main(["--validate-only", "--output-dir", str(extracted)]) == 0


def test_completed_f3_bundle_flows_through_source_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "f3-data"
    spec = _write_f3_fixture(data_root)
    amplitude = np.linspace(
        -1.0,
        1.0,
        num=int(np.prod(spec.shape)),
        dtype=np.float32,
    ).reshape(spec.shape)
    _write_volume(data_root / "xs.dat", amplitude)

    f3_bundle = tmp_path / "completed-f3"
    result = _run_completed_f3_fixture(
        data_root,
        f3_bundle,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
    )
    lock = tmp_path / "uv.lock"
    lock.write_text("lock-version = 1\n", encoding="utf-8")
    output = tmp_path / "publication"
    before_bundle = _snapshot(f3_bundle)
    before_data = _snapshot(data_root)
    monkeypatch.setattr(compact_cli, "_collect_code_identity", lambda: _CODE)
    monkeypatch.setattr(compact_cli, "_collect_environment_controls", lambda: _CONTROLS)

    assert (
        compact_cli.main(
            [
                "--f3-bundle",
                str(f3_bundle),
                "--f3-data-root",
                str(data_root),
                "--environment-lock",
                str(lock),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert compact_cli.main(["--validate-only", "--output-dir", str(output)]) == 0

    manifest = json.loads((output / "publication_manifest.json").read_text(encoding="utf-8"))
    experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert manifest["source"] == {"f3_completion_sha256": _sha256(f3_bundle / "completion.json")}
    assert experiment["dataset"]["dataset_id"] == spec.dataset_id
    assert experiment["dataset"]["shape"] == list(spec.shape)
    assert experiment["dataset"]["storage_dtype"] == spec.storage_dtype
    assert {item["role"] for item in experiment["dataset"]["files"]} == {
        *spec.roles,
        "seismic_amplitude",
    }

    q_qual = next(cell for cell in result.cells if cell.label == "Q-QUAL")
    expected_fingerprints = {
        "ft": q_qual.stages.scanner,
        "fv": q_qual.stages.voting,
        "fvt": q_qual.stages.thinning,
    }
    assert {
        item["stage"]: item["q_qual_stage_fingerprint"] for item in experiment["stages"]
    } == expected_fingerprints
    with (output / "f3_q_qual_vs_public_ref_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        summary = tuple(csv.DictReader(stream))
    assert tuple(row["stage"] for row in summary) == STAGE_ORDER
    assert {
        row["stage"]: row["q_qual_stage_fingerprint"] for row in summary
    } == expected_fingerprints
    assert _snapshot(f3_bundle) == before_bundle
    assert _snapshot(data_root) == before_data

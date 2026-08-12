from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pyosv.evaluation import mode_comparison_publication as publication
from pyosv.evaluation.f3d_mode_comparison import F3DatasetSpec
from pyosv.evaluation.mode_comparison_publication import v1_bundle
from pyosv.evaluation.mode_comparison_publication.models import (
    F3SourceBundle,
    PublicationReport,
    SyntheticSourceBundle,
)
from pyosv.evaluation.mode_comparison_publication.summary import TABLE_HEADERS
from pyosv.evaluation.publication_manifest_io import validate_publication_directory

_CODE = {
    "repository": "ozw4/pyosv",
    "git_commit": "a" * 40,
    "dirty": False,
}
_CONTROLS = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_DISABLE_JIT": "0",
    "NUMBA_NUM_THREADS": "1",
    "PYOSV_ACCEL": "auto",
}
_ROOT_TABLES = {
    "publication_metrics.csv",
    "publication_contrasts.csv",
    "publication_summary.csv",
    "f3_regional_summary.csv",
    "f3_orientation_summary.csv",
    "runtime_summary.csv",
}


def test_package_default_exports_are_v1() -> None:
    assert publication.generate_publication_bundle is v1_bundle.generate_publication_bundle_v1
    assert publication.validate_publication_bundle is validate_publication_directory


def _report(root: Path) -> PublicationReport:
    root.mkdir(parents=True)
    synthetic_path = root / "synthetic-source"
    f3_path = root / "f3-source"
    data_root = root / "f3-data"
    for path in (synthetic_path, f3_path, data_root):
        path.mkdir()
        (path / "source-marker.bin").write_bytes(path.name.encode("ascii"))

    synthetic = SyntheticSourceBundle(
        path=synthetic_path,
        manifest={
            "shape": [9, 9, 9],
            "resolved_plan": {"shape": [9, 9, 9], "threshold": 0.5},
            "trials": [
                {
                    "case_id": "case-a",
                    "trial_id": "trial-1",
                    "case_generation_seed": 20260707,
                }
            ],
        },
        completion_sha256="b" * 64,
        metric_rows=(),
        contrast_rows=(),
        runtime_rows=(),
        skinning_enabled=True,
        case_order=("case-a",),
    )
    shape = (2, 3, 4)
    spec = F3DatasetSpec(
        dataset_id="fixture-f3",
        shape=shape,
        storage_dtype=">f4",
        files=(("reference", "reference.dat"), ("input", "input.dat")),
        expected_bytes=96,
    )
    f3 = F3SourceBundle(
        path=f3_path,
        data_root=data_root,
        dataset_spec=spec,
        run_manifest={"plan": {"stages": ["ft", "fv", "fvt"]}},
        completion_sha256="e" * 64,
        result=SimpleNamespace(
            dataset_id="fixture-f3",
            volume_shape=shape,
            storage_dtype=">f4",
        ),
        metric_evidence=(),
        dataset_identity={
            "dataset_id": "fixture-f3",
            "files": [
                {
                    "role": "reference",
                    "size": 96,
                    "sha256": "2" * 64,
                    "shape": list(shape),
                    "storage_dtype": ">f4",
                },
                {
                    "role": "input",
                    "size": 96,
                    "sha256": "3" * 64,
                    "shape": list(shape),
                    "storage_dtype": ">f4",
                },
            ],
        },
    )
    tables = {
        filename: ({field: None for field in header},) for filename, header in TABLE_HEADERS.items()
    }
    tables["publication_metrics.csv"] = (
        {
            **{field: None for field in TABLE_HEADERS["publication_metrics.csv"]},
            "dataset": "synthetic",
            "stage": "scanner_raw",
            "selection": "top_truth_count",
            "metric": "buffered_f1",
        },
    )
    return PublicationReport(synthetic=synthetic, f3=f3, tables=tables)


def _fake_figures(
    _report: PublicationReport,
    root: str | Path,
    *,
    png_bytes: bytes = b"fixture-png",
) -> tuple[dict[str, object], ...]:
    output = Path(root)
    (output / "figure_data").mkdir()
    (output / "figures").mkdir()
    (output / "figure_data" / "fixture.csv").write_text(
        "condition,value\nRL-REF,1.0\n", encoding="utf-8"
    )
    (output / "figures" / "fixture.png").write_bytes(png_bytes)
    return (
        {
            "figure_id": "fixture",
            "relative_path": "figures/fixture.png",
            "caption": "Fixture figure.",
            "omitted": False,
        },
    )


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    report: PublicationReport,
    output: Path,
    lock_file: Path,
    *,
    png_bytes: bytes = b"fixture-png",
    pretty: bool = False,
) -> Path:
    calls: list[tuple[object, ...]] = []

    def build(*args: object) -> PublicationReport:
        calls.append(args)
        return report

    monkeypatch.setattr(v1_bundle, "build_publication_report", build)
    monkeypatch.setattr(
        v1_bundle,
        "generate_figures",
        lambda current, root: _fake_figures(current, root, png_bytes=png_bytes),
    )
    result = v1_bundle.generate_publication_bundle_v1(
        report.synthetic.path,
        report.f3.path,
        report.f3.data_root,
        output,
        environment_lock=lock_file,
        code=_CODE,
        environment_controls=_CONTROLS,
        pretty=pretty,
    )
    assert calls == [(report.synthetic.path, report.f3.path, report.f3.data_root)]
    return result


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("pretty", [False, True])
def test_generates_self_validating_v1_bundle_with_expected_artifact_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pretty: bool,
) -> None:
    report = _report(tmp_path / "sources")
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock-version = 1\n")
    snapshots = {
        "synthetic": _snapshot(report.synthetic.path),
        "f3": _snapshot(report.f3.path),
        "data": _snapshot(report.f3.data_root),
    }
    output = tmp_path / f"publication-{pretty}"

    result = _generate(monkeypatch, report, output, lock_file, pretty=pretty)

    assert result == output
    manifest = validate_publication_directory(output)
    assert {path.name for path in output.iterdir()} == {
        "publication_manifest.json",
        "experiment.json",
        "uv.lock",
        *_ROOT_TABLES,
        "figure_data",
        "figures",
        "report.md",
    }
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    assert artifacts["uv.lock"]["tier"] == "primary"
    assert artifacts["uv.lock"]["role"] == "environment_lock"
    assert artifacts["experiment.json"]["tier"] == "primary"
    assert artifacts["experiment.json"]["role"] == "resolved_experiment"
    assert all(artifacts[name]["tier"] == "primary" for name in _ROOT_TABLES)
    assert artifacts["figure_data/fixture.csv"]["tier"] == "primary"
    assert artifacts["figure_data/fixture.csv"]["role"] == "figure_data"
    assert artifacts["figures/fixture.png"]["tier"] == "derived"
    assert artifacts["figures/fixture.png"]["role"] == "figure"
    assert artifacts["report.md"]["tier"] == "derived"
    assert artifacts["report.md"]["role"] == "report"
    assert manifest["environment"]["lock_sha256"] == artifacts["uv.lock"]["sha256"]
    assert manifest["experiment"]["config_sha256"] == artifacts["experiment.json"]["sha256"]

    with (output / "publication_metrics.csv").open(newline="", encoding="utf-8") as stream:
        assert tuple(next(csv.reader(stream))) == TABLE_HEADERS["publication_metrics.csv"]
    report_text = (output / "report.md").read_text(encoding="utf-8")
    assert "known truth" in report_text
    assert "public-reference agreement" in report_text
    assert report.synthetic.completion_sha256 in report_text
    assert report.f3.completion_sha256 in report_text
    assert str(report.synthetic.path) not in report_text
    assert str(report.f3.path) not in report_text
    assert str(report.f3.data_root) not in report_text
    assert _snapshot(report.synthetic.path) == snapshots["synthetic"]
    assert _snapshot(report.f3.path) == snapshots["f3"]
    assert _snapshot(report.f3.data_root) == snapshots["data"]


def test_real_source_fixtures_flow_through_v1_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_bundles: dict[str, Any],
) -> None:
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock-version = 1\n")
    output = tmp_path / "publication"
    monkeypatch.setattr(v1_bundle, "generate_figures", _fake_figures)

    result = v1_bundle.generate_publication_bundle_v1(
        source_bundles["synthetic"],
        source_bundles["f3"],
        source_bundles["data_root"],
        output,
        environment_lock=lock_file,
        code=_CODE,
        environment_controls=_CONTROLS,
    )

    assert result == output
    validate_publication_directory(output)
    for filename, expected_header in TABLE_HEADERS.items():
        with (output / filename).open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            assert tuple(next(reader)) == expected_header
            rows = list(reader)
        assert rows, f"{filename} must contain at least one data row"
        assert all(len(row) == len(expected_header) and any(row) for row in rows)


def test_publication_id_ignores_timestamp_and_png_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path / "sources")
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"same lock\n")

    monkeypatch.setattr(v1_bundle, "_created_at_utc", lambda: "2026-08-09T00:00:00Z")
    first = _generate(
        monkeypatch,
        report,
        tmp_path / "first",
        lock_file,
        png_bytes=b"first png",
    )
    first_manifest = validate_publication_directory(first)

    monkeypatch.setattr(v1_bundle, "_created_at_utc", lambda: "2026-08-10T00:00:00Z")
    second = _generate(
        monkeypatch,
        report,
        tmp_path / "second",
        lock_file,
        png_bytes=b"second png",
    )
    second_manifest = validate_publication_directory(second)

    assert first_manifest["created_at_utc"] != second_manifest["created_at_utc"]
    assert first_manifest["publication_id"] == second_manifest["publication_id"]
    first_png = next(
        item for item in first_manifest["artifacts"] if item["path"] == "figures/fixture.png"
    )
    second_png = next(
        item for item in second_manifest["artifacts"] if item["path"] == "figures/fixture.png"
    )
    assert first_png["sha256"] != second_png["sha256"]


def test_pretty_formatting_does_not_change_publication_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path / "sources")
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"same lock\n")
    compact_output = tmp_path / "compact"
    pretty_output = tmp_path / "pretty"
    monkeypatch.setattr(v1_bundle, "_created_at_utc", lambda: "2026-08-09T00:00:00Z")

    _generate(monkeypatch, report, compact_output, lock_file, pretty=False)
    _generate(monkeypatch, report, pretty_output, lock_file, pretty=True)

    compact_manifest = validate_publication_directory(compact_output)
    pretty_manifest = validate_publication_directory(pretty_output)
    assert compact_manifest["publication_id"] == pretty_manifest["publication_id"]
    assert (compact_output / "experiment.json").read_bytes() == (
        pretty_output / "experiment.json"
    ).read_bytes()
    assert (compact_output / "publication_manifest.json").read_bytes() != (
        pretty_output / "publication_manifest.json"
    ).read_bytes()


@pytest.mark.parametrize("location", ["synthetic", "f3", "data"])
def test_rejects_output_inside_source_or_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    report = _report(tmp_path / "sources")
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock\n")
    roots = {
        "synthetic": report.synthetic.path,
        "f3": report.f3.path,
        "data": report.f3.data_root,
    }
    monkeypatch.setattr(v1_bundle, "build_publication_report", lambda *_args: report)

    with pytest.raises(ValueError, match="inside"):
        v1_bundle.generate_publication_bundle_v1(
            report.synthetic.path,
            report.f3.path,
            report.f3.data_root,
            roots[location] / "publication",
            environment_lock=lock_file,
            code=_CODE,
            environment_controls=_CONTROLS,
        )


def test_rejects_existing_output_before_loading_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publication"
    output.mkdir()
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock\n")
    monkeypatch.setattr(
        v1_bundle,
        "build_publication_report",
        lambda *_args: pytest.fail("source report must not be loaded"),
    )

    with pytest.raises(FileExistsError):
        v1_bundle.generate_publication_bundle_v1(
            "synthetic",
            "f3",
            "data",
            output,
            environment_lock=lock_file,
            code=_CODE,
            environment_controls=_CONTROLS,
        )


@pytest.mark.parametrize(
    ("missing_table", "unknown_table"),
    [
        ("runtime_summary.csv", None),
        (None, "unexpected.csv"),
    ],
)
def test_rejects_root_table_file_set_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_table: str | None,
    unknown_table: str | None,
) -> None:
    report = _report(tmp_path / "sources")
    tables = dict(report.tables)
    if missing_table is not None:
        del tables[missing_table]
    if unknown_table is not None:
        tables[unknown_table] = ()
    invalid_report = PublicationReport(
        synthetic=report.synthetic,
        f3=report.f3,
        tables=tables,
    )
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock\n")
    output = tmp_path / "publication"
    monkeypatch.setattr(
        v1_bundle,
        "build_publication_report",
        lambda *_args: invalid_report,
    )

    with pytest.raises(ValueError, match="publication root table set mismatch"):
        v1_bundle.generate_publication_bundle_v1(
            report.synthetic.path,
            report.f3.path,
            report.f3.data_root,
            output,
            environment_lock=lock_file,
            code=_CODE,
            environment_controls=_CONTROLS,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".publication.tmp-*")) == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            ({"first": "value"},),
            r"table\.csv row 0 field mismatch: missing=\['second'\], unknown=\[\]",
        ),
        (
            ({"first": "value", "second": 2, "extra": 3},),
            r"table\.csv row 0 field mismatch: missing=\[\], unknown=\['extra'\]",
        ),
    ],
)
def test_write_csv_rejects_row_field_mismatch_before_creating_file(
    tmp_path: Path,
    rows: tuple[dict[str, object], ...],
    message: str,
) -> None:
    path = tmp_path / "table.csv"

    with pytest.raises(ValueError, match=message):
        v1_bundle._write_csv(path, ("first", "second"), rows)

    assert not path.exists()


@pytest.mark.parametrize("failure", ["table", "figure"])
def test_generation_failure_leaves_no_final_or_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    report = _report(tmp_path / "sources")
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock\n")
    output = tmp_path / "publication"
    monkeypatch.setattr(v1_bundle, "build_publication_report", lambda *_args: report)
    if failure == "table":
        monkeypatch.setattr(
            v1_bundle,
            "_write_csv",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("table failed")),
        )
    else:
        monkeypatch.setattr(
            v1_bundle,
            "generate_figures",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("figure failed")),
        )

    with pytest.raises(RuntimeError, match=failure):
        v1_bundle.generate_publication_bundle_v1(
            report.synthetic.path,
            report.f3.path,
            report.f3.data_root,
            output,
            environment_lock=lock_file,
            code=_CODE,
            environment_controls=_CONTROLS,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".publication.tmp-*")) == []


def test_manifest_is_written_after_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path / "sources")
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock\n")
    output = tmp_path / "publication"
    real_writer = v1_bundle.write_publication_manifest
    observations: list[set[str]] = []

    def write_last(root: str | Path, manifest: dict[str, object], *, pretty: bool) -> Path:
        root_path = Path(root)
        assert not (root_path / "publication_manifest.json").exists()
        actual = {
            path.relative_to(root_path).as_posix()
            for path in root_path.rglob("*")
            if path.is_file()
        }
        expected = {item["path"] for item in manifest["artifacts"]}
        assert actual == expected
        observations.append(actual)
        return real_writer(root_path, manifest, pretty=pretty)

    monkeypatch.setattr(v1_bundle, "write_publication_manifest", write_last)

    _generate(monkeypatch, report, output, lock_file)

    assert len(observations) == 1
    assert (output / "publication_manifest.json").is_file()


def test_artifact_failure_during_manifest_write_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path / "sources")
    lock_file = tmp_path / "uv.lock"
    lock_file.write_bytes(b"lock\n")
    output = tmp_path / "publication"

    real_writer = v1_bundle.write_publication_manifest

    def fail_validation(root: str | Path, manifest: dict[str, object], *, pretty: bool) -> Path:
        Path(root, "publication_metrics.csv").write_bytes(b"corrupt")
        return real_writer(root, manifest, pretty=pretty)

    monkeypatch.setattr(v1_bundle, "write_publication_manifest", fail_validation)

    with pytest.raises(ValueError, match="(size|SHA-256)"):
        _generate(monkeypatch, report, output, lock_file)

    assert not output.exists()
    assert list(tmp_path.glob(".publication.tmp-*")) == []

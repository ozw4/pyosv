from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyosv.evaluation import f3_compact_publication as public_package
from pyosv.evaluation.f3_compact_publication import bundle as bundle_module
from pyosv.evaluation.f3_compact_publication import figures as figures_module
from pyosv.evaluation.f3_compact_publication import manifest as manifest_module
from pyosv.evaluation.f3_compact_publication import source as source_module
from pyosv.evaluation.f3_compact_publication import summary as summary_module
from pyosv.evaluation.f3_compact_publication.bundle import (
    generate_f3_compact_publication_bundle,
    validate_f3_compact_publication_bundle,
)

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
_CODE = {"repository": "ozw4/pyosv", "git_commit": "1" * 40, "dirty": False}
_STAGES = ("ft", "fv", "fvt")
_SUMMARY = (
    b"stage,public_reference_file,q_qual_stage_fingerprint,normalized_correlation,"
    b"mean_absolute_difference,nonzero_fraction_ratio,buffered_f1,"
    b"candidate_to_reference_p95_voxel,reference_to_candidate_p95_voxel\n"
    b"ft,fl.dat,111,0.9,0.1,1.0,0.8,1.0,1.0\n"
    b"fv,fv.dat,222,0.8,0.2,1.0,0.7,2.0,2.0\n"
    b"fvt,fvt.dat,333,0.7,0.3,1.0,0.6,,\n"
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dataset_files() -> list[dict[str, object]]:
    identities = (
        ("input", "ep.dat"),
        ("reference_fault_likelihood", "fl.dat"),
        ("reference_fault_votes", "fv.dat"),
        ("reference_thinned_fault_votes", "fvt.dat"),
        ("seismic_amplitude", "xs.dat"),
    )
    return [
        {
            "role": role,
            "filename": filename,
            "shape": [2, 3, 4],
            "storage_dtype": ">f4",
            "size": 96,
            "sha256": _sha(filename.encode()),
        }
        for role, filename in identities
    ]


def _experiment() -> dict[str, object]:
    return {
        "schema": "pyosv.f3_compact_publication_experiment.v1",
        "source": {"f3_completion_sha256": "a" * 64},
        "dataset": {
            "dataset_id": "compact-bundle-fixture",
            "shape": [2, 3, 4],
            "storage_dtype": ">f4",
            "files": _dataset_files(),
        },
        "display": {
            "public_reference_label": "PUBLIC-REF",
            "candidate_label": "Q-QUAL",
            "stage_order": list(_STAGES),
        },
        "stages": [],
        "slice": {
            "axis": "i2",
            "index": 1,
            "selection_policy": "public_fvt_positive_p99_peak",
            "score": 7,
            "public_fvt_reference_threshold": 0.6,
        },
        "ridge_thresholds": [],
        "visualization": {},
    }


def _summary_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "stage": stage,
            "normalized_correlation": 0.9 - index * 0.1,
            "mean_absolute_difference": 0.1 + index * 0.1,
            "nonzero_fraction_ratio": 1.0,
            "buffered_f1": 0.8 - index * 0.1,
            "candidate_to_reference_p95_voxel": None if stage == "fvt" else index + 1.0,
            "reference_to_candidate_p95_voxel": None if stage == "fvt" else index + 1.0,
        }
        for index, stage in enumerate(_STAGES)
    )


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _install_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    png_payload: bytes = b"fixture-png",
) -> tuple[Path, Path, Path, SimpleNamespace]:
    source_root = tmp_path / "source"
    data_root = tmp_path / "data"
    source_root.mkdir()
    data_root.mkdir()
    (source_root / "completion.json").write_bytes(b"validated source\n")
    for filename in ("ep.dat", "fl.dat", "fv.dat", "fvt.dat", "xs.dat"):
        (data_root / filename).write_bytes(filename.encode() + b"\n")
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"lock-version = 1\n")

    context = SimpleNamespace(
        f3=SimpleNamespace(
            completion_sha256="a" * 64,
            dataset_spec=SimpleNamespace(
                dataset_id="compact-bundle-fixture",
                shape=(2, 3, 4),
                storage_dtype=">f4",
            ),
        ),
        amplitude=SimpleNamespace(sha256=_sha(b"xs.dat")),
        selected_slice=SimpleNamespace(
            axis="i2",
            index=1,
            policy="public_fvt_positive_p99_peak",
            ridge_count_score=7,
        ),
    )
    monkeypatch.setattr(source_module, "load_compact_source", lambda *_args: context)
    monkeypatch.setattr(summary_module, "build_summary_rows", lambda _context: _summary_rows())
    monkeypatch.setattr(summary_module, "summary_csv_bytes", lambda _rows: _SUMMARY)
    monkeypatch.setattr(summary_module, "build_experiment", lambda _context: _experiment())
    monkeypatch.setattr(
        summary_module,
        "experiment_json_bytes",
        lambda experiment, *, pretty=False: json.dumps(
            experiment, sort_keys=True, separators=(",", ":")
        ).encode(),
    )

    def generate_figures(_context: object, root: str | Path) -> tuple[dict[str, object], ...]:
        output = Path(root)
        (output / "figures").mkdir()
        (output / "figure_data").mkdir()
        records = []
        for stage in _STAGES:
            figure_id = f"f3_{stage}_public_ref_vs_q_qual_i2_1"
            png = f"figures/{figure_id}.png"
            csv = f"figure_data/{figure_id}.csv"
            (output / png).write_bytes(png_payload + stage.encode())
            (output / csv).write_text(
                "panel_label,source_label\nPUBLIC-REF,PUBLIC-REF\n"
                "Q-QUAL,Q-QUAL\ndifference,Q-QUAL - PUBLIC-REF\n",
                encoding="utf-8",
            )
            records.append(
                {
                    "figure_id": figure_id,
                    "relative_path": png,
                    "figure_data_csv": csv,
                    "stage": stage,
                    "caption": f"F3 {stage} PUBLIC-REF versus Q-QUAL at i2=1.",
                }
            )
        return tuple(records)

    monkeypatch.setattr(figures_module, "generate_figures", generate_figures)
    return source_root, data_root, lock, context


def _generate(
    source_root: Path,
    data_root: Path,
    lock: Path,
    output: Path,
    *,
    pretty: bool = False,
) -> Path:
    return generate_f3_compact_publication_bundle(
        source_root,
        data_root,
        output,
        environment_lock=lock,
        code=_CODE,
        environment_controls=_CONTROLS,
        pretty=pretty,
    )


def test_package_root_exports_only_bundle_generation_and_validation() -> None:
    assert public_package.__all__ == [
        "generate_f3_compact_publication_bundle",
        "validate_f3_compact_publication_bundle",
    ]
    assert not hasattr(public_package, "CompactSourceContext")
    assert not hasattr(public_package, "generate_figures")


def test_generates_exact_layout_and_valid_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    output = _generate(source_root, data_root, lock, tmp_path / "publication")

    files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected_figure_data = {
        f"figure_data/f3_{stage}_public_ref_vs_q_qual_i2_1.csv" for stage in _STAGES
    }
    expected_figures = {f"figures/f3_{stage}_public_ref_vs_q_qual_i2_1.png" for stage in _STAGES}
    assert files == {
        "publication_manifest.json",
        "experiment.json",
        "uv.lock",
        "f3_q_qual_vs_public_ref_summary.csv",
        "report.md",
        *expected_figure_data,
        *expected_figures,
    }
    assert {path.name for path in (output / "figures").iterdir()} == {
        Path(path).name for path in expected_figures
    }
    assert {path.name for path in (output / "figure_data").iterdir()} == {
        Path(path).name for path in expected_figure_data
    }
    manifest = validate_f3_compact_publication_bundle(output)
    assert manifest == bundle_module.validate_publication_directory(output)

    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    assert artifacts["uv.lock"]["tier"] == "primary"
    assert artifacts["uv.lock"]["role"] == "environment_lock"
    assert artifacts["experiment.json"]["tier"] == "primary"
    assert artifacts["experiment.json"]["role"] == "resolved_experiment"
    assert artifacts["f3_q_qual_vs_public_ref_summary.csv"]["role"] == "summary_table"
    assert all(artifacts[path]["tier"] == "primary" for path in expected_figure_data)
    assert all(artifacts[path]["role"] == "figure_data" for path in expected_figure_data)
    assert all(artifacts[path]["tier"] == "derived" for path in expected_figures)
    assert all(artifacts[path]["role"] == "figure" for path in expected_figures)
    assert artifacts["report.md"]["tier"] == "derived"
    assert artifacts["report.md"]["role"] == "report"

    assert manifest["source"] == {"f3_completion_sha256": "a" * 64}
    assert manifest["dataset"]["dataset_id"] == "compact-bundle-fixture"
    assert manifest["environment"]["lock_sha256"] == artifacts["uv.lock"]["sha256"]
    assert manifest["experiment"]["config_sha256"] == artifacts["experiment.json"]["sha256"]


def test_report_has_required_semantics_and_only_displayed_conditions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    output = _generate(source_root, data_root, lock, tmp_path / "publication")

    report = (output / "report.md").read_text(encoding="utf-8")
    for section in (
        "# F3 PUBLIC-REF vs Q-QUAL Compact Publication",
        "## Source and experiment",
        "## Stage comparison",
        "## Selected slice",
        "## Metrics",
        "## Figures",
        "## Interpretation limits",
    ):
        assert section in report
    assert "validated F3 source bundle" in report
    assert "compact-bundle-fixture" in report
    assert "shape `(2, 3, 4)`" in report
    assert "storage dtype `>f4`" in report
    assert "Amplitude input: `xs.dat`; SHA-256" in report
    assert "Q-QUAL lineage" in report
    assert "quality-workflow-specific processing has not acted" in report
    assert "`i2=1`" in report
    assert "`public_fvt_positive_p99_peak`" in report
    assert "not geological truth" in report
    for stage in _STAGES:
        link = f"figures/f3_{stage}_public_ref_vs_q_qual_i2_1.png"
        assert f"]({link})" in report

    public_text = report + (output / "f3_q_qual_vs_public_ref_summary.csv").read_text()
    public_text += "".join(
        path.read_text() for path in sorted((output / "figure_data").glob("*.csv"))
    )
    for excluded in ("RL-REF", "RL-QUAL", "Q-REF", "Synthetic"):
        assert excluded not in public_text


def test_source_and_data_bytes_and_mtimes_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    source_before = _snapshot(source_root)
    data_before = _snapshot(data_root)

    _generate(source_root, data_root, lock, tmp_path / "publication")

    assert _snapshot(source_root) == source_before
    assert _snapshot(data_root) == data_before


def test_rejects_existing_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    output = tmp_path / "publication"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        _generate(source_root, data_root, lock, output)


@pytest.mark.parametrize("source_name", ["source", "data"])
def test_rejects_output_inside_source_or_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source_name: str
) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    parent = source_root if source_name == "source" else data_root

    with pytest.raises(ValueError, match="must not be inside"):
        _generate(source_root, data_root, lock, parent / "publication")


def test_rejects_missing_and_symlink_environment_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _generate(source_root, data_root, tmp_path / "missing.lock", tmp_path / "missing-output")

    symlink = tmp_path / "lock-link"
    symlink.symlink_to(lock)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _generate(source_root, data_root, symlink, tmp_path / "symlink-output")


@pytest.mark.parametrize(
    "name",
    [
        "publication_manifest.json",
        "experiment.json",
        "f3_q_qual_vs_public_ref_summary.csv",
        "report.md",
        "figure_data",
        "figures",
    ],
)
def test_rejects_reserved_lock_basename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    source_root, data_root, _, _ = _install_fixture(monkeypatch, tmp_path)
    lock = tmp_path / name
    lock.write_bytes(b"lock\n")

    with pytest.raises(ValueError, match="conflicts"):
        _generate(source_root, data_root, lock, tmp_path / "publication")


def test_figure_failure_removes_private_temp_and_does_not_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    output = tmp_path / "publication"

    def fail(*_args: object) -> None:
        raise RuntimeError("figure failed")

    monkeypatch.setattr(figures_module, "generate_figures", fail)
    with pytest.raises(RuntimeError, match="figure failed"):
        _generate(source_root, data_root, lock, output)

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".publication.tmp-*"))


def test_manifest_validation_failure_does_not_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    output = tmp_path / "publication"

    def fail(_root: object) -> None:
        raise ValueError("manifest validation failed")

    monkeypatch.setattr(manifest_module, "validate_publication_directory", fail)
    with pytest.raises(ValueError, match="manifest validation failed"):
        _generate(source_root, data_root, lock, output)

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".publication.tmp-*"))


def test_pretty_changes_only_manifest_formatting_and_not_primary_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root, data_root, lock, _ = _install_fixture(monkeypatch, tmp_path)
    compact = _generate(source_root, data_root, lock, tmp_path / "compact", pretty=False)
    pretty = _generate(source_root, data_root, lock, tmp_path / "pretty", pretty=True)
    compact_manifest = validate_f3_compact_publication_bundle(compact)
    pretty_manifest = validate_f3_compact_publication_bundle(pretty)

    assert (compact / "publication_manifest.json").read_bytes() != (
        pretty / "publication_manifest.json"
    ).read_bytes()
    assert compact_manifest["publication_id"] == pretty_manifest["publication_id"]
    primary_paths = {
        item["path"] for item in compact_manifest["artifacts"] if item["tier"] == "primary"
    }
    assert primary_paths == {
        item["path"] for item in pretty_manifest["artifacts"] if item["tier"] == "primary"
    }
    assert all(
        (compact / path).read_bytes() == (pretty / path).read_bytes() for path in primary_paths
    )


def test_derived_png_bytes_do_not_change_publication_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root, data_root, lock, _ = _install_fixture(
        monkeypatch, tmp_path, png_payload=b"first-png"
    )
    first = _generate(source_root, data_root, lock, tmp_path / "first")
    first_manifest = validate_f3_compact_publication_bundle(first)

    second_input = tmp_path / "second-input"
    second_input.mkdir()
    _install_fixture(monkeypatch, second_input, png_payload=b"changed-png")
    second = _generate(source_root, data_root, lock, tmp_path / "second")
    second_manifest = validate_f3_compact_publication_bundle(second)

    assert first_manifest["publication_id"] == second_manifest["publication_id"]
    first_png = next(
        item
        for item in first_manifest["artifacts"]
        if item["path"].endswith("ft_public_ref_vs_q_qual_i2_1.png")
    )
    second_png = next(
        item for item in second_manifest["artifacts"] if item["path"] == first_png["path"]
    )
    assert first_png["sha256"] != second_png["sha256"]

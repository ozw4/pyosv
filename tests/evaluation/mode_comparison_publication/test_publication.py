from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from pyosv.evaluation import f3d_mode_comparison, synthetic_mode_comparison
from pyosv.evaluation.mode_comparison_publication import (
    generate_publication_bundle,
    validate_publication_bundle,
)
from pyosv.evaluation.mode_comparison_publication import artifacts as publication_artifacts
from pyosv.evaluation.mode_comparison_publication import validation as publication_validation
from pyosv.evaluation.mode_comparison_publication.config import CANONICAL_CELL_ORDER
from pyosv.evaluation.mode_comparison_publication.semantic import (
    FIGURE_DATA_FIELD_TYPES,
    ROOT_TABLE_FIELD_TYPES,
    ROOT_TABLE_IDENTITY_FIELDS,
    build_table_contract,
    canonical_digest,
    normalize_typed_csv_row,
)
from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    run_mode_comparison,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.evaluation.mode_comparison_publication.summary import TABLE_HEADERS

from tests.evaluation.mode_comparison_publication.artifact_test_support import (
    assert_completion_matches,
    read_csv_rows,
    read_json,
    rewrite_completion,
    write_csv_rows,
    write_json,
    write_png,
)


def test_generation_preserves_sources_and_has_fixed_artifact_set(
    publication_bundle: tuple[Path, dict[str, Any]],
) -> None:
    output, sources = publication_bundle
    assert validate_publication_bundle(output)
    assert {item.name for item in output.iterdir()} == {
        "manifest.json",
        "publication_metrics.csv",
        "publication_contrasts.csv",
        "publication_summary.csv",
        "f3_regional_summary.csv",
        "f3_orientation_summary.csv",
        "runtime_summary.csv",
        "figure_manifest.json",
        "report.md",
        "figure_data",
        "figures",
        "completion.json",
    }
    assert sources["synthetic_snapshot"] == _snapshot(sources["synthetic"])
    assert sources["f3_snapshot"] == _snapshot(sources["f3"])
    assert sources["data_snapshot"] == _snapshot(sources["data_root"])
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    assert "completion.json" not in completion["required_files"]
    assert all(set(item) == {"path", "size", "sha256"} for item in completion["files"])


def test_f3_ridge_threshold_metadata_matches_records_and_figure_data(
    publication_bundle: tuple[Path, dict[str, Any]],
) -> None:
    output, _sources = publication_bundle
    figure_manifest = read_json(output / "figure_manifest.json")
    contract = figure_manifest["f3_ridge_threshold_contract"]

    assert contract["selection"] == "positive_p99_radius2"
    assert contract["percentile"] == 99.0
    assert contract["buffer_radius"] == 2.0
    assert set(contract["stages"]) == {"ft", "fv", "fvt"}
    for stage, stage_contract in contract["stages"].items():
        assert isinstance(stage_contract["reference_threshold"], float)
        assert set(stage_contract["candidate_thresholds"]) == set(CANONICAL_CELL_ORDER)

    for record in figure_manifest["figures"]:
        if record["category"] == "f3_spatial":
            assert (
                record["selection_threshold"]
                == contract["stages"][record["source_stage"]]["reference_threshold"]
            )
            assert record["candidate_selection_thresholds"] is None
        elif record["category"] == "f3_ridge_overlay":
            expected = contract["stages"]["fvt"]
            assert record["selection_threshold"] == expected["reference_threshold"]
            assert record["candidate_selection_thresholds"] == expected["candidate_thresholds"]
            header, rows = read_csv_rows(output / record["figure_data_csv"])
            assert "candidate_selection_threshold" in header
            assert [row["cell_label"] for row in rows] == list(CANONICAL_CELL_ORDER)
            for row in rows:
                assert float(row["selection_threshold"]) == expected["reference_threshold"]
                assert (
                    float(row["candidate_selection_threshold"])
                    == expected["candidate_thresholds"][row["cell_label"]]
                )
        else:
            assert record["candidate_selection_thresholds"] is None
            if not record["omitted"]:
                _header, rows = read_csv_rows(output / record["figure_data_csv"])
                assert all(row["candidate_selection_threshold"] == "" for row in rows)


def test_publication_source_runner_functions_are_never_called(
    source_bundles: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("publication generation called a source runner")

    monkeypatch.setattr(synthetic_mode_comparison, "run_mode_comparison", fail)
    monkeypatch.setattr(synthetic_mode_comparison, "run_synthetic_trial", fail)
    monkeypatch.setattr(f3d_mode_comparison, "run_scanner_stages", fail)
    monkeypatch.setattr(f3d_mode_comparison, "run_f3d_mode_comparison", fail, raising=False)
    workflow_module = __import__("pyosv.evaluation.workflow3d", fromlist=["execute_workflow3d"])
    monkeypatch.setattr(workflow_module, "execute_workflow3d", fail)

    output = tmp_path / "publication"
    generate_publication_bundle(
        source_bundles["synthetic"],
        source_bundles["f3"],
        source_bundles["data_root"],
        output,
    )
    assert validate_publication_bundle(output)


def test_existing_output_and_failed_generation_leave_no_completed_bundle(
    source_bundles: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publication"
    output.mkdir()
    with pytest.raises(FileExistsError):
        generate_publication_bundle(
            source_bundles["synthetic"],
            source_bundles["f3"],
            source_bundles["data_root"],
            output,
        )
    output.rmdir()

    original = publication_artifacts.validate_publication_bundle

    def fail_validation(path: Path) -> bool:
        del path
        raise ValueError("forced publication validation failure")

    monkeypatch.setattr(publication_artifacts, "validate_publication_bundle", fail_validation)
    with pytest.raises(ValueError, match="forced publication"):
        generate_publication_bundle(
            source_bundles["synthetic"],
            source_bundles["f3"],
            source_bundles["data_root"],
            output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".publication.tmp-*"))
    monkeypatch.setattr(publication_artifacts, "validate_publication_bundle", original)


def test_completion_is_present_before_atomic_rename(
    source_bundles: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, bool] = {}
    original = publication_artifacts._rename_new

    def check_then_rename(source: Path, destination: Path) -> None:
        observed["completion"] = (source / "completion.json").is_file()
        observed["destination_absent"] = not destination.exists()
        original(source, destination)

    monkeypatch.setattr(publication_artifacts, "_rename_new", check_then_rename)
    generate_publication_bundle(
        source_bundles["synthetic"],
        source_bundles["f3"],
        source_bundles["data_root"],
        tmp_path / "publication",
    )
    assert observed == {"completion": True, "destination_absent": True}


def test_validate_only_needs_no_matplotlib_or_sources(
    publication_bundle: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _sources = publication_bundle
    imported: list[str] = []
    original_import = builtins.__import__

    def track_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("matplotlib"):
            imported.append(name)
            raise AssertionError("publication validator imported matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)
    assert validate_publication_bundle(output)
    assert imported == []


def test_data_identity_mismatch_fails_before_output(
    source_bundles: dict[str, Any],
    tmp_path: Path,
) -> None:
    bad_data_root = tmp_path / "bad-f3-data"
    shutil.copytree(source_bundles["data_root"], bad_data_root)
    target = bad_data_root / "fl.dat"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 1
    target.write_bytes(payload)
    output = tmp_path / "publication"
    with pytest.raises(ValueError, match="identity|SHA-256|checksum"):
        generate_publication_bundle(
            source_bundles["synthetic"],
            source_bundles["f3"],
            bad_data_root,
            output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("filename", "mutation"),
    (
        ("publication_metrics.csv", lambda path: path.write_bytes(path.read_bytes() + b"\n")),
        (
            "figure_manifest.json",
            lambda path: path.write_bytes(
                path.read_bytes().replace(b'"main"', b'"supplementary"', 1)
            ),
        ),
    ),
)
def test_completion_detects_publication_tampering(
    publication_bundle: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    filename: str,
    mutation: Any,
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "tampered"
    shutil.copytree(source, output)
    mutation(output / filename)
    with pytest.raises(ValueError, match="hash|size|completion"):
        validate_publication_bundle(output)


def test_png_set_tampering_is_rejected(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "tampered"
    shutil.copytree(source, output)
    png = next((output / "figures").glob("*.png"))
    png.unlink()
    with pytest.raises(ValueError):
        validate_publication_bundle(output)


def test_disabled_synthetic_skinning_omits_skin_figure(
    source_bundles: dict[str, Any],
    tmp_path: Path,
) -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        trial_seeds=(20260707,),
        shape=(9, 9, 9),
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )
    source = write_artifact_bundle(
        run_mode_comparison(config),
        tmp_path / "synthetic-disabled",
        config=config,
    )
    output = tmp_path / "publication"
    generate_publication_bundle(
        source,
        source_bundles["f3"],
        source_bundles["data_root"],
        output,
    )
    manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    omitted = next(
        item
        for item in manifest["figures"]
        if item["figure_id"] == "synthetic_skin_buffered_f1_by_case"
    )
    assert omitted["omitted"] is True
    assert not (output / "figures" / "synthetic_skin_buffered_f1_by_case.png").exists()
    for field in (
        "figure_data_row_count",
        "figure_data_identity_fields",
        "figure_data_identity_sha256",
        "figure_data_semantic_sha256",
        "pixel_width",
        "pixel_height",
        "png_size",
        "png_sha256",
    ):
        assert omitted[field] is None
    assert validate_publication_bundle(output)

    tampered = tmp_path / "omitted-skin-record-removed"
    shutil.copytree(output, tampered)
    figure_manifest = read_json(tampered / "figure_manifest.json")
    figure_manifest["figures"] = [
        record
        for record in figure_manifest["figures"]
        if record["figure_id"] != "synthetic_skin_buffered_f1_by_case"
    ]
    write_json(tampered / "figure_manifest.json", figure_manifest)
    rewrite_completion(tampered)
    assert_completion_matches(tampered)
    with pytest.raises(ValueError):
        validate_publication_bundle(tampered)


def _snapshot(root: Path) -> dict[str, tuple[bytes, int, int, str]]:
    from tests.evaluation.mode_comparison_publication.conftest import snapshot_files

    return snapshot_files(root)


def test_publication_validator_has_no_source_runner_imports() -> None:
    source = Path(publication_validation.__file__).read_text(encoding="utf-8")
    assert "run_mode_comparison" not in source
    assert "run_f3d_mode_comparison" not in source


def _rewrite_table(
    root: Path,
    filename: str,
    mutation: Callable[[list[dict[str, str]]], None],
) -> None:
    path = root / filename
    header, rows = read_csv_rows(path)
    mutation(rows)
    write_csv_rows(path, header, rows)


def _rewrite_root_table_contract(root: Path, filename: str) -> None:
    """Make a CSV semantic digest coherent while retaining source coverage evidence."""

    header, rows = read_csv_rows(root / filename)
    assert header == TABLE_HEADERS[filename]
    typed_rows = tuple(
        normalize_typed_csv_row(
            row,
            header,
            ROOT_TABLE_FIELD_TYPES[filename],
            context=f"test.{filename}[{index}]",
        )
        for index, row in enumerate(rows)
    )
    updated = build_table_contract(
        header,
        typed_rows,
        ROOT_TABLE_IDENTITY_FIELDS[filename],
        ROOT_TABLE_FIELD_TYPES[filename],
    )
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    contract = manifest["table_contracts"][filename]
    contract.update(updated)
    write_json(manifest_path, manifest)


def _rewrite_figure_data_contract(root: Path, figure_id: str | None = None) -> None:
    """Rebuild one figure-data digest to reach duplicate-identity validation."""

    manifest_path = root / "figure_manifest.json"
    manifest = read_json(manifest_path)
    record = next(
        record
        for record in manifest["figures"]
        if not record["omitted"] and (figure_id is None or record["figure_id"] == figure_id)
    )
    header, rows = read_csv_rows(root / record["figure_data_csv"])
    from pyosv.evaluation.mode_comparison_publication.config import (
        FIGURE_DATA_HEADER,
        FIGURE_DATA_IDENTITY_FIELDS,
    )

    assert header == FIGURE_DATA_HEADER
    typed_rows = tuple(
        normalize_typed_csv_row(
            row,
            header,
            FIGURE_DATA_FIELD_TYPES,
            context=f"test.{record['figure_id']}[{index}]",
        )
        for index, row in enumerate(rows)
    )
    updated = build_table_contract(
        header,
        typed_rows,
        FIGURE_DATA_IDENTITY_FIELDS,
        FIGURE_DATA_FIELD_TYPES,
    )
    record["figure_data_row_count"] = updated["row_count"]
    record["figure_data_identity_fields"] = updated["identity_fields"]
    record["figure_data_identity_sha256"] = updated["ordered_identity_sha256"]
    record["figure_data_semantic_sha256"] = updated["ordered_semantic_rows_sha256"]
    write_json(manifest_path, manifest)


def _require_index(rows: list[dict[str, str]], predicate: Callable[[dict[str, str]], bool]) -> int:
    for index, row in enumerate(rows):
        if predicate(row):
            return index
    raise AssertionError("test fixture did not contain the requested publication row")


def _duplicate_publication_metric(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        rows.append(dict(rows[_require_index(rows, lambda row: row["dataset"] == "synthetic")]))

    _rewrite_table(root, "publication_metrics.csv", mutation)


def _remove_f3_metric(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        del rows[_require_index(rows, lambda row: row["dataset"] == "f3")]

    _rewrite_table(root, "publication_metrics.csv", mutation)


def _remove_synthetic_canonical_metric(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        del rows[
            _require_index(
                rows,
                lambda row: row["dataset"] == "synthetic" and row["cell_label"] == "RL-REF",
            )
        ]

    _rewrite_table(root, "publication_metrics.csv", mutation)


def _replace_nullable_distance_with_zero(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        index = _require_index(
            rows,
            lambda row: (
                row["dataset"] == "f3"
                and row["metric"] == "candidate_to_reference_p95"
                and row["value"] == ""
            ),
        )
        rows[index]["value"] = "0"

    _rewrite_table(root, "publication_metrics.csv", mutation)


def _remove_contrast(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        del rows[0]

    _rewrite_table(root, "publication_contrasts.csv", mutation)


def _clear_contrasts(root: Path) -> None:
    _rewrite_table(root, "publication_contrasts.csv", lambda rows: rows.clear())


def _duplicate_contrast(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        rows.append(dict(rows[0]))

    _rewrite_table(root, "publication_contrasts.csv", mutation)


def _change_contrast_raw_value(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        row = rows[0]
        raw = float(row["raw_value"]) + 0.125
        row["raw_value"] = repr(raw)
        if row["direction"] == "higher":
            row["improvement_value"] = repr(raw)
        elif row["direction"] == "lower":
            row["improvement_value"] = repr(-raw)
        else:
            row["improvement_value"] = ""

    _rewrite_table(root, "publication_contrasts.csv", mutation)


def _duplicate_summary(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        rows.append(dict(rows[0]))

    _rewrite_table(root, "publication_summary.csv", mutation)


def _remove_regional_row(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        del rows[0]

    _rewrite_table(root, "f3_regional_summary.csv", mutation)


def _duplicate_regional_row(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        rows.append(dict(rows[0]))

    _rewrite_table(root, "f3_regional_summary.csv", mutation)


def _remove_orientation_row(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        del rows[0]

    _rewrite_table(root, "f3_orientation_summary.csv", mutation)


def _duplicate_orientation_row(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        rows.append(dict(rows[0]))

    _rewrite_table(root, "f3_orientation_summary.csv", mutation)


def _remove_runtime_row(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        del rows[0]

    _rewrite_table(root, "runtime_summary.csv", mutation)


def _duplicate_runtime_row(root: Path) -> None:
    def mutation(rows: list[dict[str, str]]) -> None:
        rows.append(dict(rows[0]))

    _rewrite_table(root, "runtime_summary.csv", mutation)


@pytest.mark.parametrize(
    ("name", "mutation"),
    (
        ("duplicate-metric", _duplicate_publication_metric),
        ("missing-f3-metric", _remove_f3_metric),
        ("missing-synthetic-canonical-cell", _remove_synthetic_canonical_metric),
        ("nullable-distance-zero", _replace_nullable_distance_with_zero),
        ("missing-contrast", _remove_contrast),
        ("duplicate-contrast", _duplicate_contrast),
        ("changed-contrast-raw-value", _change_contrast_raw_value),
        ("duplicate-summary", _duplicate_summary),
        ("missing-regional-row", _remove_regional_row),
        ("duplicate-regional-row", _duplicate_regional_row),
        ("missing-orientation-row", _remove_orientation_row),
        ("duplicate-orientation-row", _duplicate_orientation_row),
        ("missing-runtime-row", _remove_runtime_row),
        ("duplicate-runtime-row", _duplicate_runtime_row),
    ),
)
def test_semantic_table_tampering_is_rejected_after_completion_rehash(
    publication_bundle: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    name: str,
    mutation: Callable[[Path], None],
) -> None:
    """A recomputed byte hash cannot conceal a table-level semantic mutation."""

    source, _sources = publication_bundle
    output = tmp_path / name
    shutil.copytree(source, output)
    mutation(output)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError):
        validate_publication_bundle(output)


@pytest.mark.parametrize(
    ("filename", "mutation"),
    (
        ("publication_metrics.csv", _duplicate_publication_metric),
        ("publication_contrasts.csv", _duplicate_contrast),
        ("publication_summary.csv", _duplicate_summary),
        ("f3_regional_summary.csv", _duplicate_regional_row),
        ("f3_orientation_summary.csv", _duplicate_orientation_row),
        ("runtime_summary.csv", _duplicate_runtime_row),
    ),
)
def test_duplicate_root_row_identity_is_rejected_after_digest_rewrite(
    publication_bundle: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    filename: str,
    mutation: Callable[[Path], None],
) -> None:
    """Duplicate rejection is independent of a recomputed table digest."""

    source, _sources = publication_bundle
    output = tmp_path / f"duplicate-identity-{filename}"
    shutil.copytree(source, output)
    mutation(output)
    _rewrite_root_table_contract(output, filename)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="duplicate publication row identity"):
        validate_publication_bundle(output)


def test_header_only_contrasts_are_rejected_after_contract_rewrite(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """An empty contrast file cannot hide behind a coherently rewritten digest."""

    source, _sources = publication_bundle
    output = tmp_path / "header-only-contrasts"
    shutil.copytree(source, output)
    _clear_contrasts(output)
    _rewrite_root_table_contract(output, "publication_contrasts.csv")
    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path)
    contract = manifest["table_contracts"]["publication_contrasts.csv"]
    contract["source_expected_identities"] = []
    contract["source_expected_identity_sha256"] = canonical_digest([])
    write_json(manifest_path, manifest)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="header-only"):
        validate_publication_bundle(output)


@pytest.mark.parametrize(
    "field",
    (
        ("synthetic_source", "identity_digest"),
        ("f3_source", "identity_digest"),
        ("f3_source", "dataset_identity_digest"),
    ),
)
def test_source_identity_tampering_is_rejected_after_completion_rehash(
    publication_bundle: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    field: tuple[str, str],
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / f"identity-{field[0]}-{field[1]}"
    shutil.copytree(source, output)
    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path)
    manifest[field[0]][field[1]] = "0" * 64
    write_json(manifest_path, manifest)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError):
        validate_publication_bundle(output)


def test_v1_publication_artifact_is_not_implicitly_upgraded(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "legacy-v1-publication"
    shutil.copytree(source, output)
    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["publication_artifact_schema_version"] = 1
    write_json(manifest_path, manifest)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="predates the semantic table contract"):
        validate_publication_bundle(output)


def test_v1_figure_contract_is_not_implicitly_upgraded(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "legacy-v1-figure-contract"
    shutil.copytree(source, output)
    figure_manifest_path = output / "figure_manifest.json"
    figure_manifest = read_json(figure_manifest_path)
    figure_manifest["publication_figure_contract_version"] = 1
    write_json(figure_manifest_path, figure_manifest)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="predates the semantic figure-data contract"):
        validate_publication_bundle(output)


@pytest.mark.parametrize("filename", ("manifest.json", "figure_manifest.json"))
def test_v2_figure_contract_is_not_implicitly_upgraded(
    publication_bundle: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    filename: str,
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / f"legacy-v2-{filename.removesuffix('.json')}"
    shutil.copytree(source, output)
    path = output / filename
    document = read_json(path)
    document["publication_figure_contract_version"] = 2
    write_json(path, document)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="predates the per-candidate ridge-threshold contract"):
        validate_publication_bundle(output)


def _ridge_overlay_record(root: Path) -> dict[str, Any]:
    figure_manifest = read_json(root / "figure_manifest.json")
    return next(
        record for record in figure_manifest["figures"] if record["category"] == "f3_ridge_overlay"
    )


def test_ridge_record_candidate_threshold_tampering_is_rejected_after_completion_rehash(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "ridge-record-candidate-threshold"
    shutil.copytree(source, output)
    path = output / "figure_manifest.json"
    figure_manifest = read_json(path)
    record = next(
        item for item in figure_manifest["figures"] if item["category"] == "f3_ridge_overlay"
    )
    record["candidate_selection_thresholds"]["RL-REF"] += 0.125
    write_json(path, figure_manifest)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="candidate selection thresholds"):
        validate_publication_bundle(output)


def test_ridge_figure_data_candidate_threshold_tampering_is_rejected_after_digest_and_completion_rehash(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "ridge-figure-data-candidate-threshold"
    shutil.copytree(source, output)
    record = _ridge_overlay_record(output)
    header, rows = read_csv_rows(output / record["figure_data_csv"])
    row = next(item for item in rows if item["cell_label"] == "RL-REF")
    row["candidate_selection_threshold"] = repr(float(row["candidate_selection_threshold"]) + 0.125)
    write_csv_rows(output / record["figure_data_csv"], header, rows)
    _rewrite_figure_data_contract(output, record["figure_id"])
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="figure-data candidate selection threshold"):
        validate_publication_bundle(output)


def test_top_level_candidate_threshold_tampering_is_rejected_after_completion_rehash(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "ridge-contract-candidate-threshold"
    shutil.copytree(source, output)
    path = output / "figure_manifest.json"
    figure_manifest = read_json(path)
    figure_manifest["f3_ridge_threshold_contract"]["stages"]["fvt"]["candidate_thresholds"][
        "RL-REF"
    ] += 0.125
    write_json(path, figure_manifest)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="candidate selection thresholds"):
        validate_publication_bundle(output)


def test_top_level_reference_threshold_tampering_is_rejected_after_completion_rehash(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "ridge-contract-reference-threshold"
    shutil.copytree(source, output)
    path = output / "figure_manifest.json"
    figure_manifest = read_json(path)
    figure_manifest["f3_ridge_threshold_contract"]["stages"]["ft"]["reference_threshold"] += 0.125
    write_json(path, figure_manifest)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="selection threshold"):
        validate_publication_bundle(output)


def test_source_paths_are_provenance_not_source_identity(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "relocated-source-provenance"
    shutil.copytree(source, output)
    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["synthetic_source"]["path"] = "/relocated/synthetic"
    manifest["f3_source"]["path"] = "/relocated/f3"
    write_json(manifest_path, manifest)
    rewrite_completion(output)
    assert_completion_matches(output)
    assert validate_publication_bundle(output)


def _remove_figure(root: Path, predicate: Callable[[dict[str, Any]], bool]) -> None:
    manifest_path = root / "figure_manifest.json"
    manifest = read_json(manifest_path)
    records = manifest["figures"]
    index = _require_index(records, predicate)
    record = records.pop(index)
    for field in ("relative_path", "figure_data_csv"):
        value = record[field]
        if isinstance(value, str):
            (root / value).unlink()
    write_json(manifest_path, manifest)


def _duplicate_figure_data_row(root: Path) -> None:
    manifest = read_json(root / "figure_manifest.json")
    record = next(record for record in manifest["figures"] if not record["omitted"])
    _rewrite_table(
        root,
        record["figure_data_csv"],
        lambda rows: rows.append(dict(rows[0])),
    )


def _change_png_dimension_field(root: Path) -> None:
    path = root / "figure_manifest.json"
    manifest = read_json(path)
    record = next(record for record in manifest["figures"] if not record["omitted"])
    record["pixel_width"] += 1
    write_json(path, manifest)


def _replace_png_with_different_dimensions(root: Path) -> None:
    manifest = read_json(root / "figure_manifest.json")
    record = next(record for record in manifest["figures"] if not record["omitted"])
    write_png(root / record["relative_path"], width=2, height=3)


def _corrupt_png_ihdr(root: Path) -> None:
    manifest = read_json(root / "figure_manifest.json")
    record = next(record for record in manifest["figures"] if not record["omitted"])
    path = root / record["relative_path"]
    payload = bytearray(path.read_bytes())
    payload[12:16] = b"BAD!"
    path.write_bytes(payload)


def test_duplicate_figure_data_identity_is_rejected_after_digest_rewrite(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "duplicate-figure-data-identity"
    shutil.copytree(source, output)
    _duplicate_figure_data_row(output)
    _rewrite_figure_data_contract(output)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError, match="duplicate publication row identity"):
        validate_publication_bundle(output)


@pytest.mark.parametrize(
    ("name", "mutation"),
    (
        (
            "removed-scalar-figure",
            lambda root: _remove_figure(
                root,
                lambda record: record["figure_id"] == "f3_normalized_correlation_by_stage",
            ),
        ),
        (
            "removed-spatial-figure",
            lambda root: _remove_figure(
                root,
                lambda record: record["category"] == "f3_spatial",
            ),
        ),
        ("duplicate-figure-data", _duplicate_figure_data_row),
        ("changed-png-dimension-field", _change_png_dimension_field),
        ("different-valid-png-dimensions", _replace_png_with_different_dimensions),
        ("broken-png-ihdr", _corrupt_png_ihdr),
    ),
)
def test_semantic_figure_tampering_is_rejected_after_completion_rehash(
    publication_bundle: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    name: str,
    mutation: Callable[[Path], None],
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / name
    shutil.copytree(source, output)
    mutation(output)
    rewrite_completion(output)
    assert_completion_matches(output)
    with pytest.raises(ValueError):
        validate_publication_bundle(output)

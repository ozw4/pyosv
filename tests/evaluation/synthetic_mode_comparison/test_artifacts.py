from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import fields, replace
from functools import cache
from pathlib import Path

import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    HASHED_BUNDLE_FILES,
    METRIC_SCHEMA_VERSION,
    REQUIRED_BUNDLE_FILES,
    RUNTIME_CONTRACT_VERSION,
    SCALAR_EVIDENCE_CONTRACT_VERSION,
    AggregateRow,
    ContrastRow,
    MetricRow,
    RuntimeRow,
    SyntheticModeComparisonConfig,
    SyntheticModeComparisonResult,
    aggregate_contrast_rows,
    aggregate_metric_rows,
    compute_contrast_rows,
    run_mode_comparison,
    validate_completed_bundle,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_mode_comparison import artifacts
from pyosv.evaluation.synthetic_quality import SyntheticScannerConfig, SyntheticSkinningConfig


@cache
def _base_result() -> SyntheticModeComparisonResult:
    return run_mode_comparison(
        SyntheticModeComparisonConfig(
            case_ids=("single_vertical_plane",),
            shape=(9, 9, 9),
        )
    )


def _fixture() -> tuple[SyntheticModeComparisonConfig, SyntheticModeComparisonResult]:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
    )
    return config, _base_result()


def _write_bundle(path: Path, *, pretty: bool = False) -> Path:
    config, result = _fixture()
    return write_artifact_bundle(result, path, config=config, pretty=pretty)


def _rehash(bundle: Path, filename: str) -> None:
    completion_path = bundle / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    payload = (bundle / filename).read_bytes()
    completion["files"][filename] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    completion_path.write_text(
        json.dumps(completion, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _set_nested(value, path: tuple[str, ...], replacement) -> None:
    for name in path[:-1]:
        value = value[name]
    value[path[-1]] = replacement


def _different_scanner_metric_value(metric: str, current: float) -> float:
    if metric == "hausdorff_p95":
        return current + 0.5
    return 0.25 if current != 0.25 else 0.75


def _read_csv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        return reader.fieldnames, list(reader)


def _write_csv_dicts(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _tamper_downstream_scalar_algebra(
    reports: list[dict], tamper: str
) -> dict[tuple[str, str, str, str], float]:
    cells = reports[0]["cells"]

    def quality_payloads(label: str) -> tuple[dict, ...]:
        cell = cells[label]
        payloads = [cell["quality"]]
        if "active_pipeline" in cell:
            payloads.append(cell["pipelines"][cell["active_pipeline"]]["quality"])
        return tuple(payloads)

    metric_updates: dict[tuple[str, str, str, str], float] = {}
    if tamper == "fv_union_count":
        for label in ("RL-REF", "RL-QUAL"):
            for quality in quality_payloads(label):
                overlap = quality["fv_top_truth_count"]["buffered_overlap_radius2"]
                overlap["union_count"] = (
                    overlap["candidate_count"]
                    + overlap["truth_count"]
                    - overlap["intersection_count"]
                    + 1
                )
    elif tamper == "fvt_buffered_numerator":
        for label in ("Q-REF", "Q-QUAL"):
            for quality in quality_payloads(label):
                overlap = quality["fvt_top_truth_count"]["buffered_overlap_radius2"]
                overlap["candidate_in_truth_buffer_count"] = overlap["candidate_count"] - 1
    elif tamper == "skin_empty_candidate_buffered_hit":
        for quality in quality_payloads("Q-REF"):
            overlap = quality["skin"]["buffered_overlap_radius2"]
            assert overlap["candidate_count"] == 0
            assert overlap["truth_count"] > 0
            overlap["truth_in_candidate_buffer_count"] = 1
            overlap["buffered_recall"] = 1.0 / overlap["truth_count"]
            overlap["buffered_f1"] = (
                2.0
                * overlap["buffered_precision"]
                * overlap["buffered_recall"]
                / (overlap["buffered_precision"] + overlap["buffered_recall"])
            )
        metric_updates.update(
            {
                ("Q-REF", "skin", "skin_cells", metric): overlap[metric]
                for metric in ("buffered_recall", "buffered_f1")
            }
        )
    elif tamper == "skin_empty_distance_penalty":
        distance_metrics = (
            "candidate_to_truth_median",
            "candidate_to_truth_p95",
            "truth_to_candidate_median",
            "truth_to_candidate_p95",
            "hausdorff_p95",
        )
        for quality in quality_payloads("Q-REF"):
            distance = quality["skin"]["surface_distance"]
            for name in (
                "candidate_to_truth_mean",
                "candidate_to_truth_median",
                "candidate_to_truth_p90",
                "candidate_to_truth_p95",
                "truth_to_candidate_mean",
                "truth_to_candidate_median",
                "truth_to_candidate_p90",
                "truth_to_candidate_p95",
                "symmetric_chamfer_mean",
                "hausdorff_p95",
            ):
                distance[name] = 0.0
        metric_updates.update(
            {("Q-REF", "skin", "skin_cells", name): 0.0 for name in distance_metrics}
        )
    elif tamper == "skin_orientation_zero":
        orientation_metrics = ("strike_median", "strike_p95", "dip_median", "dip_p95")
        for quality in quality_payloads("Q-REF"):
            orientation = quality["skin"]["orientation_error"]
            for name in (
                "strike_mean",
                "strike_median",
                "strike_p90",
                "strike_p95",
                "dip_mean",
                "dip_median",
                "dip_p90",
                "dip_p95",
            ):
                orientation[name] = 0.25
        metric_updates.update(
            {("Q-REF", "skin", "skin_cells", name): 0.25 for name in orientation_metrics}
        )
    elif tamper == "skin_edge_count_hierarchy":
        for quality in quality_payloads("Q-REF"):
            edge = quality["edge_false_positive"]["skin"]
            edge["edge_candidate_count"] = edge["candidate_count"] + 1
    elif tamper == "skin_topology_fraction":
        cell = cells["RL-REF"]
        for payload in (cell, cell["pipelines"][cell["active_pipeline"]]):
            fraction = payload["pyosv"]["skins"]["largest_skin_fraction"] - 0.1
            payload["pyosv"]["skins"]["largest_skin_fraction"] = fraction
            payload["quality"]["skin"]["topology"]["largest_skin_fraction"] = fraction
        metric_updates[("RL-REF", "skin", "skin_cells", "largest_skin_fraction")] = fraction
    else:
        raise AssertionError(f"unknown downstream algebra tamper: {tamper}")
    return metric_updates


def _coherent_publication_rows(
    result: SyntheticModeComparisonResult,
    metric_updates: dict[tuple[str, str, str, str], float],
) -> tuple[
    tuple[MetricRow, ...],
    tuple[AggregateRow, ...],
    tuple[ContrastRow, ...],
    tuple[AggregateRow, ...],
]:
    metric_rows = tuple(
        replace(row, value=metric_updates[identity])
        if (identity := (row.cell_label, row.stage, row.selection, row.metric)) in metric_updates
        else row
        for row in result.metric_rows
    )
    contrast_rows = compute_contrast_rows(metric_rows)
    return (
        metric_rows,
        aggregate_metric_rows(metric_rows),
        contrast_rows,
        aggregate_contrast_rows(contrast_rows),
    )


def _tamper_surface_distance_upper_bound(
    reports: list[dict], stage: str, selection: str
) -> dict[tuple[str, str, str, str], float]:
    distance_names = (
        "candidate_to_truth_mean",
        "candidate_to_truth_median",
        "candidate_to_truth_p90",
        "candidate_to_truth_p95",
        "truth_to_candidate_mean",
        "truth_to_candidate_median",
        "truth_to_candidate_p90",
        "truth_to_candidate_p95",
        "symmetric_chamfer_mean",
        "hausdorff_p95",
    )
    published_names = (
        "candidate_to_truth_median",
        "candidate_to_truth_p95",
        "truth_to_candidate_median",
        "truth_to_candidate_p95",
        "hausdorff_p95",
    )
    unreachable = 20.0
    metric_updates: dict[tuple[str, str, str, str], float] = {}

    for label, cell in reports[0]["cells"].items():
        distances = []
        if stage.startswith("scanner_"):
            evidence = cell.get("scanner_metric_evidence")
            if evidence is None:
                continue
            quality_entry = next(
                item
                for item in evidence
                if (item["stage"], item["selection"], item["metric"])
                == (stage, selection, "candidate_count")
            )
            distances.append(quality_entry["quality_report"]["surface_distance"])
            for item in evidence:
                if (
                    item["stage"] == stage
                    and item["selection"] == selection
                    and item["metric"] in published_names
                ):
                    item["value"] = unreachable
        else:
            quality_name = "skin" if stage == "skin" else f"{stage}_{selection}"
            if quality_name not in cell.get("quality", {}):
                continue
            payloads = [cell["quality"]]
            if "active_pipeline" in cell:
                payloads.append(cell["pipelines"][cell["active_pipeline"]]["quality"])
            for quality in payloads:
                distances.append(quality[quality_name]["surface_distance"])

        for distance in distances:
            for name in distance_names:
                distance[name] = unreachable
        metric_updates.update(
            {(label, stage, selection, name): unreachable for name in published_names}
        )

    return metric_updates


def _set_coherent_candidate_count(report: dict, candidate_count: int) -> dict[str, float]:
    overlap = report["buffered_overlap_radius2"]
    overlap["candidate_count"] = candidate_count
    overlap["union_count"] = (
        candidate_count + overlap["truth_count"] - overlap["intersection_count"]
    )
    precision = overlap["intersection_count"] / candidate_count if candidate_count else 1.0
    recall = (
        overlap["intersection_count"] / overlap["truth_count"] if overlap["truth_count"] else 1.0
    )
    overlap["precision"] = precision
    overlap["recall"] = recall
    overlap["f1"] = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    overlap["jaccard"] = (
        overlap["intersection_count"] / overlap["union_count"] if overlap["union_count"] else 1.0
    )
    buffered_precision = (
        overlap["candidate_in_truth_buffer_count"] / candidate_count if candidate_count else 1.0
    )
    buffered_recall = (
        overlap["truth_in_candidate_buffer_count"] / overlap["truth_count"]
        if overlap["truth_count"]
        else 1.0
    )
    overlap["buffered_precision"] = buffered_precision
    overlap["buffered_recall"] = buffered_recall
    overlap["buffered_f1"] = (
        2.0 * buffered_precision * buffered_recall / (buffered_precision + buffered_recall)
        if buffered_precision + buffered_recall
        else 0.0
    )

    report["surface_distance"]["candidate_count"] = candidate_count
    report["orientation_error"]["count"] = candidate_count
    edge = report.get("edge_false_positive")
    if edge is not None:
        edge["candidate_count"] = candidate_count
        edge["edge_candidate_fraction"] = (
            edge["edge_candidate_count"] / candidate_count if candidate_count else 0.0
        )
        edge["edge_false_positive_fraction_of_candidates"] = (
            edge["edge_false_positive_count"] / candidate_count if candidate_count else 0.0
        )

    return {
        "candidate_count": float(candidate_count),
        "buffered_precision": overlap["buffered_precision"],
        "buffered_f1": overlap["buffered_f1"],
        **(
            {
                "edge_false_positive_fraction_of_candidates": edge[
                    "edge_false_positive_fraction_of_candidates"
                ]
            }
            if edge is not None
            else {}
        ),
    }


def _tamper_top_truth_count_cardinality(
    reports: list[dict], stages: tuple[str, ...]
) -> dict[tuple[str, str, str, str], float]:
    metric_updates: dict[tuple[str, str, str, str], float] = {}
    cells = reports[0]["cells"]
    if stages == ("scanner_raw", "scanner_thinned"):
        for label, cell in cells.items():
            evidence = cell.get("scanner_metric_evidence")
            if evidence is None:
                continue
            scanner_quality_payloads = [cell["scanner_quality"]]
            if "active_pipeline" in cell:
                scanner_quality_payloads.append(
                    cell["pipelines"][cell["active_pipeline"]]["scanner_quality"]
                )
            candidate_count = (
                scanner_quality_payloads[0]["ft_top_truth_count"]["buffered_overlap_radius2"][
                    "candidate_count"
                ]
                + 1
            )
            for scanner_quality in scanner_quality_payloads:
                legacy_report = scanner_quality["ft_top_truth_count"]
                _set_coherent_candidate_count(
                    {
                        **legacy_report,
                        "orientation_error": scanner_quality["orientation_error"][
                            "raw_scan_top_truth_count"
                        ],
                    },
                    candidate_count,
                )
                scanner_quality["orientation_error"]["used_attributes_top_truth_count"]["count"] = (
                    candidate_count
                )

            for stage in stages:
                quality_entry = next(
                    item
                    for item in evidence
                    if (item["stage"], item["selection"], item["metric"])
                    == (stage, "top_truth_count", "candidate_count")
                )
                published = _set_coherent_candidate_count(
                    quality_entry["quality_report"], candidate_count
                )
                for item in evidence:
                    if (
                        item["stage"] == stage
                        and item["selection"] == "top_truth_count"
                        and item["metric"] in published
                    ):
                        item["value"] = published[item["metric"]]
                metric_updates.update(
                    {
                        (label, stage, "top_truth_count", metric): value
                        for metric, value in published.items()
                    }
                )
        return metric_updates

    (stage,) = stages
    quality_name = f"{stage}_top_truth_count"
    for label, cell in cells.items():
        if quality_name not in cell.get("quality", {}):
            continue
        quality_payloads = [cell["quality"]]
        if "active_pipeline" in cell:
            quality_payloads.append(cell["pipelines"][cell["active_pipeline"]]["quality"])
        published = None
        for quality in quality_payloads:
            report = {
                **quality[quality_name],
                "edge_false_positive": quality["edge_false_positive"][quality_name],
            }
            candidate_count = report["buffered_overlap_radius2"]["candidate_count"] + 1
            published = _set_coherent_candidate_count(report, candidate_count)
        assert published is not None
        metric_updates.update(
            {
                (label, stage, "top_truth_count", metric): value
                for metric, value in published.items()
            }
        )
    return metric_updates


def test_writer_creates_complete_valid_bundle_with_stable_headers(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")

    assert {path.name for path in bundle.iterdir()} == set(REQUIRED_BUNDLE_FILES)
    assert validate_completed_bundle(bundle)
    expected_headers = {
        "metrics_long.csv": MetricRow,
        "metric_aggregates.csv": AggregateRow,
        "contrasts.csv": ContrastRow,
        "contrast_aggregates.csv": AggregateRow,
        "runtime.csv": RuntimeRow,
    }
    for filename, model in expected_headers.items():
        with (bundle / filename).open(encoding="utf-8", newline="") as stream:
            assert next(csv.reader(stream)) == [field.name for field in fields(model)]

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema_version"] == artifacts.ARTIFACT_SCHEMA_VERSION == 3
    assert manifest["scalar_evidence_contract_version"] == SCALAR_EVIDENCE_CONTRACT_VERSION == 2
    assert manifest["runtime_contract_version"] == RUNTIME_CONTRACT_VERSION == 1
    assert manifest["input_config"]["case_set"] is None
    assert manifest["input_config"]["case_ids"] == ["single_vertical_plane"]
    assert manifest["resolved_plan"]["shape"] == [9, 9, 9]
    assert [cell["label"] for cell in manifest["canonical_cells"]] == [
        "RL-SCAN",
        "Q-SCAN",
        "ORACLE-REF",
        "ORACLE-QUAL",
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert manifest["trials"] == [
        {
            "order": 0,
            "case_id": "single_vertical_plane",
            "trial_id": "single_vertical_plane",
            "stochastic": False,
            "case_generation_seed": None,
            "scanner_input_seed": 20260706,
        }
    ]
    reports = json.loads((bundle / "cell_reports.json").read_text(encoding="utf-8"))
    assert reports[0]["trial_id"] == "single_vertical_plane"
    assert list(reports[0]["cells"]) == [cell["label"] for cell in manifest["canonical_cells"]]
    completion = json.loads((bundle / "completion.json").read_text(encoding="utf-8"))
    assert completion["required_files"] == list(REQUIRED_BUNDLE_FILES)
    for filename in HASHED_BUNDLE_FILES:
        payload = (bundle / filename).read_bytes()
        assert completion["files"][filename] == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }


def test_bundle_round_trip_preserves_integer_valued_real_config(tmp_path: Path) -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
        scanner_template=replace(SyntheticScannerConfig(), phi_min=0),
    )
    result = run_mode_comparison(config)

    bundle = write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert validate_completed_bundle(bundle)


@pytest.mark.parametrize("min_likelihood", (None, 0.0, 0.5, 1.0, 1.5))
def test_bundle_round_trip_accepts_nonnegative_skinner_likelihood_thresholds(
    tmp_path: Path, min_likelihood: float | None
) -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
        skinning_config=SyntheticSkinningConfig(min_likelihood=min_likelihood),
        skinner_min_likelihood_explicit=True,
    )
    result = run_mode_comparison(config)
    reports = result.as_dict()["cell_reports"]
    downstream_cells = reports[0]["cells"]

    for label in ("ORACLE-REF", "ORACLE-QUAL", "RL-REF", "RL-QUAL", "Q-REF", "Q-QUAL"):
        cell = downstream_cells[label]
        assert cell["config"]["skinning"]["min_likelihood"] == min_likelihood
        if min_likelihood is not None:
            assert cell["skinning"]["diagnostics"]["seed_threshold"] == min_likelihood
            assert cell["skinning"]["diagnostics"]["grow_threshold"] == min_likelihood

    if min_likelihood is None:
        for label in ("ORACLE-QUAL", "RL-QUAL", "Q-QUAL"):
            assert downstream_cells[label]["config"]["skinning"]["adaptive_min_likelihood"]
    if min_likelihood == 1.5:
        assert all(
            downstream_cells[label]["skinning"]["enabled"]
            and downstream_cells[label]["skinning"]["diagnostics"]["skin_primary_count"] == 0
            for label in (
                "ORACLE-REF",
                "ORACLE-QUAL",
                "RL-REF",
                "RL-QUAL",
                "Q-REF",
                "Q-QUAL",
            )
        )
        assert any(row.stage == "skin" for row in result.metric_rows)

    bundle = write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert validate_completed_bundle(bundle)
    artifact_reports = json.loads((bundle / "cell_reports.json").read_text(encoding="utf-8"))
    assert artifact_reports == reports


def test_default_manifest_records_only_minimal_case_set(tmp_path: Path) -> None:
    _, result = _fixture()
    bundle = write_artifact_bundle(
        result,
        tmp_path / "bundle",
        config=SyntheticModeComparisonConfig(shape=(9, 9, 9)),
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_config"]["case_set"] == "minimal"
    assert manifest["input_config"]["case_ids"] is None


def test_same_result_and_pretty_setting_writes_identical_hashed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = {
        "status": "available",
        "method": "git_cli",
        "commit": "1" * 40,
        "dirty": False,
    }
    monkeypatch.setattr(artifacts, "_source_provenance", lambda: provenance)
    config, result = _fixture()

    first = write_artifact_bundle(result, tmp_path / "first", config=config, pretty=True)
    second = write_artifact_bundle(result, tmp_path / "second", config=config, pretty=True)

    assert all(
        (first / name).read_bytes() == (second / name).read_bytes() for name in HASHED_BUNDLE_FILES
    )


def test_pretty_only_changes_json_whitespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = {
        "status": "available",
        "method": "git_cli",
        "commit": "1" * 40,
        "dirty": False,
    }
    monkeypatch.setattr(artifacts, "_source_provenance", lambda: provenance)
    config, result = _fixture()

    compact = write_artifact_bundle(result, tmp_path / "compact", config=config)
    pretty = write_artifact_bundle(
        result,
        tmp_path / "pretty",
        config=config,
        pretty=True,
    )

    for filename in (
        "metrics_long.csv",
        "metric_aggregates.csv",
        "contrasts.csv",
        "contrast_aggregates.csv",
        "runtime.csv",
    ):
        assert (compact / filename).read_bytes() == (pretty / filename).read_bytes()


def test_writer_rejects_noncanonical_result_order(tmp_path: Path) -> None:
    config, result = _fixture()
    result = replace(
        result,
        metric_rows=(
            replace(result.metric_rows[0], selection="z-first"),
            replace(result.metric_rows[0], selection="a-second"),
        ),
        metric_aggregates=(
            replace(result.metric_aggregates[0], selection="z-first"),
            replace(result.metric_aggregates[0], selection="a-second"),
        ),
        contrast_rows=(
            replace(result.contrast_rows[0], selection="z-first"),
            replace(result.contrast_rows[0], selection="a-second"),
        ),
        contrast_aggregates=(
            replace(result.contrast_aggregates[0], selection="z-first"),
            replace(result.contrast_aggregates[0], selection="a-second"),
        ),
        runtime_rows=(
            replace(result.runtime_rows[0], stage="z-first"),
            replace(result.runtime_rows[0], stage="a-second"),
        ),
    )

    with pytest.raises(ValueError, match="do not match canonical"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_invalid_result_is_rejected_before_any_artifact_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, result = _fixture()
    invalid = replace(result, metric_rows=result.metric_rows[:-1])
    calls: list[str] = []

    for name in (
        "_write_bytes",
        "_file_metadata",
        "_fsync_directory",
        "_finalize_bundle",
    ):
        monkeypatch.setattr(
            artifacts,
            name,
            lambda *args, _name=name, **kwargs: calls.append(_name),
        )
    monkeypatch.setattr(
        artifacts.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: calls.append("mkdtemp"),
    )

    with pytest.raises(ValueError, match="metric_rows do not match canonical"):
        write_artifact_bundle(invalid, tmp_path / "bundle", config=config)

    assert calls == []
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_existing_final_path_is_not_modified(tmp_path: Path) -> None:
    final = tmp_path / "bundle"
    final.mkdir()
    sentinel = final / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    config, result = _fixture()

    with pytest.raises(FileExistsError):
        write_artifact_bundle(result, final, config=config)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(final.iterdir()) == [sentinel]


def test_writer_rejects_result_from_a_different_config(tmp_path: Path) -> None:
    _, result = _fixture()
    different_config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(11, 11, 11),
    )
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="plan_metadata does not match the canonical plan"):
        write_artifact_bundle(result, output, config=different_config)

    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_writer_rejects_numeric_type_tampering_before_creating_output(tmp_path: Path) -> None:
    config, result = _fixture()
    plan_metadata = result.as_dict()["plan_metadata"]
    plan_metadata["shape"][0] = 9.0
    invalid = replace(result, plan_metadata=plan_metadata)
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="plan_metadata does not match the canonical plan"):
        write_artifact_bundle(invalid, output, config=config)

    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_final_path_created_during_finalize_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "bundle"
    config, result = _fixture()
    original_rename_noreplace = artifacts._rename_noreplace

    def create_destination_then_rename(temporary: Path, destination: Path) -> None:
        destination.mkdir()
        original_rename_noreplace(temporary, destination)

    monkeypatch.setattr(artifacts, "_rename_noreplace", create_destination_then_rename)
    with pytest.raises(FileExistsError):
        write_artifact_bundle(result, final, config=config)

    assert final.is_dir()
    assert not list(final.iterdir())
    assert not list(tmp_path.glob(".bundle.tmp-*"))


@pytest.mark.parametrize("failed_write", (0, 3, 7))
def test_write_failures_remove_temporary_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_write: int
) -> None:
    config, result = _fixture()
    original = artifacts._write_bytes
    calls = 0

    def failing_write(path, payload):
        nonlocal calls
        if calls == failed_write:
            raise OSError("injected write failure")
        calls += 1
        original(path, payload)

    monkeypatch.setattr(artifacts, "_write_bytes", failing_write)
    with pytest.raises(OSError, match="injected write failure"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("_json_bytes", "serialization failed"),
        ("_file_metadata", "hashing failed"),
    ),
)
def test_serialization_and_hashing_failures_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    config, result = _fixture()
    monkeypatch.setattr(
        artifacts,
        target,
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(message)),
    )

    with pytest.raises(OSError, match=message):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_file_fsync_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, result = _fixture()
    monkeypatch.setattr(
        artifacts.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises(OSError, match="fsync failed"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_pre_finalize_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, result = _fixture()
    monkeypatch.setattr(
        artifacts,
        "_fsync_directory",
        lambda path: (_ for _ in ()).throw(OSError("pre-finalize failed")),
    )

    with pytest.raises(OSError, match="pre-finalize failed"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_finalize_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, result = _fixture()
    monkeypatch.setattr(
        artifacts,
        "_finalize_bundle",
        lambda temporary, final: (_ for _ in ()).throw(OSError("finalize failed")),
    )
    with pytest.raises(OSError, match="finalize failed"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_validator_rejects_missing_unexpected_and_changed_files(tmp_path: Path) -> None:
    missing_completion = _write_bundle(tmp_path / "missing-completion")
    (missing_completion / "completion.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_completed_bundle(missing_completion)

    unexpected = _write_bundle(tmp_path / "unexpected")
    (unexpected / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        validate_completed_bundle(unexpected)

    changed = _write_bundle(tmp_path / "changed")
    with (changed / "manifest.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_completed_bundle(changed)

    same_size_change = _write_bundle(tmp_path / "same-size-change")
    manifest_path = same_size_change / "manifest.json"
    payload = bytearray(manifest_path.read_bytes())
    payload[0] = ord("[")
    manifest_path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_completed_bundle(same_size_change)


def test_validator_rejects_malformed_and_nonfinite_content_after_valid_hash(
    tmp_path: Path,
) -> None:
    malformed = _write_bundle(tmp_path / "malformed")
    (malformed / "cell_reports.json").write_text("{", encoding="utf-8")
    _rehash(malformed, "cell_reports.json")
    with pytest.raises(ValueError, match="malformed JSON"):
        validate_completed_bundle(malformed)

    malformed_csv = _write_bundle(tmp_path / "malformed-csv")
    metrics_path = malformed_csv / "metrics_long.csv"
    metrics_path.write_text('"unterminated\n', encoding="utf-8", newline="\n")
    _rehash(malformed_csv, "metrics_long.csv")
    with pytest.raises(ValueError, match="malformed CSV"):
        validate_completed_bundle(malformed_csv)

    nonfinite_json = _write_bundle(tmp_path / "nonfinite-json")
    manifest_path = nonfinite_json / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace("{", '{"bad":NaN,', 1),
        encoding="utf-8",
    )
    _rehash(nonfinite_json, "manifest.json")
    with pytest.raises(ValueError, match="non-finite JSON"):
        validate_completed_bundle(nonfinite_json)

    nonfinite_csv = _write_bundle(tmp_path / "nonfinite-csv")
    metrics_path = nonfinite_csv / "metrics_long.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    rows[1][[field.name for field in fields(MetricRow)].index("value")] = "Infinity"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    _rehash(nonfinite_csv, "metrics_long.csv")
    with pytest.raises(ValueError, match="non-finite number"):
        validate_completed_bundle(nonfinite_csv)


def test_validator_rejects_negative_truth_metric_config_after_valid_hash(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "negative-truth-metric")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_config"]["truth_metric_config"]["buffer_radius"] = -0.1
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "manifest.json")

    with pytest.raises(ValueError, match="^buffer_radius must be non-negative$"):
        validate_completed_bundle(bundle)


def test_validator_rejects_incompatible_metric_schema_after_valid_hash(
    tmp_path: Path,
) -> None:
    incompatible_manifest = _write_bundle(tmp_path / "incompatible-manifest")
    manifest_path = incompatible_manifest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metric_schema_version"] = METRIC_SCHEMA_VERSION + 1
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(incompatible_manifest, "manifest.json")
    with pytest.raises(ValueError, match="unsupported metric schema version"):
        validate_completed_bundle(incompatible_manifest)

    incompatible_rows = _write_bundle(tmp_path / "incompatible-rows")
    metrics_path = incompatible_rows / "metrics_long.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    rows[1][[field.name for field in fields(MetricRow)].index("schema_version")] = str(
        METRIC_SCHEMA_VERSION + 1
    )
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    _rehash(incompatible_rows, "metrics_long.csv")
    with pytest.raises(ValueError, match="unsupported metric schema version"):
        validate_completed_bundle(incompatible_rows)


def test_validator_explicitly_rejects_rehashed_legacy_v1_bundle(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "legacy-v1")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_schema_version"] = 1
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "manifest.json")

    with pytest.raises(ValueError, match="legacy bundle.*scanner metric evidence"):
        validate_completed_bundle(bundle)


def test_validator_explicitly_rejects_rehashed_legacy_v2_bundle(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "legacy-v2")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_schema_version"] = 2
    manifest.pop("scalar_evidence_contract_version")
    manifest.pop("runtime_contract_version")
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "manifest.json")

    with pytest.raises(ValueError, match="legacy bundle.*runtime coverage"):
        validate_completed_bundle(bundle)


def test_validator_explicitly_rejects_rehashed_scalar_contract_v1_bundle(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "legacy-scalar-v1")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scalar_evidence_contract_version"] = 1
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "manifest.json")

    with pytest.raises(ValueError, match="legacy schema-v3 bundle.*trial truth evidence"):
        validate_completed_bundle(bundle)


def test_validator_rejects_rehashed_trial_truth_evidence_tampering(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "truth-evidence-tamper")
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    reports[0]["truth_evidence"]["surface_voxel_count"] += 1
    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError, match="truth_evidence.surface_voxel_count"):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("scalar_evidence_contract_version", None, "is required"),
        ("scalar_evidence_contract_version", True, "must be an integer"),
        ("scalar_evidence_contract_version", 1.0, "must be an integer"),
        ("scalar_evidence_contract_version", -1, "unsupported scalar evidence"),
        (
            "scalar_evidence_contract_version",
            SCALAR_EVIDENCE_CONTRACT_VERSION + 1,
            "unsupported scalar evidence",
        ),
        ("runtime_contract_version", None, "is required"),
        ("runtime_contract_version", True, "must be an integer"),
        ("runtime_contract_version", 1.0, "must be an integer"),
        ("runtime_contract_version", -1, "unsupported runtime"),
        (
            "runtime_contract_version",
            RUNTIME_CONTRACT_VERSION + 1,
            "unsupported runtime",
        ),
        ("unknown_contract_version", 1, "invalid fields.*unknown"),
    ),
)
def test_validator_rejects_invalid_manifest_contract_version(
    tmp_path: Path,
    field: str,
    replacement: object,
    error: str,
) -> None:
    bundle = _write_bundle(tmp_path / f"invalid-{field}-{replacement!s}")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if replacement is None:
        manifest.pop(field)
    else:
        manifest[field] = replacement
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "manifest.json")

    with pytest.raises(ValueError, match=error):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_field",
        "unknown_field",
        "nested_unknown_field",
        "wrong_section_type",
        "wrong_scalar_type",
    ),
)
def test_validator_rejects_rehashed_invalid_cell_report_contract(
    tmp_path: Path,
    tamper: str,
) -> None:
    bundle = _write_bundle(tmp_path / tamper)
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    scanner_cell = reports[0]["cells"]["RL-SCAN"]
    if tamper == "missing_field":
        del scanner_cell["scanner_quality"]
    elif tamper == "unknown_field":
        scanner_cell["unexpected"] = {}
    elif tamper == "nested_unknown_field":
        scanner_cell["scanner_quality"]["input_association"]["unexpected"] = 0.0
    elif tamper == "wrong_section_type":
        scanner_cell["scanner_quality"] = []
    else:
        scanner_cell["scanner"]["input"]["finite_count"] = "729"
    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("RL-SCAN", "scanner", "input", "finite_count"), 730),
        (("RL-SCAN", "scanner", "input", "finite_fraction"), 2.0),
        (("RL-SCAN", "scanner", "input", "nonzero_fraction"), -0.5),
        (("RL-SCAN", "scanner", "input", "min"), 2.0),
        (("RL-SCAN", "scanner", "input", "mean"), -1.0),
        (("Q-SCAN", "scanner", "confidence", "max"), 1.1),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "ft_top_truth_count",
                "buffered_overlap_radius2",
                "candidate_count",
            ),
            -1,
        ),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "ft_top_truth_count",
                "buffered_overlap_radius2",
                "f1",
            ),
            1.1,
        ),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "ft_top_truth_count",
                "surface_distance",
                "candidate_count",
            ),
            730,
        ),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "ft_top_truth_count",
                "surface_distance",
                "candidate_to_truth_mean",
            ),
            -1.0,
        ),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "orientation_error",
                "raw_scan_top_truth_count",
                "count",
            ),
            730,
        ),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "ft_top_truth_count",
                "buffered_overlap_radius2",
                "radius",
            ),
            -1.0,
        ),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "orientation_error",
                "raw_scan_top_truth_count",
                "strike_mean",
            ),
            -1.0,
        ),
        (
            (
                "RL-REF",
                "quality",
                "edge_false_positive",
                "fv_top_truth_count",
                "edge_candidate_fraction",
            ),
            1.1,
        ),
        (
            (
                "RL-REF",
                "quality",
                "edge_false_positive",
                "fv_top_truth_count",
                "candidate_count",
            ),
            730,
        ),
        (
            (
                "RL-REF",
                "quality",
                "skin",
                "component_topology",
                "mean_skin_purity",
            ),
            1.1,
        ),
        (
            (
                "RL-REF",
                "skinning",
                "diagnostics",
                "fallback_candidate_count",
            ),
            -1,
        ),
        (
            (
                "RL-REF",
                "quality",
                "skin",
                "component_topology",
                "truth_component_count",
            ),
            -1,
        ),
        (
            (
                "RL-REF",
                "skinning",
                "diagnostics",
                "skin_fvt_to_scanner_target_distance_p95",
            ),
            -1.0,
        ),
        (("RL-REF", "config", "skinning", "min_likelihood"), -1.0),
        (("RL-REF", "skinning", "diagnostics", "seed_min_ep"), 1.1),
        (("RL-REF", "skinning", "diagnostics", "seed_threshold"), -1.0),
        (("RL-REF", "skinning", "diagnostics", "grow_threshold"), -1.0),
    ),
)
def test_validator_rejects_rehashed_impossible_scalar_evidence(
    tmp_path: Path,
    path: tuple[str, ...],
    value,
) -> None:
    bundle = _write_bundle(tmp_path / "impossible-scalar")
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    _set_nested(reports[0]["cells"], path, value)
    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("fv_union_count", "union_count"),
        ("fvt_buffered_numerator", "buffered_precision"),
        ("skin_empty_candidate_buffered_hit", "nonempty candidate mask"),
        ("skin_empty_distance_penalty", "candidate_to_truth_mean"),
        ("skin_orientation_zero", "strike_mean"),
        ("skin_edge_count_hierarchy", "edge_candidate_count"),
        ("skin_topology_fraction", "largest_skin_fraction"),
    ),
)
def test_downstream_scalar_algebra_tampering_is_rejected_across_artifact_paths(
    tmp_path: Path, tamper: str, message: str
) -> None:
    config, result = _fixture()
    reports = result.as_dict()["cell_reports"]
    metric_updates = _tamper_downstream_scalar_algebra(reports, tamper)
    metric_rows, metric_aggregates, contrast_rows, contrast_aggregates = _coherent_publication_rows(
        result, metric_updates
    )
    plan = artifacts.build_mode_comparison_plan(config)

    with pytest.raises(ValueError, match=message):
        artifacts._load_cell_reports(reports, plan)

    invalid = replace(
        result,
        cell_reports=tuple(reports),
        metric_rows=metric_rows,
        metric_aggregates=metric_aggregates,
        contrast_rows=contrast_rows,
        contrast_aggregates=contrast_aggregates,
    )
    writer_output = tmp_path / f"writer-{tamper}"
    with pytest.raises(ValueError, match=message):
        write_artifact_bundle(invalid, writer_output, config=config)
    assert not writer_output.exists()
    assert not list(tmp_path.glob(f".{writer_output.name}.tmp-*"))

    bundle = _write_bundle(tmp_path / f"completed-{tamper}")
    reports_path = bundle / "cell_reports.json"
    persisted_reports = json.loads(reports_path.read_text(encoding="utf-8"))
    persisted_updates = _tamper_downstream_scalar_algebra(persisted_reports, tamper)
    assert persisted_updates == metric_updates
    reports_path.write_text(
        json.dumps(persisted_reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    coherent_rows = {
        "metrics_long.csv": (metric_rows, MetricRow),
        "metric_aggregates.csv": (metric_aggregates, AggregateRow),
        "contrasts.csv": (contrast_rows, ContrastRow),
        "contrast_aggregates.csv": (contrast_aggregates, AggregateRow),
    }
    for filename, (rows, model) in coherent_rows.items():
        (bundle / filename).write_bytes(artifacts._csv_bytes(rows, model))
        _rehash(bundle, filename)

    with pytest.raises(ValueError, match=message):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    ("stage", "selection"),
    (
        ("scanner_raw", "top_truth_count"),
        ("scanner_thinned", "top_truth_count"),
        ("fv", "top_truth_count"),
        ("fvt", "top_truth_count"),
        ("skin", "skin_cells"),
    ),
)
def test_validator_rejects_rehashed_coherent_unreachable_surface_distances(
    tmp_path: Path, stage: str, selection: str
) -> None:
    _, result = _fixture()
    bundle = _write_bundle(tmp_path / f"unreachable-{stage}")
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    metric_updates = _tamper_surface_distance_upper_bound(reports, stage, selection)
    metric_rows, metric_aggregates, contrast_rows, contrast_aggregates = _coherent_publication_rows(
        result, metric_updates
    )

    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")
    coherent_rows = {
        "metrics_long.csv": (metric_rows, MetricRow),
        "metric_aggregates.csv": (metric_aggregates, AggregateRow),
        "contrasts.csv": (contrast_rows, ContrastRow),
        "contrast_aggregates.csv": (contrast_aggregates, AggregateRow),
    }
    for filename, (rows, model) in coherent_rows.items():
        (bundle / filename).write_bytes(artifacts._csv_bytes(rows, model))
        _rehash(bundle, filename)

    with pytest.raises(ValueError, match="exceeds the volume diagonal"):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    "stages",
    (
        ("scanner_raw", "scanner_thinned"),
        ("fv",),
        ("fvt",),
    ),
    ids=("scanner_raw_and_thinned", "fv", "fvt"),
)
def test_validator_rejects_rehashed_coherent_top_truth_count_cardinality_tampering(
    tmp_path: Path, stages: tuple[str, ...]
) -> None:
    _, result = _fixture()
    bundle = _write_bundle(tmp_path / f"cardinality-{'-'.join(stages)}")
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    metric_updates = _tamper_top_truth_count_cardinality(reports, stages)
    metric_rows, metric_aggregates, contrast_rows, contrast_aggregates = _coherent_publication_rows(
        result, metric_updates
    )

    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")
    coherent_rows = {
        "metrics_long.csv": (metric_rows, MetricRow),
        "metric_aggregates.csv": (metric_aggregates, AggregateRow),
        "contrasts.csv": (contrast_rows, ContrastRow),
        "contrast_aggregates.csv": (contrast_aggregates, AggregateRow),
    }
    for filename, (rows, model) in coherent_rows.items():
        (bundle / filename).write_bytes(artifacts._csv_bytes(rows, model))
        _rehash(bundle, filename)

    with pytest.raises(ValueError, match="top_truth_count candidate_count"):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    ("stage", "metric"),
    (
        ("scanner_raw", "buffered_f1"),
        ("scanner_thinned", "dip_median"),
    ),
)
def test_cell_report_loader_rejects_scanner_metric_evidence_quality_mismatch(
    stage: str, metric: str
) -> None:
    config, result = _fixture()
    reports = result.as_dict()["cell_reports"]
    evidence = reports[0]["cells"]["RL-SCAN"]["scanner_metric_evidence"]
    entry = next(
        item
        for item in evidence
        if item["stage"] == stage
        and item["selection"] == "top_truth_count"
        and item["metric"] == metric
    )
    entry["value"] = entry["value"] * 0.9 if entry["value"] else 0.01
    plan = artifacts.build_mode_comparison_plan(config)

    with pytest.raises(ValueError, match="does not match scanner_quality"):
        artifacts._load_cell_reports(reports, plan)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "exactly every applicable"),
        ("extra", "exactly every applicable"),
        ("duplicate", "identity does not match"),
        ("unknown", "identity does not match"),
    ),
)
def test_cell_report_loader_rejects_malformed_scanner_evidence_identity(
    mutation: str, message: str
) -> None:
    config, result = _fixture()
    reports = result.as_dict()["cell_reports"]
    evidence = reports[0]["cells"]["RL-SCAN"]["scanner_metric_evidence"]
    if mutation == "missing":
        evidence.pop()
    elif mutation == "extra":
        evidence.append(dict(evidence[-1]))
    elif mutation == "duplicate":
        for name in ("stage", "selection", "metric"):
            evidence[1][name] = evidence[0][name]
    else:
        evidence[1]["metric"] = "unknown_metric"
    plan = artifacts.build_mode_comparison_plan(config)

    with pytest.raises(ValueError, match=message):
        artifacts._load_cell_reports(reports, plan)


@pytest.mark.parametrize(
    "metric",
    (
        "buffered_f1",
        "hausdorff_p95",
        "edge_false_positive_fraction_of_candidates",
    ),
)
def test_writer_rejects_coordinated_scanner_thinned_metric_tampering(
    tmp_path: Path, metric: str
) -> None:
    config, result = _fixture()
    reports = result.as_dict()["cell_reports"]
    cells = reports[0]["cells"]
    original = next(
        entry["value"]
        for entry in cells["RL-SCAN"]["scanner_metric_evidence"]
        if (entry["stage"], entry["selection"], entry["metric"])
        == ("scanner_thinned", "top_truth_count", metric)
    )
    tampered = _different_scanner_metric_value(metric, original)
    report_name = {
        "buffered_f1": "buffered_overlap_radius2",
        "hausdorff_p95": "surface_distance",
        "edge_false_positive_fraction_of_candidates": "edge_false_positive",
    }[metric]
    for label in ("RL-SCAN", "RL-REF", "RL-QUAL"):
        evidence = cells[label]["scanner_metric_evidence"]
        metric_entry = next(
            item
            for item in evidence
            if (item["stage"], item["selection"], item["metric"])
            == ("scanner_thinned", "top_truth_count", metric)
        )
        metric_entry["value"] = tampered
        quality_entry = next(
            item
            for item in evidence
            if (item["stage"], item["selection"], item["metric"])
            == ("scanner_thinned", "top_truth_count", "candidate_count")
        )
        quality_entry["quality_report"][report_name][metric] = tampered

    metric_rows = tuple(
        replace(row, value=tampered)
        if (
            row.cell_label,
            row.stage,
            row.selection,
            row.metric,
        )
        == ("RL-SCAN", "scanner_thinned", "top_truth_count", metric)
        else row
        for row in result.metric_rows
    )
    contrast_rows = compute_contrast_rows(metric_rows)
    invalid = replace(
        result,
        cell_reports=tuple(reports),
        metric_rows=metric_rows,
        contrast_rows=contrast_rows,
        metric_aggregates=aggregate_metric_rows(metric_rows),
        contrast_aggregates=aggregate_contrast_rows(contrast_rows),
    )
    output = tmp_path / f"coordinated-{metric}"

    with pytest.raises(ValueError, match="invalid scalar evidence"):
        write_artifact_bundle(invalid, output, config=config)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))


def test_scanner_quality_loader_rejects_distinct_raw_and_thinned_counts() -> None:
    config, result = _fixture()
    plan = artifacts.build_mode_comparison_plan(config)
    quality = result.as_dict()["cell_reports"][0]["cells"]["RL-SCAN"]["scanner_quality"]
    orientations = quality["orientation_error"]
    orientations["used_attributes_top_truth_count"]["count"] = (
        orientations["raw_scan_top_truth_count"]["count"] + 1
    )

    with pytest.raises(ValueError, match="top_truth_count candidate_count"):
        artifacts._load_scanner_quality_report(
            quality,
            buffer_radius=plan.truth_metric_config.buffer_radius,
            shape=plan.shape,
            context="scanner_quality",
        )


def test_validator_rejects_skin_orientation_candidate_count_mismatch(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "skin-orientation-count-mismatch")
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    skin_quality = reports[0]["cells"]["RL-REF"]["quality"]["skin"]
    mismatched_count = skin_quality["orientation_error"]["count"] + 1
    skin_quality["orientation_error"]["count"] = mismatched_count
    skin_quality["topology"]["cell_count"] = mismatched_count
    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError, match="candidate counts must match"):
        validate_completed_bundle(bundle)


def test_cell_report_loader_requires_exact_matching_skin_topologies() -> None:
    config, result = _fixture()
    reports = result.as_dict()["cell_reports"]
    cell = reports[0]["cells"]["RL-REF"]
    assert cell["pyosv"]["skins"]["largest_skin_fraction"] == 1.0
    cell["pyosv"]["skins"]["largest_skin_fraction"] = 1

    with pytest.raises(ValueError, match="pyosv.skins does not match quality.skin.topology"):
        artifacts._load_cell_reports(reports, artifacts.build_mode_comparison_plan(config))


def test_bundle_requires_empty_skin_topology_when_disabled(tmp_path: Path) -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )
    result = run_mode_comparison(config)
    bundle = write_artifact_bundle(result, tmp_path / "disabled-skinning", config=config)
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    cell = reports[0]["cells"]["RL-REF"]
    topology = _fixture()[1].as_dict()["cell_reports"][0]["cells"]["RL-REF"]["pyosv"]["skins"]
    for payload in (cell, cell["pipelines"][cell["active_pipeline"]]):
        payload["pyosv"]["skins"] = topology
    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError, match="must be empty when skinning is disabled"):
        validate_completed_bundle(bundle)


def test_bundle_accepts_explicit_small_skin_size_for_both_workflows(tmp_path: Path) -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
        skinning_config=SyntheticSkinningConfig(small_skin_size=2),
    )
    result = run_mode_comparison(config)
    reports = result.as_dict()["cell_reports"]
    for label in ("RL-REF", "RL-QUAL"):
        payload = reports[0]["cells"][label]
        assert payload["config"]["skinning"]["small_skin_size"] == 2
        assert payload["quality"]["skin"]["topology"]["small_skin_size"] == 2

    bundle = write_artifact_bundle(result, tmp_path / "small-skin-size", config=config)
    assert validate_completed_bundle(bundle)


def test_cell_report_loader_accepts_nonnegative_coverage_above_one() -> None:
    config, result = _fixture()
    reports = result.as_dict()["cell_reports"]
    cell = reports[0]["cells"]["RL-REF"]
    for payload in (cell, cell["pipelines"][cell["active_pipeline"]]):
        payload["skinning"]["diagnostics"]["skin_primary_cell_coverage_of_fvt_positive"] = 1.1

    artifacts._load_cell_reports(reports, artifacts.build_mode_comparison_plan(config))


def test_cell_report_loader_separates_planarity_and_likelihood_threshold_ranges() -> None:
    config, result = _fixture()
    reports = result.as_dict()["cell_reports"]
    cell = reports[0]["cells"]["RL-REF"]
    for payload in (cell, cell["pipelines"][cell["active_pipeline"]]):
        diagnostics = payload["skinning"]["diagnostics"]
        diagnostics["seed_threshold"] = 1.1
        diagnostics["grow_threshold"] = 1.1

    artifacts._load_cell_reports(reports, artifacts.build_mode_comparison_plan(config))

    for payload in (cell, cell["pipelines"][cell["active_pipeline"]]):
        payload["skinning"]["diagnostics"]["seed_min_ep"] = 1.1

    with pytest.raises(ValueError, match="closed unit interval"):
        artifacts._load_cell_reports(reports, artifacts.build_mode_comparison_plan(config))


@pytest.mark.parametrize(
    "tamper",
    ("array_shape", "scanner_config", "active_pipeline_duplicate"),
)
def test_validator_binds_cell_report_contents_to_resolved_plan(
    tmp_path: Path,
    tamper: str,
) -> None:
    bundle = _write_bundle(tmp_path / tamper)
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    cells = reports[0]["cells"]
    if tamper == "array_shape":
        cells["RL-SCAN"]["scanner"]["ft"]["shape"] = [8, 9, 9]
    elif tamper == "scanner_config":
        cells["RL-SCAN"]["scanner"]["config"]["phi_min"] = 1.0
    else:
        duplicate = cells["RL-REF"]["pipelines"]["scanner"]
        duplicate["scanner"]["input"]["mean"] += 0.01
    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError, match="cell_reports"):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    "tamper",
    (
        "artifact_schema_version",
        "metric_schema_version",
        "canonical_cell_order",
        "resolved_plan_integer",
    ),
)
def test_validator_rejects_rehashed_wrong_manifest_scalar_types(
    tmp_path: Path, tamper: str
) -> None:
    bundle = _write_bundle(tmp_path / tamper)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "artifact_schema_version":
        manifest["artifact_schema_version"] = True
    elif tamper == "metric_schema_version":
        manifest["metric_schema_version"] = True
    elif tamper == "canonical_cell_order":
        manifest["canonical_cells"][0]["order"] = 0.0
    else:
        manifest["resolved_plan"]["shape"][0] = 9.0
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "manifest.json")

    with pytest.raises(ValueError):
        validate_completed_bundle(bundle)


def test_cell_report_loader_rejects_nonfinite_nested_scalar(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    reports = json.loads((bundle / "cell_reports.json").read_text(encoding="utf-8"))
    reports[0]["cells"]["RL-SCAN"]["scanner"]["input"]["mean"] = float("nan")
    config, _ = _fixture()
    plan = artifacts.build_mode_comparison_plan(config)

    with pytest.raises(ValueError, match="finite"):
        artifacts._load_cell_reports(reports, plan)


@pytest.mark.parametrize("tamper", ("drop_metric", "change_value", "change_unit"))
def test_validator_rejects_rehashed_cross_file_metric_tampering(
    tmp_path: Path, tamper: str
) -> None:
    bundle = _write_bundle(tmp_path / tamper)
    metrics_path = bundle / "metrics_long.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    if tamper == "drop_metric":
        rows.pop(1)
    elif tamper == "change_value":
        rows[1][header.index("value")] = "0.123456789"
    else:
        rows[1][header.index("unit")] = "wrong-unit"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    _rehash(bundle, "metrics_long.csv")

    with pytest.raises(ValueError):
        validate_completed_bundle(bundle)


def test_validator_rejects_coordinated_confidence_and_aggregate_tampering(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "coordinated-confidence")
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    cells = reports[0]["cells"]
    metrics_path = bundle / "metrics_long.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        metric_rows = list(csv.reader(stream))
    metric_header = metric_rows[0]
    metric_row = next(
        row
        for row in metric_rows[1:]
        if (
            row[metric_header.index("cell_label")],
            row[metric_header.index("stage")],
            row[metric_header.index("selection")],
            row[metric_header.index("metric")],
        )
        == ("Q-SCAN", "scanner_confidence", "finite", "confidence_mean")
    )
    tampered_value = float(metric_row[metric_header.index("value")]) + 0.01
    for label in ("Q-SCAN", "Q-REF", "Q-QUAL"):
        evidence = next(
            entry
            for entry in cells[label]["scanner_metric_evidence"]
            if (entry["stage"], entry["selection"], entry["metric"])
            == ("scanner_confidence", "finite", "confidence_mean")
        )
        evidence["value"] = tampered_value
    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    metric_row[metric_header.index("value")] = repr(tampered_value)
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(metric_rows)
    _rehash(bundle, "metrics_long.csv")

    aggregates_path = bundle / "metric_aggregates.csv"
    with aggregates_path.open(encoding="utf-8", newline="") as stream:
        aggregate_rows = list(csv.reader(stream))
    aggregate_header = aggregate_rows[0]
    aggregate_row = next(
        row
        for row in aggregate_rows[1:]
        if (
            row[aggregate_header.index("cell_label")],
            row[aggregate_header.index("stage")],
            row[aggregate_header.index("selection")],
            row[aggregate_header.index("metric")],
        )
        == ("Q-SCAN", "scanner_confidence", "finite", "confidence_mean")
    )
    for name in ("mean", "median", "min", "max", "q25", "q75"):
        aggregate_row[aggregate_header.index(name)] = repr(tampered_value)
    with aggregates_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(aggregate_rows)
    _rehash(bundle, "metric_aggregates.csv")

    with pytest.raises(ValueError, match="scanner_quality/scanner evidence"):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    "metric",
    (
        "buffered_f1",
        "hausdorff_p95",
        "edge_false_positive_fraction_of_candidates",
    ),
)
def test_validator_rejects_rehashed_coordinated_scanner_thinned_metric_tampering(
    tmp_path: Path, metric: str
) -> None:
    bundle = _write_bundle(tmp_path / f"coordinated-scanner-thinned-{metric}")
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    cells = reports[0]["cells"]
    original = next(
        entry["value"]
        for entry in cells["RL-SCAN"]["scanner_metric_evidence"]
        if (entry["stage"], entry["selection"], entry["metric"])
        == ("scanner_thinned", "top_truth_count", metric)
    )
    tampered = _different_scanner_metric_value(metric, original)
    report_name = {
        "buffered_f1": "buffered_overlap_radius2",
        "hausdorff_p95": "surface_distance",
        "edge_false_positive_fraction_of_candidates": "edge_false_positive",
    }[metric]
    for label in ("RL-SCAN", "RL-REF", "RL-QUAL"):
        evidence = cells[label]["scanner_metric_evidence"]
        metric_entry = next(
            item
            for item in evidence
            if (item["stage"], item["selection"], item["metric"])
            == ("scanner_thinned", "top_truth_count", metric)
        )
        metric_entry["value"] = tampered
        quality_entry = next(
            item
            for item in evidence
            if (item["stage"], item["selection"], item["metric"])
            == ("scanner_thinned", "top_truth_count", "candidate_count")
        )
        quality_entry["quality_report"][report_name][metric] = tampered
    reports_path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    metric_fields, metric_rows = _read_csv_dicts(bundle / "metrics_long.csv")
    quality_value = next(
        float(row["value"])
        for row in metric_rows
        if (row["cell_label"], row["stage"], row["selection"], row["metric"])
        == ("Q-SCAN", "scanner_thinned", "top_truth_count", metric)
    )
    metric_row = next(
        row
        for row in metric_rows
        if (row["cell_label"], row["stage"], row["selection"], row["metric"])
        == ("RL-SCAN", "scanner_thinned", "top_truth_count", metric)
    )
    metric_row["value"] = repr(tampered)
    _write_csv_dicts(bundle / "metrics_long.csv", metric_fields, metric_rows)
    _rehash(bundle, "metrics_long.csv")

    aggregate_fields, aggregate_rows = _read_csv_dicts(bundle / "metric_aggregates.csv")
    metric_aggregate = next(
        row
        for row in aggregate_rows
        if (row["cell_label"], row["stage"], row["selection"], row["metric"])
        == ("RL-SCAN", "scanner_thinned", "top_truth_count", metric)
    )
    for name in ("mean", "median", "min", "max", "q25", "q75"):
        metric_aggregate[name] = repr(tampered)
    _write_csv_dicts(bundle / "metric_aggregates.csv", aggregate_fields, aggregate_rows)
    _rehash(bundle, "metric_aggregates.csv")

    raw_value = quality_value - tampered
    contrast_fields, contrast_rows = _read_csv_dicts(bundle / "contrasts.csv")
    contrast_row = next(
        row
        for row in contrast_rows
        if (row["contrast_name"], row["stage"], row["selection"], row["metric"])
        == ("scanner_only_effect", "scanner_thinned", "top_truth_count", metric)
    )
    contrast_row["raw_value"] = repr(raw_value)
    contrast_row["improvement_value"] = repr(
        raw_value if contrast_row["direction"] == "higher" else -raw_value
    )
    _write_csv_dicts(bundle / "contrasts.csv", contrast_fields, contrast_rows)
    _rehash(bundle, "contrasts.csv")

    contrast_aggregate_fields, contrast_aggregate_rows = _read_csv_dicts(
        bundle / "contrast_aggregates.csv"
    )
    contrast_aggregate = next(
        row
        for row in contrast_aggregate_rows
        if (row["contrast_name"], row["stage"], row["selection"], row["metric"])
        == ("scanner_only_effect", "scanner_thinned", "top_truth_count", metric)
    )
    for name in ("mean", "median", "min", "max", "q25", "q75"):
        contrast_aggregate[name] = repr(raw_value)
    _write_csv_dicts(
        bundle / "contrast_aggregates.csv",
        contrast_aggregate_fields,
        contrast_aggregate_rows,
    )
    _rehash(bundle, "contrast_aggregates.csv")

    with pytest.raises(ValueError, match=metric):
        validate_completed_bundle(bundle)


def test_validator_rejects_rehashed_manifest_plan_split_and_missing_coverage(
    tmp_path: Path,
) -> None:
    split_plan = _write_bundle(tmp_path / "split-plan")
    manifest_path = split_plan / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolved_plan"]["shape"] = [11, 11, 11]
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(split_plan, "manifest.json")
    with pytest.raises(ValueError, match="resolved_plan does not match input_config"):
        validate_completed_bundle(split_plan)

    missing_report = _write_bundle(tmp_path / "missing-report")
    reports_path = missing_report / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    reports.clear()
    reports_path.write_text("[]\n", encoding="utf-8", newline="\n")
    _rehash(missing_report, "cell_reports.json")
    with pytest.raises(ValueError, match="exactly one report"):
        validate_completed_bundle(missing_report)


def test_git_failure_records_explicit_unavailable_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", fail_git)
    bundle = _write_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_provenance"] == {
        "status": "not_available",
        "method": "git_cli",
        "commit": None,
        "dirty": None,
    }
    assert validate_completed_bundle(bundle)


def test_source_provenance_uses_package_location_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_file = Path(artifacts.__file__).resolve()
    source_path = source_file.parent
    source_root = source_path.parents[3]
    source_relative = source_file.relative_to(source_root).as_posix()
    calls: list[list[str]] = []

    def fake_git(command, **kwargs):
        calls.append(command)
        if command == [
            "git",
            "-C",
            str(source_path),
            "rev-parse",
            "--show-toplevel",
        ]:
            return subprocess.CompletedProcess(command, 0, f"{source_root}\n", "")
        if command == [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "--error-unmatch",
            "--",
            source_relative,
        ]:
            return subprocess.CompletedProcess(command, 0, f"{source_relative}\n", "")
        if command == ["git", "-C", str(source_root), "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, f"{'1' * 40}\n", "")
        if command == ["git", "-C", str(source_root), "status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_git)

    assert artifacts._source_provenance() == {
        "status": "available",
        "method": "git_cli",
        "commit": "1" * 40,
        "dirty": False,
    }
    assert calls[0][2] == str(source_path)


def test_source_provenance_rejects_untracked_package_in_enclosing_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "consumer"
    source_file = repository / ".venv/site-packages/pyosv/evaluation/artifacts.py"
    source_relative = source_file.relative_to(repository).as_posix()
    calls: list[list[str]] = []

    def fake_git(command, **kwargs):
        calls.append(command)
        if command == [
            "git",
            "-C",
            str(source_file.parent),
            "rev-parse",
            "--show-toplevel",
        ]:
            return subprocess.CompletedProcess(command, 0, f"{repository}\n", "")
        if command == [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--error-unmatch",
            "--",
            source_relative,
        ]:
            raise subprocess.CalledProcessError(1, command)
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(artifacts, "__file__", str(source_file))
    monkeypatch.setattr(subprocess, "run", fake_git)

    assert artifacts._source_provenance() == {
        "status": "not_available",
        "method": "git_cli",
        "commit": None,
        "dirty": None,
    }
    assert len(calls) == 2


def test_writer_and_validator_leave_no_open_handles(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    validate_completed_bundle(bundle)

    shutil.rmtree(bundle)

    assert not bundle.exists()

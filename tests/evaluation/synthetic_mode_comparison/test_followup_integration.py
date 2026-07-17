from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    SyntheticModeComparisonResult,
    build_mode_comparison_plan,
    run_mode_comparison,
    validate_completed_bundle,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_mode_comparison import runner as comparison_runner
from pyosv.evaluation.synthetic_quality import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
)
from pyosv.evaluation.synthetic_quality import pipeline as quality_pipeline
from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.synthetic_quality import runner as quality_runner
from pyosv.evaluation.synthetic_quality import scanner as quality_scanner
from pyosv.evaluation.synthetic_quality.application import build_report
from pyosv.evaluation.synthetic_quality.cases import CASE_IDS


@cache
def _result() -> SyntheticModeComparisonResult:
    return run_mode_comparison(
        SyntheticModeComparisonConfig(
            case_ids=("single_vertical_plane",),
            shape=(9, 9, 9),
        )
    )


def _config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
    )


def _bundle(path: Path) -> Path:
    return write_artifact_bundle(_result(), path, config=_config())


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


def _mutate_csv(bundle: Path, filename: str, mutation) -> None:
    path = bundle / filename
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    mutation(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rehash(bundle, filename)


def test_nonzero_fraction_matches_legacy_report_summaries_and_metric_rows() -> None:
    epsilon = np.float32(quality_metrics.NONZERO_EPSILON)
    values = np.array(
        [
            np.nextafter(epsilon, np.float32(0.0)),
            np.nextafter(epsilon, np.float32(np.inf)),
            np.nextafter(-epsilon, np.float32(0.0)),
            np.nextafter(-epsilon, np.float32(-np.inf)),
            epsilon,
            -epsilon,
        ],
        dtype=np.float32,
    )
    expected = 2.0 / 6.0
    assert quality_metrics.array_nonzero_fraction(values) == expected
    assert quality_scanner._array_summary(values)["nonzero_fraction"] == expected
    assert quality_pipeline._array_summary(values)["nonzero_fraction"] == expected

    config = _config()
    result = _result()
    legacy_reports = {}
    for scanner_backend in ("reference-like", "quality"):
        for workflow_mode in ("reference", "quality"):
            report = build_report(
                case_set="minimal",
                shape=config.shape,
                voting_config=config.voting_config,
                scanner_config=replace(
                    config.scanner_template,
                    backend=scanner_backend,
                ),
                truth_metric_config=config.truth_metric_config,
                variants=(config.comparison_variant,),
                skinning_config=config.skinning_config,
                input_mode="both",
                workflow_mode=workflow_mode,
                skinner_method_explicit=config.skinner_method_explicit,
                skinner_min_likelihood_explicit=(config.skinner_min_likelihood_explicit),
                skinner_growth_source_explicit=config.skinner_growth_source_explicit,
                skinner_accepted_occupancy_radius_explicit=(
                    config.skinner_accepted_occupancy_radius_explicit
                ),
                skinner_boundary_fallback_explicit=(config.skinner_boundary_fallback_explicit),
            )
            legacy_reports[(scanner_backend, workflow_mode)] = report["cases"][0]

    report_names = {
        "scanner_raw": ("scanner", "ft"),
        "scanner_thinned": ("scanner", "fet"),
        "fv": ("pyosv", "fv"),
        "fvt": ("pyosv", "fvt"),
    }
    rows = [row for row in result.metric_rows if row.metric == "array_nonzero_fraction"]
    assert rows
    for row in rows:
        section, report_name = report_names[row.stage]
        axes = (row.scanner_backend or "reference-like", row.workflow_mode or "reference")
        pipeline_name = "oracle" if row.input_mode == "oracle" else "scanner"
        legacy_pipeline = legacy_reports[axes]["pipelines"][pipeline_name]["variants"][
            config.comparison_variant
        ]
        assert row.value == legacy_pipeline[section][report_name]["nonzero_fraction"]


def test_public_experiment_rejects_empty_truth_support_before_scanner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"case_generation": 0, "scanner_input": 0}
    original_case_factory = comparison_runner._build_trial_case
    original_scanner_input = quality_runner.make_scanner_input_from_case

    def counted_case_factory(*args, **kwargs):
        calls["case_generation"] += 1
        return original_case_factory(*args, **kwargs)

    def counted_scanner_input(*args, **kwargs):
        calls["scanner_input"] += 1
        return original_scanner_input(*args, **kwargs)

    monkeypatch.setattr(comparison_runner, "_build_trial_case", counted_case_factory)
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", counted_scanner_input)

    with pytest.raises(ValueError, match="empty truth-surface support"):
        run_mode_comparison(
            SyntheticModeComparisonConfig(
                case_ids=("single_dipping_plane",),
                shape=(10, 10, 10),
                skinning_config=SyntheticSkinningConfig(enabled=False),
                truth_metric_config=SyntheticTruthMetricConfig(
                    truth_surface_half_width=0.0,
                ),
            )
        )

    assert calls == {"case_generation": 1, "scanner_input": 0}


def test_case_selection_is_canonical_before_execution_and_unique_in_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_generation_calls = 0

    def counted_case_factory(*args, **kwargs):
        nonlocal case_generation_calls
        case_generation_calls += 1
        raise AssertionError("case generation must not run during selection validation")

    monkeypatch.setattr(comparison_runner, "_build_trial_case", counted_case_factory)

    default = SyntheticModeComparisonConfig(shape=(9, 9, 9))
    case_set = SyntheticModeComparisonConfig(case_set="geometry", shape=(9, 9, 9))
    explicit_ids = (CASE_IDS[2], CASE_IDS[0])
    explicit = SyntheticModeComparisonConfig(case_ids=explicit_ids, shape=(9, 9, 9))

    assert (default.case_set, default.case_ids) == ("minimal", None)
    assert build_mode_comparison_plan(case_set).case_ids == CASE_IDS[:3]
    assert (case_set.case_set, case_set.case_ids) == ("geometry", None)
    assert build_mode_comparison_plan(explicit).case_ids == explicit_ids
    assert (explicit.case_set, explicit.case_ids) == (None, explicit_ids)

    invalid = (
        {"case_set": "minimal", "case_ids": (CASE_IDS[0],)},
        {"case_set": "missing"},
        {"case_ids": ("missing",)},
    )
    for arguments in invalid:
        with pytest.raises(ValueError):
            SyntheticModeComparisonConfig(shape=(9, 9, 9), **arguments)
    assert case_generation_calls == 0
    monkeypatch.undo()

    default_result = run_mode_comparison(default)
    default_bundle = write_artifact_bundle(
        default_result,
        tmp_path / "default",
        config=default,
    )
    explicit_bundle = write_artifact_bundle(
        _result(),
        tmp_path / "explicit",
        config=_config(),
    )
    default_input = json.loads((default_bundle / "manifest.json").read_text(encoding="utf-8"))[
        "input_config"
    ]
    explicit_input = json.loads((explicit_bundle / "manifest.json").read_text(encoding="utf-8"))[
        "input_config"
    ]
    assert (default_input["case_set"], default_input["case_ids"]) == ("minimal", None)
    assert (explicit_input["case_set"], explicit_input["case_ids"]) == (
        None,
        ["single_vertical_plane"],
    )


@pytest.mark.parametrize(
    ("filename", "mutation"),
    (
        (
            "metrics_long.csv",
            lambda rows: rows.__setitem__(
                slice(None), [row for row in rows if row["cell_label"] != "Q-SCAN"]
            ),
        ),
        (
            "contrasts.csv",
            lambda rows: rows[0].__setitem__("raw_value", str(float(rows[0]["raw_value"]) + 0.25)),
        ),
        (
            "metric_aggregates.csv",
            lambda rows: rows[0].__setitem__("mean", str(float(rows[0]["mean"]) + 0.25)),
        ),
        (
            "contrast_aggregates.csv",
            lambda rows: rows[0].__setitem__("mean", str(float(rows[0]["mean"]) + 0.25)),
        ),
        (
            "runtime.csv",
            lambda rows: rows[0].__setitem__("seed", "7"),
        ),
    ),
)
def test_rehashed_csv_semantic_tampering_is_rejected(
    tmp_path: Path,
    filename: str,
    mutation,
) -> None:
    bundle = _bundle(tmp_path / filename.removesuffix(".csv"))
    _mutate_csv(bundle, filename, mutation)

    with pytest.raises(ValueError):
        validate_completed_bundle(bundle)


def test_rehashed_cell_report_metric_evidence_tampering_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "cell-report")
    path = bundle / "cell_reports.json"
    reports = json.loads(path.read_text(encoding="utf-8"))
    summary = reports[0]["cells"]["RL-SCAN"]["scanner"]["ft"]
    summary["nonzero_fraction"] = (summary["nonzero_fraction"] + 0.25) % 1.0
    path.write_text(
        json.dumps(reports, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError, match="cell_reports"):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize("tamper", ("cache_counter", "trial_seed"))
def test_rehashed_manifest_semantic_tampering_is_rejected(
    tmp_path: Path,
    tamper: str,
) -> None:
    bundle = _bundle(tmp_path / tamper)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "cache_counter":
        manifest["cache_stats"][0]["seed_hits"] += 1
    else:
        manifest["trials"][0]["case_generation_seed"] = 7
    path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(bundle, "manifest.json")

    with pytest.raises(ValueError):
        validate_completed_bundle(bundle)

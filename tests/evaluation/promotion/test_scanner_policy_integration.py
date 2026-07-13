from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyosv.evaluation.promotion.comparison import compare_reports
from pyosv.evaluation.promotion.gates import add_required_coverage, build_promotion_report
from pyosv.evaluation.promotion.markdown import comparison_markdown, promotion_markdown
from pyosv.evaluation.reporting import write_summary_csv


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_quality_reports.py"
METRICS_FIXTURE = REPO_ROOT / (
    "tests/fixtures/synthetic_quality_refactor/17_quality_ref2_metrics.json"
)


def _metrics(scanner_thin_mode: str) -> dict[str, object]:
    report = json.loads(METRICS_FIXTURE.read_text(encoding="utf-8"))
    report["config"]["shape"] = [49, 49, 49]
    for case in report["cases"]:
        case["shape"] = [49, 49, 49]
        for variant in case["pipelines"]["scanner"]["variants"].values():
            quality = variant["quality"]
            fvt_count = quality["fvt_positive_top_truth_count"]["buffered_overlap_radius2"][
                "candidate_count"
            ]
            quality["skin"]["buffered_overlap_radius2"]["buffered_f1"] = 0.95
            quality["skin"]["topology"]["cell_count"] = fvt_count
    _replace_scanner_mode(report, scanner_thin_mode)
    return report


def _replace_scanner_mode(value: object, scanner_thin_mode: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "scanner_thin_mode":
                value[key] = scanner_thin_mode
            else:
                _replace_scanner_mode(child, scanner_thin_mode)
    elif isinstance(value, list):
        for child in value:
            _replace_scanner_mode(child, scanner_thin_mode)


def _write_metrics(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def _policy_inputs(
    tmp_path: Path,
    *,
    baseline_mode: str = "reference",
    candidate_mode: str = "normal",
) -> tuple[Path, Path, Path, Path, dict[str, object], dict[str, object]]:
    baseline_report = _metrics(baseline_mode)
    candidate_report = _metrics(candidate_mode)
    baseline_summary = write_summary_csv(baseline_report, tmp_path / "baseline")
    candidate_summary = write_summary_csv(candidate_report, tmp_path / "candidate")
    baseline_metrics = baseline_summary.with_name("metrics.json")
    candidate_metrics = candidate_summary.with_name("metrics.json")
    _write_metrics(baseline_metrics, baseline_report)
    _write_metrics(candidate_metrics, candidate_report)
    return (
        baseline_summary,
        candidate_summary,
        baseline_metrics,
        candidate_metrics,
        baseline_report,
        candidate_report,
    )


def _compare(
    paths: tuple[Path, Path, Path, Path, dict[str, object], dict[str, object]],
    *,
    baseline_variant: str = "current_default",
    candidate_variant: str = "current_default",
) -> dict[str, object]:
    baseline_summary, candidate_summary, baseline_metrics, candidate_metrics, _, _ = paths
    return compare_reports(
        baseline_summary,
        candidate_summary,
        baseline_variant,
        candidate_variant,
        promotion_gate="scanner-boundary",
        strict_missing_rows=True,
        comparison_profile="scanner-thinning-policy-v1",
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )


def _csv_contents(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    return rows[0], rows[1:]


def _write_csv_contents(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_valid_scanner_policy_comparison_matches_rows_and_passes_gate(tmp_path: Path) -> None:
    report = _compare(_policy_inputs(tmp_path))

    contract = report["scanner_policy_contract"]
    assert report["row_count"] == 14
    assert report["missing_baseline_rows"] == []
    assert report["missing_candidate_rows"] == []
    assert contract["passed"] is True
    assert contract["baseline"] == {
        "policy_id": "quality_scanner_reference_v1",
        "scanner_thin_mode": "reference",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": True,
    }
    assert contract["candidate"] == {
        "policy_id": "quality_scanner_thin_normal_v1",
        "scanner_thin_mode": "normal",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": None,
    }
    assert contract["observed_config_differences"] == [
        {
            "path": "config.scanner.scanner_thin_mode",
            "baseline": "reference",
            "candidate": "normal",
            "allowed": True,
        }
    ]
    assert report["promotion_gate"]["coverage"]["passed"] is True
    assert report["promotion_gate"]["passed"] is True


def test_scanner_policy_comparison_requires_both_metrics_paths(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="requires a baseline metrics path"):
        compare_reports(
            paths[0],
            paths[1],
            "current_default",
            "current_default",
            comparison_profile="scanner-thinning-policy-v1",
        )
    with pytest.raises(ValueError, match="requires a candidate metrics path"):
        compare_reports(
            paths[0],
            paths[1],
            "current_default",
            "current_default",
            comparison_profile="scanner-thinning-policy-v1",
            baseline_metrics=paths[2],
        )


def test_summary_metrics_pair_mismatch_is_rejected_before_comparison(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"baseline summary CSV .* does not match baseline metrics JSON .*missing canonical",
    ):
        compare_reports(
            paths[1],
            paths[1],
            "current_default",
            "current_default",
            comparison_profile="scanner-thinning-policy-v1",
            baseline_metrics=paths[2],
            candidate_metrics=paths[3],
        )
    with pytest.raises(ValueError, match=r"baseline summary CSV .*baseline metrics JSON"):
        compare_reports(
            paths[0],
            paths[1],
            "current_default",
            "current_default",
            comparison_profile="scanner-thinning-policy-v1",
            baseline_metrics=paths[3],
            candidate_metrics=paths[3],
        )


@pytest.mark.parametrize("column", ("scanner_thin_mode", "skin_buffered_f1_r2"))
def test_summary_cell_mutation_is_rejected(tmp_path: Path, column: str) -> None:
    paths = _policy_inputs(tmp_path)
    header, rows = _csv_contents(paths[1])
    index = header.index(column)
    rows[0][index] = "tampered"
    _write_csv_contents(paths[1], header, rows)

    with pytest.raises(
        ValueError,
        match=r"candidate summary CSV .*missing canonical rows=1, unexpected rows=1",
    ):
        _compare(paths)


def test_summary_data_row_order_is_ignored(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    header, rows = _csv_contents(paths[1])
    _write_csv_contents(paths[1], header, list(reversed(rows)))

    assert _compare(paths)["scanner_policy_contract"]["passed"] is True


@pytest.mark.parametrize(
    ("mutation", "missing", "unexpected"),
    (("delete", 1, 0), ("duplicate", 0, 1)),
)
def test_summary_row_multiplicity_is_enforced(
    tmp_path: Path,
    mutation: str,
    missing: int,
    unexpected: int,
) -> None:
    paths = _policy_inputs(tmp_path)
    header, rows = _csv_contents(paths[1])
    if mutation == "delete":
        rows.pop()
    else:
        rows.append(rows[0])
    _write_csv_contents(paths[1], header, rows)

    with pytest.raises(
        ValueError,
        match=rf"missing canonical rows={missing}, unexpected rows={unexpected}",
    ):
        _compare(paths)


@pytest.mark.parametrize("mutation", ("delete", "reorder"))
def test_summary_header_must_match_schema(tmp_path: Path, mutation: str) -> None:
    paths = _policy_inputs(tmp_path)
    header, rows = _csv_contents(paths[1])
    if mutation == "delete":
        header.pop()
    else:
        header[0], header[1] = header[1], header[0]
    _write_csv_contents(paths[1], header, rows)

    with pytest.raises(ValueError, match=r"candidate summary CSV .*header does not match"):
        _compare(paths)


def test_summary_row_width_is_validated(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    header, rows = _csv_contents(paths[1])
    rows[0].pop()
    _write_csv_contents(paths[1], header, rows)

    with pytest.raises(ValueError, match=r"candidate summary CSV .*row 2 has .* columns"):
        _compare(paths)


def test_incomplete_metrics_has_canonical_summary_error(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    candidate = copy.deepcopy(paths[5])
    candidate.pop("cases")
    _write_metrics(paths[3], candidate)

    with pytest.raises(
        ValueError,
        match=r"candidate metrics JSON .*cannot produce canonical summary.csv",
    ):
        _compare(paths)


def test_selected_variant_must_be_declared_by_metrics_config(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    candidate = copy.deepcopy(paths[5])
    candidate["config"]["variants"] = ["boundary_aware_voter_v1"]
    _write_metrics(paths[3], candidate)

    report = _compare(paths)

    assert report["scanner_policy_contract"]["passed"] is False
    assert report["promotion_gate"]["passed"] is False
    assert any(
        "candidate selected variant 'current_default' is not present in config.variants" in reason
        for reason in report["scanner_policy_contract"]["reasons"]
    )


@pytest.mark.parametrize("variants", ("current_default", [""], [1]))
def test_metrics_config_variants_must_be_nonempty_string_array(
    tmp_path: Path, variants: object
) -> None:
    paths = _policy_inputs(tmp_path)
    candidate = copy.deepcopy(paths[5])
    candidate["config"]["variants"] = variants
    _write_metrics(paths[3], candidate)

    with pytest.raises(ValueError, match=r"candidate metrics JSON .*config.variants"):
        _compare(paths)


def test_variant_profile_does_not_read_optional_metrics_paths(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    report = compare_reports(
        paths[0],
        paths[1],
        "current_default",
        "current_default",
        baseline_metrics=tmp_path / "missing-baseline.json",
        candidate_metrics=tmp_path / "missing-candidate.json",
    )

    assert "comparison_profile" not in report["config"]
    assert "scanner_policy_contract" not in report


@pytest.mark.parametrize(
    ("mutation", "baseline_variant", "candidate_variant", "reason_fragment"),
    (
        ("candidate_reference", "current_default", "current_default", "scanner_thin_mode"),
        ("baseline_normal", "current_default", "current_default", "scanner_thin_mode"),
        ("candidate_backend", "current_default", "current_default", "scanner.backend"),
        ("candidate_refinement", "current_default", "current_default", "refinement_factor"),
        ("candidate_sigma", "current_default", "current_default", "scanner.sigma1"),
        ("candidate_seed", "current_default", "current_default", "scanner.input.seed"),
        ("candidate_voting", "current_default", "current_default", "voting.voter_thin_mode"),
        ("candidate_skinning", "current_default", "current_default", "skinning.method"),
        ("candidate_edges", "current_default", "current_default", "remove_edge_effects"),
        ("variant_mismatch", "current_default", "other", "selected variants differ"),
        ("non_default", "other", "other", "selected variant must be 'current_default'"),
    ),
)
def test_contract_failure_forces_numeric_gate_failure(
    tmp_path: Path,
    mutation: str,
    baseline_variant: str,
    candidate_variant: str,
    reason_fragment: str,
) -> None:
    paths = _policy_inputs(tmp_path)
    baseline_report = copy.deepcopy(paths[4])
    candidate_report = copy.deepcopy(paths[5])
    baseline_scanner = baseline_report["config"]["scanner"]
    candidate_config = candidate_report["config"]
    candidate_scanner = candidate_config["scanner"]
    if mutation == "candidate_reference":
        candidate_scanner["scanner_thin_mode"] = "reference"
    elif mutation == "baseline_normal":
        baseline_scanner["scanner_thin_mode"] = "normal"
    elif mutation == "candidate_backend":
        candidate_scanner["backend"] = "fast"
    elif mutation == "candidate_refinement":
        candidate_scanner["refinement_factor"] = 3
    elif mutation == "candidate_sigma":
        candidate_scanner["sigma1"] = 3.0
    elif mutation == "candidate_seed":
        candidate_scanner["input"]["seed"] = 7
    elif mutation == "candidate_voting":
        candidate_config["voting"]["voter_thin_mode"] = "reference"
    elif mutation == "candidate_skinning":
        candidate_config["skinning"]["method"] = "reference"
    elif mutation == "candidate_edges":
        candidate_scanner["remove_edge_effects"] = False
    _write_metrics(paths[2], baseline_report)
    _write_metrics(paths[3], candidate_report)

    report = _compare(
        paths,
        baseline_variant=baseline_variant,
        candidate_variant=candidate_variant,
    )

    assert report["scanner_policy_contract"]["passed"] is False
    assert report["promotion_gate"]["passed"] is False
    assert any(reason_fragment in reason for reason in report["scanner_policy_contract"]["reasons"])
    assert any(
        reason.startswith("scanner policy contract:")
        for reason in report["promotion_gate"]["reasons"]
    )


def test_contract_failure_makes_fail_on_gate_failure_exit_two(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path, candidate_mode="reference")
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(paths[0]),
            str(paths[1]),
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--baseline-metrics",
            str(paths[2]),
            "--candidate-metrics",
            str(paths[3]),
            "--promotion-gate",
            "scanner-boundary",
            "--fail-on-gate-failure",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_aggregate_contract_failure_makes_fail_on_gate_failure_exit_two(
    tmp_path: Path,
) -> None:
    paths = _policy_inputs(tmp_path, candidate_mode="reference")
    gate_script = REPO_ROOT / "scripts" / "check_synthetic_quality_promotion_gate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(gate_script),
            "--baseline-summary",
            str(paths[0]),
            "--candidate-summary",
            str(paths[1]),
            "--candidate-variant",
            "current_default",
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--baseline-metrics",
            str(paths[2]),
            "--candidate-metrics",
            str(paths[3]),
            "--fail-on-gate-failure",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_policy_contract_is_preserved_by_coverage_and_aggregate_report(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    comparison = add_required_coverage(_compare(paths))
    aggregate = build_promotion_report(
        paths[0],
        paths[1],
        candidate_variants=("current_default",),
        comparison_profile="scanner-thinning-policy-v1",
        baseline_metrics=paths[2],
        candidate_metrics=paths[3],
    )

    assert comparison["promotion_gate"]["coverage"]["passed"] is True
    assert comparison["promotion_gate"]["passed"] is True
    assert aggregate["scanner_policy_contract"]["passed"] is True
    assert aggregate["promotion_gate"]["passed"] is True


def test_new_profile_markdown_shows_policy_semantics_and_sorted_differences(
    tmp_path: Path,
) -> None:
    paths = _policy_inputs(tmp_path)
    valid_text = comparison_markdown(_compare(paths))
    assert "Contract result: pass" in valid_text
    assert "quality_scanner_reference_v1" in valid_text
    assert "quality_scanner_thin_normal_v1" in valid_text
    assert "requested=true, effective=true" in valid_text
    assert "requested=true, effective=null" in valid_text

    candidate_report = copy.deepcopy(paths[5])
    candidate_report["config"]["voting"]["voter_thin_mode"] = "reference"
    candidate_report["config"]["scanner"]["sigma1"] = 3.0
    _write_metrics(paths[3], candidate_report)
    comparison = _compare(paths)
    aggregate = build_promotion_report(
        paths[0],
        paths[1],
        candidate_variants=("current_default",),
        comparison_profile="scanner-thinning-policy-v1",
        baseline_metrics=paths[2],
        candidate_metrics=paths[3],
    )

    comparison_text = comparison_markdown(comparison)
    promotion_text = promotion_markdown(aggregate)
    for text in (comparison_text, promotion_text):
        assert "## Scanner Policy Contract" in text
        assert "Contract result: fail" in text
        assert "quality_scanner_reference_v1" in text
        assert "quality_scanner_thin_normal_v1" not in text
        assert "requested=true, effective=true" in text
        assert "requested=true, effective=null" in text
        assert text.index("config.scanner.sigma1") < text.index("config.voting.voter_thin_mode")
        assert text.endswith("\n")

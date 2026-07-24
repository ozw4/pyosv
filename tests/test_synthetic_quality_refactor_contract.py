from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_synthetic_quality_refactor_contract.py"
)
SPEC = importlib.util.spec_from_file_location(
    "check_synthetic_quality_refactor_contract", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def _write_output(output_dir: Path, value: int = 1) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps({"nested": {"value": value}}), encoding="utf-8"
    )
    (output_dir / "summary.csv").write_bytes(b"case,value\r\nexample,1\r\n")
    artifact = output_dir / "case" / "scanner" / "variant" / "fvt.dat"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"volume")
    (artifact.parent / "skins.json").write_text('{"skins": []}', encoding="utf-8")


def test_compare_output_accepts_matching_fixture(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    fixture_dir = tmp_path / "fixtures"
    output_dir.mkdir()
    _write_output(output_dir)
    contract.update_fixtures(output_dir, fixture_dir)

    assert contract.compare_output(output_dir, fixture_dir) == []


def test_compare_output_reports_json_csv_and_artifact_differences(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    fixture_dir = tmp_path / "fixtures"
    output_dir.mkdir()
    _write_output(output_dir)
    contract.update_fixtures(output_dir, fixture_dir)

    _write_output(output_dir, value=2)
    (output_dir / "summary.csv").write_bytes(b"case,value\r\nexample,2\r\n")
    artifact = output_dir / "case" / "scanner" / "variant" / "fvt.dat"
    artifact.write_bytes(b"changed")
    extra = output_dir / "extra.dat"
    extra.write_bytes(b"extra")

    differences = contract.compare_output(output_dir, fixture_dir)

    assert any("JSON $.nested.value" in difference for difference in differences)
    assert any("CSV row 2, column 2" in difference for difference in differences)
    assert any(
        "artifact case/scanner/variant/fvt.dat size" in difference for difference in differences
    )
    assert any(
        "artifact case/scanner/variant/fvt.dat sha256" in difference for difference in differences
    )
    assert any("artifact extra.dat: unexpected" in difference for difference in differences)


def test_json_contract_allows_only_additive_buffered_overlap_counts() -> None:
    expected = {"buffered_overlap_radius2": {"candidate_count": 3}}
    actual = {
        "buffered_overlap_radius2": {
            "candidate_count": 3,
            "candidate_in_truth_buffer_count": 2,
            "truth_in_candidate_buffer_count": 1,
        }
    }

    assert contract._json_differences(expected, actual) == []
    actual["buffered_overlap_radius2"]["unexpected_count"] = 1
    assert contract._json_differences(expected, actual) == [
        "JSON $.buffered_overlap_radius2.unexpected_count: unexpected in actual output"
    ]


def test_json_contract_allows_only_additive_component_incidence_fields() -> None:
    expected = {
        "component_topology": {
            "truth_components": [{"truth_id": 1}],
            "skins": [{"skin_index": 0}],
        }
    }
    actual = {
        "component_topology": {
            "qualification_min_fraction": 0.05,
            "truth_components": [
                {
                    "truth_id": 1,
                    "skin_cell_counts": [{"skin_index": 0, "covered_cell_count": 2}],
                    "qualifying_skin_count": 1,
                }
            ],
            "skins": [
                {
                    "skin_index": 0,
                    "truth_component_cell_counts": [{"truth_id": 1, "cell_count": 2}],
                    "qualifying_truth_component_count": 1,
                }
            ],
        }
    }

    assert contract._json_differences(expected, actual) == []
    actual["component_topology"]["unexpected_summary"] = 1
    assert contract._json_differences(expected, actual) == [
        "JSON $.component_topology.unexpected_summary: unexpected in actual output"
    ]


@pytest.mark.parametrize(
    ("path", "key"),
    (
        (
            "$.component_topology.truth_components[0].skin_cell_counts[0]",
            "qualifying_skin_count",
        ),
        (
            "$.component_topology.skins[0].truth_component_cell_counts[0]",
            "qualifying_truth_component_count",
        ),
    ),
)
def test_component_incidence_additions_are_limited_to_array_items(path: str, key: str) -> None:
    assert not contract._is_additive_component_topology_field(path, key)


def test_artifact_manifest_does_not_follow_symlinks(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    external_artifact = tmp_path / "external.dat"
    external_artifact.write_bytes(b"external volume")
    (output_dir / "linked.dat").symlink_to(external_artifact)

    assert contract.artifact_manifest(output_dir) == {}


def test_update_requires_flag_and_environment_variable(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_output(output_dir)
    monkeypatch.delenv(contract.UPDATE_ENVIRONMENT_VARIABLE, raising=False)

    assert contract.main(["--existing-output", str(output_dir), "--update-fixtures"]) == 2

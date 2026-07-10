from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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


def test_update_requires_flag_and_environment_variable(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_output(output_dir)
    monkeypatch.delenv(contract.UPDATE_ENVIRONMENT_VARIABLE, raising=False)

    assert contract.main(["--existing-output", str(output_dir), "--update-fixtures"]) == 2

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tarfile
import tomllib
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = REPOSITORY_ROOT / "public_release" / "template"
ARCHIVE_SHA256 = "872fa183e1016b70ccd41449e689b545adb49e70cb29e563bf6f35133834c13d"
PUBLICATION_ID = "c20a3a4195fb5598a9661d16cf368610ba7081c28ece2e63549706fec6a35322"
SOURCE_COMPLETION_SHA256 = "3cc8818b27c9ea68d7fc4f5c9fc8d072aaaeb81cfd672c1c79d54c9fe8c1ae72"
DATASET_ID = "f3d-official-v1"
LICENSE_SHA256 = "9698f2ba346a875a47a9cc6bb602e3d126758774dde1f575b4f3369dc3f2574f"


@pytest.fixture
def bundle_identity() -> dict[str, object]:
    return {
        "schema": "pyosv.f3_compact_publication_manifest.v1",
        "publication_id": PUBLICATION_ID,
        "source_completion_sha256": SOURCE_COMPLETION_SHA256,
        "dataset_id": DATASET_ID,
        "shape": [420, 400, 100],
        "figure": 6,
        "figure_data": 6,
    }


def _template_text_files() -> list[Path]:
    suffixes = {".cff", ".md", ".toml"}
    return sorted(path for path in TEMPLATE_ROOT.rglob("*") if path.suffix in suffixes)


def _member_bytes(archive: tarfile.TarFile, relative_name: str) -> bytes:
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile()
        and (member.name == relative_name or member.name.endswith(f"/{relative_name}"))
    ]
    assert len(matches) == 1
    stream = archive.extractfile(matches[0])
    assert stream is not None
    return stream.read()


def test_template_layout_and_license() -> None:
    expected = {
        "CITATION.cff",
        "DATA_ATTRIBUTION.md",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/manual.md",
        "docs/public_reference_comparison.md",
        "docs/reproducibility.md",
        "examples/run_qqual3d.py",
        "examples/validate_compact_publication.py",
        "pyproject.toml",
    }
    actual = {
        path.relative_to(TEMPLATE_ROOT).as_posix()
        for path in TEMPLATE_ROOT.rglob("*")
        if path.is_file()
    }
    assert actual == expected

    license_bytes = (TEMPLATE_ROOT / "LICENSE").read_bytes()
    assert license_bytes.splitlines()[0] == b"Common Public License Version 1.0"
    assert hashlib.sha256(license_bytes).hexdigest() == LICENSE_SHA256
    assert all(
        "mit license" not in path.read_text(encoding="utf-8").lower()
        for path in _template_text_files()
    )


def test_approved_license_matches_when_configured() -> None:
    configured = os.environ.get("APPROVED_LICENSE_FILE")
    if configured is None:
        pytest.skip("APPROVED_LICENSE_FILE is not configured")
    approved = Path(configured)
    assert approved.is_file()
    assert not approved.is_symlink()
    assert approved.read_bytes() == (TEMPLATE_ROOT / "LICENSE").read_bytes()


def test_third_party_and_data_attribution() -> None:
    notice = (TEMPLATE_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for expected in (
        "xinwucwp/osv",
        "f4e2564fc27b9539edc4caff0944b1ddb94997b8",
        "CPL-1.0",
        "Xinming Wu",
        "Colorado School of Mines and others",
        "Dave Hale",
    ):
        assert expected in notice

    attribution = (TEMPLATE_ROOT / "DATA_ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "does not redistribute the raw F3 DAT volumes" in attribution
    assert "not geological truth" in attribution
    assert "Creative Commons subtype" in attribution


def test_public_metadata_and_citation() -> None:
    project = tomllib.loads((TEMPLATE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    metadata = project["project"]
    assert metadata["name"] == "pyosv-qqual-poc"
    assert metadata["version"] == "0.1.0"
    assert metadata["license"] == {"file": "LICENSE"}
    assert "License :: OSI Approved :: Common Public License" in metadata["classifiers"]
    assert metadata["scripts"] == {
        "pyosv-qqual3d": "pyosv.cli.qqual3d:main",
        "pyosv-validate-compact": "pyosv.cli.validate_compact_publication:main",
    }

    citation = (TEMPLATE_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for expected in (
        "cff-version: 1.2.0",
        "title: PyOSV Q-QUAL PoC",
        "version: 0.1.0",
        "license: CPL-1.0",
        "type: software",
        "family-names: Ozawa",
        "given-names: Mitsuyuki",
    ):
        assert expected in citation
    assert "orcid:" not in citation.lower()
    assert "repository-code:" not in citation.lower()


def test_bundle_identity_is_documented(bundle_identity: dict[str, object]) -> None:
    comparison = (TEMPLATE_ROOT / "docs/public_reference_comparison.md").read_text(encoding="utf-8")
    reproducibility = (TEMPLATE_ROOT / "docs/reproducibility.md").read_text(encoding="utf-8")
    for value in (
        ARCHIVE_SHA256,
        str(bundle_identity["publication_id"]),
        str(bundle_identity["source_completion_sha256"]),
        str(bundle_identity["dataset_id"]),
    ):
        assert value in comparison
    for value in (ARCHIVE_SHA256, PUBLICATION_ID, SOURCE_COMPLETION_SHA256):
        assert value in reproducibility
    assert "6 figures and 6 figure-data tables" in comparison


def test_document_contract_and_relative_links() -> None:
    required_headings = {
        "README.md": ("## Installation", "## Five-minute quick start", "## Limitations"),
        "docs/manual.md": (
            "## Environment setup",
            "## Input DAT contract",
            "## Q-QUAL CLI",
            "## Output layout",
            "## run.json",
            "## Compact bundle validation",
            "## Troubleshooting",
        ),
        "docs/public_reference_comparison.md": (
            "## Evidence identity",
            "## Stage mapping",
            "## Selected sections",
            "## Summary table",
            "## Visualization contract",
            "## Interpretation boundary",
        ),
        "docs/reproducibility.md": (
            "## Publicly reproducible",
            "## Not included",
            "## Provenance",
        ),
    }
    for relative, headings in required_headings.items():
        text = (TEMPLATE_ROOT / relative).read_text(encoding="utf-8")
        for heading in headings:
            assert heading in text

    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for document in TEMPLATE_ROOT.rglob("*.md"):
        for target in link_pattern.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            relative_target = target.split("#", 1)[0]
            assert (document.parent / relative_target).is_file(), (document, target)

    combined = "\n".join(path.read_text(encoding="utf-8") for path in _template_text_files())
    lowered = combined.lower()
    assert "/workspace/" not in combined
    assert "/home/dcuser/" not in combined
    assert re.search(r"\b(?:task|issue|pr)\s*#?\d+\b", combined, re.IGNORECASE) is None
    assert "migration history" not in lowered
    assert "public reference is geological truth" not in lowered
    assert "reference_osv" not in combined
    assert "fv.dat` is the voted likelihood" in combined
    assert "fvt.dat` is the voter-thinned ridge volume" in combined


def test_examples_are_thin_cli_wrappers() -> None:
    run_example = (TEMPLATE_ROOT / "examples/run_qqual3d.py").read_text(encoding="utf-8")
    validate_example = (TEMPLATE_ROOT / "examples/validate_compact_publication.py").read_text(
        encoding="utf-8"
    )
    assert "from pyosv.cli.qqual3d import main" in run_example
    assert "from pyosv.cli.validate_compact_publication import main" in validate_example
    assert "raise SystemExit(main())" in run_example
    assert "raise SystemExit(main())" in validate_example


def test_formal_archive_matches_documented_identity() -> None:
    configured_archive = os.environ.get("COMPACT_BUNDLE_ARCHIVE")
    configured_checksum = os.environ.get("COMPACT_BUNDLE_SHA256_FILE")
    if configured_archive is None or configured_checksum is None:
        pytest.skip("formal compact archive is not configured")

    archive_path = Path(configured_archive)
    checksum_path = Path(configured_checksum)
    assert archive_path.is_file() and not archive_path.is_symlink()
    assert checksum_path.is_file() and not checksum_path.is_symlink()
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == ARCHIVE_SHA256
    assert checksum_path.read_text(encoding="utf-8").split()[0] == ARCHIVE_SHA256

    with tarfile.open(archive_path, mode="r:gz") as archive:
        manifest = json.loads(_member_bytes(archive, "publication_manifest.json"))
        experiment = json.loads(_member_bytes(archive, "experiment.json"))
        summary_rows = list(
            csv.DictReader(
                io.StringIO(
                    _member_bytes(archive, "f3_q_qual_vs_public_ref_summary.csv").decode("utf-8")
                )
            )
        )

    roles: dict[str, int] = {}
    for artifact in manifest["artifacts"]:
        role = artifact["role"]
        roles[role] = roles.get(role, 0) + 1
    assert manifest["schema"] == "pyosv.f3_compact_publication_manifest.v1"
    assert manifest["publication_id"] == PUBLICATION_ID
    assert manifest["source"]["f3_completion_sha256"] == SOURCE_COMPLETION_SHA256
    assert manifest["dataset"]["dataset_id"] == DATASET_ID
    assert manifest["dataset"]["shape"] == [420, 400, 100]
    assert manifest["code"]["git_commit"] == "47f81b72a7bfab3ce259b821548ad8e6156e74cb"
    assert roles["figure"] == 6
    assert roles["figure_data"] == 6
    assert experiment["source"]["f3_completion_sha256"] == SOURCE_COMPLETION_SHA256

    comparison = (TEMPLATE_ROOT / "docs/public_reference_comparison.md").read_text(encoding="utf-8")
    normalized_comparison = " ".join(comparison.split())
    sections = experiment["sections"]
    assert sections["selection_policy"] in comparison
    for section in sections["items"]:
        expected = (
            f"`{section['axis']}={section['index']}` "
            f"(bin {section['bin_index']}, score {section['ridge_count_score']})"
        )
        assert expected in normalized_comparison

    visualization = experiment["visualization"]
    assert f"`{visualization['amplitude_filename']}`" in comparison
    assert f"`{visualization['attribute_colormap']}`" in comparison
    assert f"`{visualization['difference_colormap']}`" in comparison
    assert "one-pixel red cross halo" in comparison
    assert "alpha `0.5`" in comparison
    for row in summary_rows:
        for key, value in row.items():
            if key != "q_qual_stage_fingerprint":
                assert value in comparison

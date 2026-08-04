from __future__ import annotations

from pathlib import Path
from typing import Any

from pyosv.cli import mode_comparison_publication

pytest_plugins = ("tests.evaluation.mode_comparison_publication.conftest",)


def test_validate_only_ignores_missing_sources(
    publication_bundle: tuple[Path, dict[str, Any]],
    monkeypatch,
) -> None:
    output, _sources = publication_bundle
    monkeypatch.setattr(
        mode_comparison_publication,
        "generate_publication_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("validate-only attempted generation")
        ),
    )
    assert mode_comparison_publication.main(["--validate-only", "--output-dir", str(output)]) == 0


def test_normal_generation_requires_all_source_arguments(tmp_path: Path) -> None:
    assert mode_comparison_publication.main(["--output-dir", str(tmp_path / "out")]) == 1


def test_cli_generation_accepts_pretty_and_fixed_sources(
    source_bundles: dict[str, Any],
    tmp_path: Path,
) -> None:
    output = tmp_path / "publication"
    assert (
        mode_comparison_publication.main(
            [
                "--pretty",
                "--synthetic-bundle",
                str(source_bundles["synthetic"]),
                "--f3-bundle",
                str(source_bundles["f3"]),
                "--f3-data-root",
                str(source_bundles["data_root"]),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "completion.json").is_file()

from __future__ import annotations

from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies(pyproject_text: str) -> list[str]:
    project_section = pyproject_text.split("[project]", maxsplit=1)[1].split(
        "[project.optional-dependencies]", maxsplit=1
    )[0]
    match = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)^\s*\]", project_section)
    assert match is not None
    return re.findall(r'^\s*"([^"]+)"\s*,?\s*$', match.group(1), flags=re.MULTILINE)


def _assert_exact_numpy_one_constraint(dependencies: list[str]) -> None:
    numpy_dependencies = [
        dependency for dependency in dependencies if dependency.lower().startswith("numpy")
    ]
    assert numpy_dependencies == ["numpy<2"]


def test_authoritative_project_dependency_requires_numpy_less_than_two() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = _project_dependencies(pyproject_text)

    _assert_exact_numpy_one_constraint(dependencies)
    assert 'requires-python = ">=3.10"' in pyproject_text
    assert not any("atlas" in dependency.lower() for dependency in dependencies)


def test_devcontainer_requirement_requires_numpy_less_than_two() -> None:
    requirements = (REPO_ROOT / ".devcontainer" / "requirements-dev.txt").read_text(
        encoding="utf-8"
    )
    dependencies = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    _assert_exact_numpy_one_constraint(dependencies)


@pytest.mark.parametrize(
    "dependencies",
    [
        ["numpy"],
        ["numpy>=2"],
        ["numpy<3"],
        ["numpy<2", "numpy>=1"],
        ["scipy"],
    ],
)
def test_numpy_dependency_guard_rejects_non_authoritative_constraints(
    dependencies: list[str],
) -> None:
    with pytest.raises(AssertionError):
        _assert_exact_numpy_one_constraint(dependencies)


def test_documentation_declares_numpy_one_and_contract_only_status() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    contract = (REPO_ROOT / "docs" / "fault_warping.md").read_text(encoding="utf-8")

    assert "NumPy 1.x" in readme
    assert "`numpy<2`" in readme
    assert "docs/fault_warping.md" in readme
    assert "contract only" in readme.lower()
    for text in (
        "pyosv.fault_warping.v1",
        "volume[i3, i2, i1]",
        "positive_side(t) ≈ negative_side(t + tau)",
        "p2 = dx1 / dx2",
        "p3 = dx1 / dx3",
        "valid_mask",
    ):
        assert text in contract

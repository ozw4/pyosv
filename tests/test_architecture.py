from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_quality_example_is_cli_only() -> None:
    source = (REPO_ROOT / "examples" / "report_3d_synthetic_quality.py").read_text(encoding="utf-8")

    assert "from pyosv.cli.synthetic_quality import main" in source
    for implementation_name in ("numpy", "scanner", "voter", "skinner"):
        assert implementation_name not in source.lower()


def test_variant_registry_has_one_source_of_truth() -> None:
    definitions = []
    for path in (REPO_ROOT / "src" / "pyosv").rglob("*.py"):
        if "VARIANT_REGISTRY:" in path.read_text(encoding="utf-8"):
            definitions.append(path.relative_to(REPO_ROOT).as_posix())

    assert definitions == ["src/pyosv/evaluation/synthetic_quality/variants.py"]


def test_promotion_thresholds_have_one_source_of_truth() -> None:
    definitions = []
    for path in (REPO_ROOT / "src" / "pyosv").rglob("*.py"):
        if "MATERIAL_REGRESSION_THRESHOLDS =" in path.read_text(encoding="utf-8"):
            definitions.append(path.relative_to(REPO_ROOT).as_posix())

    assert definitions == ["src/pyosv/evaluation/promotion/specifications.py"]


def test_compatibility_facade_imports_succeed() -> None:
    import pyosv.dp  # noqa: F401
    import pyosv.orient3d  # noqa: F401
    import pyosv.skinner  # noqa: F401
    import pyosv.voting3d  # noqa: F401


def test_package_root_public_surface_is_minimal_in_fresh_interpreter() -> None:
    command = (
        "import pyosv; print(sorted(name for name in vars(pyosv) if not name.startswith('_')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "[]"


def test_promotion_scripts_use_the_package_library() -> None:
    scripts = (
        REPO_ROOT / "scripts" / "compare_quality_reports.py",
        REPO_ROOT / "scripts" / "check_synthetic_quality_promotion_gate.py",
    )

    for script in scripts:
        assert "from pyosv.evaluation.promotion" in script.read_text(encoding="utf-8")

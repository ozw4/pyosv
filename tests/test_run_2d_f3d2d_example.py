from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_2d_f3d2d_help_exits_successfully() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "examples/run_2d_f3d2d.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "f3d2d" in result.stdout
    assert "--output-dir" in result.stdout

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import pytest

from pyosv.fault_warping import (
    FAULT_WARPING_CONTRACT_VERSION,
    FaultSurfaceGraph,
    FaultWarpingConfig,
    FaultWarpingEstimator,
    FaultWarpingInput,
    FaultWarpingResult,
    ReflectorSlopeVolume,
)
import pyosv.fault_warping as fault_warping


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORT_ROOTS = {
    "seis_atlas",
    "seis_fault_workflow",
    "torch",
    "segyio",
    "matplotlib",
    "pandas",
    "scipy",
    "numba",
    "threadpoolctl",
}


class _EstimatorDouble:
    def estimate(
        self,
        inputs: FaultWarpingInput,
        config: FaultWarpingConfig,
    ) -> FaultWarpingResult:
        raise AssertionError("the protocol test does not execute an estimator")


class _MissingEstimate:
    pass


def test_public_import_path_exports_only_the_contract_surface() -> None:
    assert FAULT_WARPING_CONTRACT_VERSION == "pyosv.fault_warping.v1"
    assert fault_warping.FaultSurfaceGraph is FaultSurfaceGraph
    assert fault_warping.ReflectorSlopeVolume is ReflectorSlopeVolume
    assert fault_warping.FaultWarpingInput is FaultWarpingInput
    assert fault_warping.FaultWarpingConfig is FaultWarpingConfig
    assert fault_warping.FaultWarpingResult is FaultWarpingResult
    assert fault_warping.FaultWarpingEstimator is FaultWarpingEstimator
    assert not hasattr(fault_warping, "estimate_fault_apparent_shifts")


def test_runtime_checkable_estimator_protocol_and_signature() -> None:
    assert isinstance(_EstimatorDouble(), FaultWarpingEstimator)
    assert not isinstance(_MissingEstimate(), FaultWarpingEstimator)
    assert not isinstance(object(), FaultWarpingEstimator)
    with pytest.raises(TypeError):
        FaultWarpingEstimator()  # type: ignore[abstract]

    signature = inspect.signature(FaultWarpingEstimator.estimate)
    assert list(signature.parameters) == ["self", "inputs", "config"]
    hints = get_type_hints(FaultWarpingEstimator.estimate)
    assert hints == {
        "inputs": FaultWarpingInput,
        "config": FaultWarpingConfig,
        "return": FaultWarpingResult,
    }


def test_contract_source_has_no_forbidden_import_direction() -> None:
    source_root = REPO_ROOT / "src" / "pyosv" / "fault_warping"
    imported_roots: set[str] = set()
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

    assert imported_roots.isdisjoint(FORBIDDEN_IMPORT_ROOTS)


def test_contract_import_does_not_load_forbidden_optional_dependencies() -> None:
    command = """
import json
import sys
before = set(sys.modules)
import pyosv.fault_warping
after = set(sys.modules)
print(json.dumps(sorted({name.split('.', 1)[0] for name in after - before})))
"""
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    imported_roots = set(json.loads(result.stdout))

    assert imported_roots.isdisjoint(FORBIDDEN_IMPORT_ROOTS)

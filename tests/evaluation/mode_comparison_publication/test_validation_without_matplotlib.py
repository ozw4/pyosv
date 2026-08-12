from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pyosv.evaluation.publication_manifest import build_publication_manifest
from pyosv.evaluation.publication_manifest_io import (
    artifact_file_record,
    validate_publication_directory,
    write_publication_manifest,
)


def _build_minimal_bundle(root: Path) -> Path:
    bundle = root / "publication"
    bundle.mkdir()
    (bundle / "experiment.json").write_text("{}\n", encoding="utf-8")
    (bundle / "uv.lock").write_text("lock-version = 1\n", encoding="utf-8")
    experiment_record = artifact_file_record(
        bundle,
        "experiment.json",
        tier="primary",
        role="resolved_experiment",
    )
    lock_record = artifact_file_record(
        bundle,
        "uv.lock",
        tier="primary",
        role="environment_lock",
    )
    manifest = build_publication_manifest(
        created_at_utc="2026-08-12T00:00:00Z",
        code={"repository": "ozw4/pyosv", "git_commit": "a" * 40, "dirty": False},
        environment={
            "python": "3.10.14",
            "lock_file": "uv.lock",
            "lock_sha256": lock_record["sha256"],
            "controls": {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "NUMBA_DISABLE_JIT": "0",
                "NUMBA_NUM_THREADS": "1",
                "PYOSV_ACCEL": "auto",
            },
        },
        datasets={
            "f3": {
                "dataset_id": "minimal-f3",
                "shape": [1, 1, 1],
                "dtype": ">f4",
                "files": [
                    {
                        "role": "input",
                        "filename": "input.dat",
                        "size": 4,
                        "sha256": "b" * 64,
                    }
                ],
            }
        },
        experiment={
            "config_file": "experiment.json",
            "config_sha256": experiment_record["sha256"],
            "source_runs": {
                "synthetic": {"completion_sha256": "c" * 64},
                "f3": {"completion_sha256": "d" * 64},
            },
        },
        semantics={
            "synthetic": "known_truth",
            "f3": "public_reference_agreement",
            "f3_public_reference_is_geological_truth": False,
            "f3_evaluation_units": 1,
        },
        artifacts=[experiment_record, lock_record],
    )
    write_publication_manifest(bundle, manifest)
    assert validate_publication_directory(bundle) == manifest
    return bundle


def test_validate_only_cli_imports_no_generation_or_runtime_stack(tmp_path: Path) -> None:
    bundle = _build_minimal_bundle(tmp_path)
    source_root = Path(__file__).parents[3] / "src"
    script = """
import builtins
import os
import subprocess
import sys

original_import = builtins.__import__
forbidden_prefixes = (
    'matplotlib',
    'numba',
    'threadpoolctl',
    'pyosv.evaluation.mode_comparison_publication',
    'pyosv.evaluation.synthetic_mode_comparison',
    'pyosv.evaluation.f3d_mode_comparison',
    'pyosv.evaluation.workflow3d',
)

def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden_prefixes):
        raise AssertionError('validate-only imported forbidden module: ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from pyosv.cli.mode_comparison_publication import main

def forbidden_command(*args, **kwargs):
    raise AssertionError('validate-only executed an external command')

original_environment = os.environ
control_names = {
    'PYTHONHASHSEED',
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_DISABLE_JIT',
    'NUMBA_NUM_THREADS',
    'PYOSV_ACCEL',
}

class ForbiddenEnvironment:
    def get(self, key, *args, **kwargs):
        if key in control_names:
            raise AssertionError('validate-only read publication controls')
        return original_environment.get(key, *args, **kwargs)

    def __getitem__(self, key):
        if key in control_names:
            raise AssertionError('validate-only read publication controls')
        return original_environment[key]

subprocess.run = forbidden_command
os.environ = ForbiddenEnvironment()
assert main(['--validate-only', '--output-dir', sys.argv[1]]) == 0
assert not any(
    name == prefix or name.startswith(prefix + '.')
    for name in sys.modules
    for prefix in forbidden_prefixes
)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", script, str(bundle)],
        cwd=source_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

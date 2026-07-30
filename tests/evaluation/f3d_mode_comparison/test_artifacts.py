from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyosv.evaluation.f3d_mode_comparison import (
    F3ArtifactError,
    F3DatasetIdentity,
    F3FileIdentity,
    F3ModeComparisonConfig,
    OFFICIAL_F3_DATASET_SPEC,
    F3StageArtifact,
    F3StageCorruptionError,
    F3WorkspaceMismatchError,
    build_f3d_mode_comparison_plan,
    canonical_fingerprint,
    canonical_json_bytes,
    implementation_identity,
    numerical_runtime_identity,
    prepare_run_workspace,
    run_fingerprint,
    validate_numerical_runtime_identity,
    validate_publication_runtime_identity,
)
from pyosv.evaluation.f3d_mode_comparison import artifacts as artifacts_module
from pyosv.evaluation.f3d_mode_comparison import runtime_identity as runtime_identity_module

_VERSIONS = {
    "pyosv": "test-pyosv",
    "python": "test-python",
    "numpy": "test-numpy",
    "scipy": "test-scipy",
}
_IMPLEMENTATION = {
    "software_versions": _VERSIONS,
    "algorithm_modules": {
        "scanner.py": {
            "sha256": hashlib.sha256(b"scanner").hexdigest(),
            "size": 7,
        }
    },
}
_RUNTIME_IDENTITY = {
    "runtime_identity_schema_version": 3,
    "python_implementation": "CPython",
    "platform_system": "Linux",
    "platform_machine": "test-machine",
    "byte_order": "little",
    "requested_acceleration_mode": "auto",
    "pyosv_accel": "auto",
    "numba_available": True,
    "numba_version": "test-numba",
    "numba_jit": {
        "status": "enabled",
        "enabled": True,
    },
    "effective_acceleration_state": "numba_jit_enabled",
    "thread_environment": {
        "OMP_NUM_THREADS": None,
        "OPENBLAS_NUM_THREADS": None,
        "MKL_NUM_THREADS": None,
        "NUMEXPR_NUM_THREADS": None,
        "GOTO_NUM_THREADS": None,
        "BLIS_NUM_THREADS": None,
        "VECLIB_MAXIMUM_THREADS": None,
    },
    "python_hash_seed": None,
    "numpy_disable_cpu_features": None,
    "numba_environment": {
        "NUMBA_DISABLE_JIT": None,
        "NUMBA_NUM_THREADS": None,
        "NUMBA_THREADING_LAYER": None,
        "NUMBA_CPU_NAME": None,
        "NUMBA_CPU_FEATURES": None,
    },
    "openblas_coretype": None,
    "numpy_build": {
        "status": "available",
        "sha256": hashlib.sha256(b"numpy-build").hexdigest(),
    },
    "numpy_runtime_cpu": {
        "status": "available",
        "features": ["AVX2", "SSE2"],
    },
    "numpy_runtime_blas": {
        "status": "available",
        "libraries": [
            {
                "implementation": "openblas",
                "version": "test-openblas",
                "threading_layer": "pthreads",
                "architecture": "test-arch",
                "effective_thread_count": 1,
            }
        ],
    },
    "scipy_build": {
        "status": "available",
        "sha256": hashlib.sha256(b"scipy-build").hexdigest(),
    },
}


def _plan():
    return build_f3d_mode_comparison_plan(F3ModeComparisonConfig())


def _identity(root: Path, content: bytes = b"dataset") -> F3DatasetIdentity:
    spec = OFFICIAL_F3_DATASET_SPEC
    return F3DatasetIdentity(
        dataset_id=spec.dataset_id,
        data_root=root,
        files=tuple(
            F3FileIdentity(
                role=role,
                filename=filename,
                resolved_path=root / filename,
                size=spec.expected_bytes,
                sha256=hashlib.sha256(content + role.encode()).hexdigest(),
                shape=spec.shape,
                storage_dtype=spec.storage_dtype,
            )
            for role, filename in spec.files
        ),
    )


def _workspace(path: Path):
    return prepare_run_workspace(
        path,
        _plan(),
        _identity(path.parent / "data"),
        resume=False,
        implementation=_IMPLEMENTATION,
        created_at="2026-01-01T00:00:00+00:00",
        source_provenance={
            "status": "not_available",
            "method": "git_cli",
            "commit": None,
            "dirty": None,
        },
    )


def _write_array(path: Path, values: np.ndarray | None = None) -> None:
    if values is None:
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.save(path / "fv.npy", values)


def _stage(workspace, writer=_write_array):
    return workspace.write_or_reuse_stage(
        "voting",
        parent_fingerprints=("0" * 64,),
        input_fingerprints={"scanner": "1" * 64},
        resolved_settings={"radius": 10, "weight": 0.5},
        artifacts=(F3StageArtifact("fv.npy", (2, 3, 4)),),
        writer=writer,
    )


def test_canonical_json_and_fingerprint_are_stable_and_finite() -> None:
    first = {"β": [1, 2.5], "a": {"y": True, "x": None}}
    second = {"a": {"x": None, "y": True}, "β": [1, 2.5]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_fingerprint(first) == canonical_fingerprint(second)
    assert canonical_json_bytes(first).decode("utf-8").startswith('{"a":')
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"bad": float("nan")})


def test_implementation_identity_hashes_source_and_versions(tmp_path: Path) -> None:
    source = tmp_path / "scanner.py"
    source.write_bytes(b"first")
    first = implementation_identity(
        software_versions=_VERSIONS,
        source_files={"scanner.py": source},
    )

    source.write_bytes(b"second")
    second = implementation_identity(
        software_versions=_VERSIONS,
        source_files={"scanner.py": source},
    )

    assert first["software_versions"] == _VERSIONS
    assert (
        first["algorithm_modules"]["scanner.py"]["sha256"]
        != (second["algorithm_modules"]["scanner.py"]["sha256"])
    )


def test_default_implementation_identity_hashes_f3_execution_sources() -> None:
    modules = implementation_identity()["algorithm_modules"]

    assert {
        "_dp/path2d.py",
        "evaluation/f3d_mode_comparison/artifacts.py",
        "evaluation/f3d_mode_comparison/builder.py",
        "evaluation/f3d_mode_comparison/data.py",
        "evaluation/f3d_mode_comparison/models.py",
        "evaluation/synthetic_quality/config.py",
        "evaluation/synthetic_quality/models.py",
        "evaluation/synthetic_quality/quality_metrics.py",
        "evaluation/synthetic_quality/stage_cache.py",
        "evaluation/synthetic_quality/stage_keys.py",
        "evaluation/synthetic_quality/variants.py",
        "evaluation/workflow3d.py",
        "experimental/boundary_seed_selection.py",
        "experimental/boundary_skinning.py",
        "experimental/boundary_thinning.py",
        "experimental/skin_diagnostics.py",
        "synthetic_metrics.py",
        "voting3d.py",
    } <= modules.keys()


def test_run_fingerprint_excludes_paths_but_includes_content_and_implementation(
    tmp_path: Path,
) -> None:
    plan = _plan()
    first = _identity(tmp_path / "first")
    moved = _identity(tmp_path / "moved")
    changed = _identity(tmp_path / "first", b"changed")

    baseline = run_fingerprint(plan, first, implementation=_IMPLEMENTATION)
    assert run_fingerprint(plan, moved, implementation=_IMPLEMENTATION) == baseline
    assert run_fingerprint(plan, changed, implementation=_IMPLEMENTATION) != baseline

    changed_versions = {
        **_IMPLEMENTATION,
        "software_versions": {**_VERSIONS, "numpy": "different"},
    }
    changed_source = {
        **_IMPLEMENTATION,
        "algorithm_modules": {
            "scanner.py": {
                "sha256": hashlib.sha256(b"different").hexdigest(),
                "size": 9,
            }
        },
    }
    assert run_fingerprint(plan, first, implementation=changed_versions) != baseline
    assert run_fingerprint(plan, first, implementation=changed_source) != baseline


def test_runtime_identity_is_canonical_and_default_is_valid() -> None:
    reordered = {
        name: (dict(reversed(tuple(value.items()))) if name == "thread_environment" else value)
        for name, value in reversed(tuple(_RUNTIME_IDENTITY.items()))
    }

    assert validate_numerical_runtime_identity(reordered) == _RUNTIME_IDENTITY
    assert canonical_json_bytes(reordered) == canonical_json_bytes(_RUNTIME_IDENTITY)
    validate_numerical_runtime_identity(numerical_runtime_identity())


@pytest.mark.parametrize("spelling", ("plain", "surrounding-whitespace", "uppercase"))
def test_runtime_identity_normalizes_configured_acceleration_mode(
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
) -> None:
    mode = runtime_identity_module._accel._ACCEL_MODE
    configured = {
        "plain": mode,
        "surrounding-whitespace": f"  {mode}  ",
        "uppercase": mode.upper(),
    }[spelling]
    monkeypatch.setenv("PYOSV_ACCEL", configured)

    assert numerical_runtime_identity()["pyosv_accel"] == mode


@pytest.mark.parametrize(
    ("mode", "numba_kind", "disable_jit", "expected_state", "expected_status"),
    (
        ("off", "available", "0", "python_only", "not_applicable"),
        ("auto", "unavailable", None, "python_only", "not_applicable"),
        ("auto", "available", "0", "numba_jit_enabled", "enabled"),
        ("auto", "available", "1", "numba_jit_disabled", "disabled"),
        ("required", "available", "0", "numba_jit_enabled", "enabled"),
        ("required", "available", "1", "numba_jit_disabled", "disabled"),
    ),
)
def test_fresh_process_runtime_identity_uses_effective_numba_jit_state(
    tmp_path: Path,
    mode: str,
    numba_kind: str,
    disable_jit: str | None,
    expected_state: str,
    expected_status: str,
) -> None:
    fake_root = tmp_path / "fake-modules"
    fake_root.mkdir()
    if numba_kind == "unavailable":
        (fake_root / "numba.py").write_text(
            "raise ImportError('Numba intentionally unavailable')\n",
            encoding="utf-8",
        )
    else:
        package = fake_root / "numba"
        package.mkdir()
        (package / "__init__.py").write_text(
            textwrap.dedent(
                """
                import os
                from types import SimpleNamespace

                __version__ = "test-fresh-numba"
                config = SimpleNamespace(
                    DISABLE_JIT=int(os.environ.get("NUMBA_DISABLE_JIT") or "0")
                )

                def njit(*args, **kwargs):
                    if len(args) == 1 and callable(args[0]) and not kwargs:
                        return args[0]
                    return lambda function: function
                """
            ),
            encoding="utf-8",
        )
    source_root = Path(__file__).resolve().parents[3] / "src"
    environment = os.environ.copy()
    environment["PYOSV_ACCEL"] = mode
    environment["PYTHONPATH"] = os.pathsep.join((str(fake_root), str(source_root)))
    if disable_jit is None:
        environment.pop("NUMBA_DISABLE_JIT", None)
    else:
        environment["NUMBA_DISABLE_JIT"] = disable_jit
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from pyosv.evaluation.f3d_mode_comparison.runtime_identity "
                "import numerical_runtime_identity; "
                "identity = numerical_runtime_identity(); "
                "print(json.dumps({"
                "'available': identity['numba_available'], "
                "'raw': identity['numba_environment']['NUMBA_DISABLE_JIT'], "
                "'jit': identity['numba_jit'], "
                "'state': identity['effective_acceleration_state']}))"
            ),
        ],
        cwd=source_root.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    identity = json.loads(completed.stdout)

    expected_available = mode != "off" and numba_kind == "available"
    assert identity == {
        "available": expected_available,
        "raw": disable_jit,
        "jit": {
            "status": expected_status,
            "enabled": (
                True
                if expected_status == "enabled"
                else False
                if expected_status == "disabled"
                else None
            ),
        },
        "state": expected_state,
    }


def test_runtime_identity_records_unknown_when_numba_jit_config_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_identity_module._accel, "NUMBA_AVAILABLE", True)
    monkeypatch.setitem(
        sys.modules,
        "numba",
        SimpleNamespace(__version__="test-numba"),
    )

    identity = numerical_runtime_identity()

    assert identity["numba_jit"] == {"status": "unknown", "enabled": None}
    assert identity["effective_acceleration_state"] == "numba_jit_unknown"
    with pytest.raises(ValueError, match="JIT state must be known"):
        validate_publication_runtime_identity(
            {
                **identity,
                "python_hash_seed": "0",
                "pyosv_accel": identity["requested_acceleration_mode"],
                "thread_environment": {
                    **identity["thread_environment"],
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
            }
        )


def _publication_runtime_identity() -> dict[str, object]:
    return {
        **_RUNTIME_IDENTITY,
        "python_hash_seed": "0",
        "thread_environment": {
            **_RUNTIME_IDENTITY["thread_environment"],
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }


def test_publication_runtime_identity_accepts_fixed_available_runtime() -> None:
    identity = _publication_runtime_identity()

    assert validate_publication_runtime_identity(identity) == identity


def test_publication_runtime_identity_rejects_auto_without_numba() -> None:
    identity = {
        **_publication_runtime_identity(),
        "numba_available": False,
        "numba_version": None,
        "numba_jit": {
            "status": "not_applicable",
            "enabled": None,
        },
        "effective_acceleration_state": "python_only",
    }

    with pytest.raises(ValueError, match="numba_jit_enabled"):
        validate_publication_runtime_identity(identity)


def test_publication_runtime_identity_rejects_explicit_acceleration_off() -> None:
    identity = {
        **_publication_runtime_identity(),
        "requested_acceleration_mode": "off",
        "pyosv_accel": "off",
        "numba_available": False,
        "numba_version": None,
        "numba_jit": {
            "status": "not_applicable",
            "enabled": None,
        },
        "effective_acceleration_state": "python_only",
    }

    with pytest.raises(ValueError, match="numba_jit_enabled"):
        validate_publication_runtime_identity(identity)


@pytest.mark.parametrize("value", (None, "", "0"))
def test_publication_runtime_identity_accepts_canonical_false_numba_disable_jit(
    value: str | None,
) -> None:
    identity = _publication_runtime_identity()
    identity["numba_environment"] = {
        **identity["numba_environment"],
        "NUMBA_DISABLE_JIT": value,
        "NUMBA_NUM_THREADS": "1",
    }

    assert validate_publication_runtime_identity(identity) == identity


@pytest.mark.parametrize("value", ("00", "false", "1"))
def test_publication_runtime_identity_rejects_noncanonical_numba_disable_jit(
    value: str,
) -> None:
    identity = _publication_runtime_identity()
    identity["numba_environment"] = {
        **identity["numba_environment"],
        "NUMBA_DISABLE_JIT": value,
    }
    if value == "1":
        identity["numba_jit"] = {"status": "disabled", "enabled": False}
        identity["effective_acceleration_state"] = "numba_jit_disabled"

    with pytest.raises(ValueError, match="Numba JIT|NUMBA_DISABLE_JIT"):
        validate_publication_runtime_identity(identity)


@pytest.mark.parametrize(("value", "accepted"), ((None, True), ("1", True), ("2", False)))
def test_publication_runtime_identity_numba_thread_policy(
    value: str | None,
    accepted: bool,
) -> None:
    identity = _publication_runtime_identity()
    identity["numba_environment"] = {
        **identity["numba_environment"],
        "NUMBA_NUM_THREADS": value,
    }

    if accepted:
        assert validate_publication_runtime_identity(identity) == identity
    else:
        with pytest.raises(ValueError, match="NUMBA_NUM_THREADS"):
            validate_publication_runtime_identity(identity)


def test_jit_disabled_identity_is_valid_internally_but_not_for_publication() -> None:
    identity = {
        **_publication_runtime_identity(),
        "numba_jit": {"status": "disabled", "enabled": False},
        "effective_acceleration_state": "numba_jit_disabled",
        "numba_environment": {
            **_RUNTIME_IDENTITY["numba_environment"],
            "NUMBA_DISABLE_JIT": "1",
            "NUMBA_NUM_THREADS": "1",
        },
    }

    assert validate_numerical_runtime_identity(identity) == identity
    with pytest.raises(ValueError, match="Numba JIT"):
        validate_publication_runtime_identity(identity)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("python_hash_seed", None, "PYTHONHASHSEED"),
        ("python_hash_seed", "1", "PYTHONHASHSEED"),
        ("pyosv_accel", None, "PYOSV_ACCEL"),
        (
            "numpy_runtime_cpu",
            {"status": "not_available", "features": None},
            "numpy_runtime_cpu",
        ),
        (
            "numpy_runtime_blas",
            {"status": "not_available", "libraries": None},
            "numpy_runtime_blas",
        ),
        (
            "scipy_build",
            {"status": "not_available", "sha256": None},
            "scipy_build",
        ),
    ),
)
def test_publication_runtime_identity_rejects_unfixed_or_unavailable_runtime(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_publication_runtime_identity({**_publication_runtime_identity(), field: value})


@pytest.mark.parametrize(
    "name",
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
)
def test_publication_runtime_identity_requires_single_thread_environment(
    name: str,
) -> None:
    identity = _publication_runtime_identity()
    identity["thread_environment"] = {
        **identity["thread_environment"],
        name: "2",
    }

    with pytest.raises(ValueError, match=name):
        validate_publication_runtime_identity(identity)


def test_runtime_identity_rejects_unsorted_features_and_runtime_library_paths() -> None:
    with pytest.raises(ValueError, match="sorted feature set"):
        validate_numerical_runtime_identity(
            {
                **_RUNTIME_IDENTITY,
                "numpy_runtime_cpu": {
                    "status": "available",
                    "features": ["SSE2", "AVX2"],
                },
            }
        )
    libraries = _RUNTIME_IDENTITY["numpy_runtime_blas"]["libraries"]
    assert isinstance(libraries, list)
    for implementation in (
        "/usr/lib/libopenblas.so",
        "openblas at /usr/lib/libopenblas.so",
    ):
        with pytest.raises(ValueError, match="absolute path"):
            validate_numerical_runtime_identity(
                {
                    **_RUNTIME_IDENTITY,
                    "numpy_runtime_blas": {
                        "status": "available",
                        "libraries": [
                            {
                                **libraries[0],
                                "implementation": implementation,
                            }
                        ],
                    },
                }
            )


def test_numpy_build_digest_normalizes_paths_and_has_stable_unavailable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = {
        "Build Dependencies": {
            "blas": {
                "name": "openblas",
                "version": "1.0",
                "lib directory": "/first/build/lib",
            }
        },
        "Compilers": {
            "c": {
                "commands": "cc",
                "c_args": [
                    "-O3",
                    "/first/build/specs",
                    "-I/first/build/include",
                    "-Wl,-rpath,/first/build/lib",
                ],
            }
        },
    }
    second = {
        "Build Dependencies": {
            "blas": {
                "lib directory": "/moved/build/lib",
                "version": "1.0",
                "name": "openblas",
            }
        },
        "Compilers": {
            "c": {
                "c_args": [
                    "-O3",
                    "/moved/build/specs",
                    "-I/moved/build/include",
                    "-Wl,-rpath,/moved/build/lib",
                ],
                "commands": "cc",
            }
        },
    }
    monkeypatch.setattr(runtime_identity_module.np.__config__, "CONFIG", first)
    first_identity = runtime_identity_module._numpy_build_identity()
    monkeypatch.setattr(runtime_identity_module.np.__config__, "CONFIG", second)

    assert runtime_identity_module._numpy_build_identity() == first_identity

    monkeypatch.setattr(runtime_identity_module.np.__config__, "CONFIG", None)
    monkeypatch.setattr(
        runtime_identity_module.np.__config__,
        "get_info",
        lambda name: {},
        raising=False,
    )
    assert runtime_identity_module._numpy_build_identity() == {
        "status": "not_available",
        "sha256": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("commands", "clang"),
        ("c_args", ["-O2"]),
    ),
)
def test_numpy_build_digest_tracks_compiler_commands_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    config = {
        "Compilers": {
            "c": {
                "commands": "cc",
                "c_args": ["-O3"],
            }
        }
    }
    monkeypatch.setattr(runtime_identity_module.np.__config__, "CONFIG", config)
    baseline = runtime_identity_module._numpy_build_identity()
    changed = {
        "Compilers": {
            "c": {
                **config["Compilers"]["c"],
                field: value,
            }
        }
    }
    monkeypatch.setattr(runtime_identity_module.np.__config__, "CONFIG", changed)

    assert runtime_identity_module._numpy_build_identity() != baseline


@pytest.mark.parametrize(
    "change",
    (
        {"requested_acceleration_mode": "required", "pyosv_accel": "required"},
        {
            "numba_available": False,
            "numba_version": None,
            "numba_jit": {
                "status": "not_applicable",
                "enabled": None,
            },
            "effective_acceleration_state": "python_only",
        },
        {"numba_version": "other-numba"},
        {"platform_machine": "other-machine"},
        {
            "thread_environment": {
                **_RUNTIME_IDENTITY["thread_environment"],
                "OMP_NUM_THREADS": "2",
            }
        },
        {"python_hash_seed": "123"},
        {
            "numpy_build": {
                "status": "available",
                "sha256": hashlib.sha256(b"other-build").hexdigest(),
            }
        },
        {
            "numpy_runtime_cpu": {
                "status": "available",
                "features": ["AVX2", "AVX512F", "SSE2"],
            }
        },
        {"numpy_disable_cpu_features": "AVX512F"},
        {
            "numpy_runtime_blas": {
                "status": "available",
                "libraries": [
                    {
                        "implementation": "openblas",
                        "version": "test-openblas",
                        "threading_layer": "pthreads",
                        "architecture": "test-arch",
                        "effective_thread_count": 2,
                    }
                ],
            }
        },
        {
            "scipy_build": {
                "status": "available",
                "sha256": hashlib.sha256(b"other-scipy-build").hexdigest(),
            }
        },
        {
            "numba_jit": {
                "status": "disabled",
                "enabled": False,
            },
            "effective_acceleration_state": "numba_jit_disabled",
            "numba_environment": {
                **_RUNTIME_IDENTITY["numba_environment"],
                "NUMBA_DISABLE_JIT": "1",
            },
        },
        {
            "numba_environment": {
                **_RUNTIME_IDENTITY["numba_environment"],
                "NUMBA_NUM_THREADS": "2",
                "NUMBA_THREADING_LAYER": "workqueue",
                "NUMBA_CPU_NAME": "generic",
                "NUMBA_CPU_FEATURES": "-avx512f",
            }
        },
        {"openblas_coretype": "Haswell"},
        {
            "thread_environment": {
                **_RUNTIME_IDENTITY["thread_environment"],
                "BLIS_NUM_THREADS": "2",
            }
        },
    ),
)
def test_run_fingerprint_tracks_runtime_identity_fields(
    tmp_path: Path,
    change: dict[str, object],
) -> None:
    plan = _plan()
    dataset = _identity(tmp_path / "data")
    changed_runtime = {**_RUNTIME_IDENTITY, **change}
    baseline = run_fingerprint(
        plan,
        dataset,
        implementation=_IMPLEMENTATION,
        runtime_identity=_RUNTIME_IDENTITY,
    )

    assert (
        run_fingerprint(
            plan,
            dataset,
            implementation=_IMPLEMENTATION,
            runtime_identity=changed_runtime,
        )
        != baseline
    )


@pytest.mark.parametrize(
    "runtime_identity",
    (
        {},
        {**_RUNTIME_IDENTITY, "unknown": None},
        {**_RUNTIME_IDENTITY, "runtime_identity_schema_version": 1},
        {**_RUNTIME_IDENTITY, "runtime_identity_schema_version": 2},
        {**_RUNTIME_IDENTITY, "runtime_identity_schema_version": True},
        {**_RUNTIME_IDENTITY, "numba_version": ""},
        {**_RUNTIME_IDENTITY, "numba_version": None},
        {**_RUNTIME_IDENTITY, "effective_acceleration_state": "unknown"},
        {
            **_RUNTIME_IDENTITY,
            "numba_jit": {"status": "unknown", "enabled": None},
            "effective_acceleration_state": "numba_jit_enabled",
        },
        {
            **_RUNTIME_IDENTITY,
            "numba_jit": {"status": "enabled", "enabled": "true"},
        },
        {
            **_RUNTIME_IDENTITY,
            "numba_available": False,
            "numba_version": None,
        },
        {
            **_RUNTIME_IDENTITY,
            "requested_acceleration_mode": "off",
            "pyosv_accel": "off",
        },
        {
            **_RUNTIME_IDENTITY,
            "requested_acceleration_mode": "required",
            "pyosv_accel": "required",
            "numba_available": False,
            "numba_version": None,
            "numba_jit": {"status": "not_applicable", "enabled": None},
            "effective_acceleration_state": "python_only",
        },
        {**_RUNTIME_IDENTITY, "python_hash_seed": object()},
        {
            **_RUNTIME_IDENTITY,
            "thread_environment": {
                **_RUNTIME_IDENTITY["thread_environment"],
                "UNKNOWN_THREADS": "1",
            },
        },
        {
            **_RUNTIME_IDENTITY,
            "numpy_runtime_cpu": {
                "status": "unknown",
                "features": None,
            },
        },
        {
            **_RUNTIME_IDENTITY,
            "numpy_runtime_blas": {
                "status": "not_available",
                "libraries": [],
            },
        },
        {
            **_RUNTIME_IDENTITY,
            "scipy_build": {
                "status": "available",
                "sha256": "A" * 64,
            },
        },
    ),
)
def test_run_fingerprint_rejects_malformed_runtime_identity(
    tmp_path: Path,
    runtime_identity: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="runtime identity"):
        run_fingerprint(
            _plan(),
            _identity(tmp_path / "data"),
            implementation=_IMPLEMENTATION,
            runtime_identity=runtime_identity,
        )


def test_runtime_mismatch_rejects_resume_without_modifying_workspace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run"
    first = prepare_run_workspace(
        path,
        _plan(),
        _identity(tmp_path / "data"),
        resume=False,
        implementation=_IMPLEMENTATION,
        runtime_identity=_RUNTIME_IDENTITY,
    )
    stage_marker = first.path / "stages" / "scanner" / "existing-stage"
    stage_marker.mkdir()
    completion = first.path / "completion.json"
    completion.write_bytes(b"existing completion")
    manifest_before = (first.path / "run_manifest.json").read_bytes()
    changed_runtime = {
        **_RUNTIME_IDENTITY,
        "platform_machine": "other-machine",
    }

    with pytest.raises(F3WorkspaceMismatchError, match="runtime_identity"):
        prepare_run_workspace(
            path,
            _plan(),
            _identity(tmp_path / "moved-data"),
            resume=True,
            implementation=_IMPLEMENTATION,
            runtime_identity=changed_runtime,
        )

    assert (first.path / "run_manifest.json").read_bytes() == manifest_before
    assert stage_marker.is_dir()
    assert completion.read_bytes() == b"existing completion"


@pytest.mark.parametrize(
    "implementation",
    (
        {},
        {"software_versions": _VERSIONS, "algorithm_modules": {}},
        {**_IMPLEMENTATION, "source_path": "/checkout/src"},
        {
            "software_versions": _VERSIONS,
            "algorithm_modules": {
                "scanner.py": {
                    **_IMPLEMENTATION["algorithm_modules"]["scanner.py"],
                    "path": "/checkout/src/scanner.py",
                }
            },
        },
    ),
)
def test_run_fingerprint_rejects_incomplete_or_path_bearing_implementation(
    tmp_path: Path,
    implementation: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="implementation"):
        run_fingerprint(
            _plan(),
            _identity(tmp_path / "data"),
            implementation=implementation,
        )


def test_run_fingerprint_rejects_dataset_identity_mismatches(tmp_path: Path) -> None:
    plan = _plan()
    valid = _identity(tmp_path / "data")
    first_file = valid.files[0]
    mismatches = (
        replace(valid, dataset_id="other-dataset"),
        replace(valid, files=valid.files[:-1]),
        replace(valid, files=(replace(first_file, filename="other.dat"), *valid.files[1:])),
    )

    for identity in mismatches:
        with pytest.raises(ValueError, match="dataset identity"):
            run_fingerprint(plan, identity, implementation=_IMPLEMENTATION)

    for field, value in (
        ("shape", (1, 1, 1)),
        ("storage_dtype", "<f4"),
        ("size", 1),
    ):
        corrupted = _identity(tmp_path / field)
        object.__setattr__(corrupted.files[0], field, value)
        with pytest.raises(ValueError, match="dataset identity"):
            run_fingerprint(plan, corrupted, implementation=_IMPLEMENTATION)


def test_run_fingerprint_revalidates_dataset_checksum(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "data")
    object.__setattr__(identity.files[0], "sha256", None)

    with pytest.raises(ValueError, match="dataset identity checksum"):
        run_fingerprint(_plan(), identity, implementation=_IMPLEMENTATION)


def test_reference_content_changes_run_fingerprint(tmp_path: Path) -> None:
    plan = _plan()
    baseline_identity = _identity(tmp_path / "data")
    reference = baseline_identity.files[1]
    changed_identity = replace(
        baseline_identity,
        files=(
            baseline_identity.files[0],
            replace(reference, sha256=hashlib.sha256(b"changed-reference").hexdigest()),
            *baseline_identity.files[2:],
        ),
    )

    assert run_fingerprint(
        plan,
        changed_identity,
        implementation=_IMPLEMENTATION,
    ) != run_fingerprint(
        plan,
        baseline_identity,
        implementation=_IMPLEMENTATION,
    )


def test_workspace_new_resume_and_manifest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "run"
    first = _workspace(path)

    assert first.resumed is False
    assert set(item.name for item in path.iterdir()) == {
        "run_manifest.json",
        "stages",
        "cells",
        "reports",
    }
    assert set(item.name for item in (path / "stages").iterdir()) == {
        "scanner",
        "voting",
        "thinning",
        "skinning",
    }
    assert not (path / "completion.json").exists()
    with pytest.raises(FileExistsError):
        _workspace(path)

    resumed = prepare_run_workspace(
        path,
        _plan(),
        _identity(tmp_path / "moved-data"),
        resume=True,
        implementation=_IMPLEMENTATION,
    )
    assert resumed.resumed is True
    assert resumed.fingerprint == first.fingerprint
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["data_root"].endswith("/data")

    changed = {**_IMPLEMENTATION, "software_versions": {**_VERSIONS, "scipy": "changed"}}
    with pytest.raises(F3WorkspaceMismatchError, match="implementation_identity"):
        prepare_run_workspace(
            path,
            _plan(),
            _identity(tmp_path / "data"),
            resume=True,
            implementation=changed,
        )


@pytest.mark.parametrize(
    ("resume", "error_type"),
    ((False, FileExistsError), (True, F3ArtifactError)),
)
def test_workspace_rejects_symlink_output_path(
    tmp_path: Path,
    resume: bool,
    error_type: type[Exception],
) -> None:
    target = tmp_path / "target"
    workspace = _workspace(target)
    link = tmp_path / "run-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    original_manifest = (target / "run_manifest.json").read_bytes()

    with pytest.raises(error_type):
        prepare_run_workspace(
            link,
            _plan(),
            _identity(tmp_path / "data"),
            resume=resume,
            implementation=_IMPLEMENTATION,
        )

    assert link.is_symlink()
    assert workspace.path == target
    assert (target / "run_manifest.json").read_bytes() == original_manifest


def test_workspace_rejects_output_inside_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_path = data_root / "run"

    with pytest.raises(ValueError, match="must not be inside"):
        prepare_run_workspace(
            output_path,
            _plan(),
            _identity(data_root),
            resume=False,
            implementation=_IMPLEMENTATION,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("resume", "error_type"),
    ((False, FileExistsError), (True, F3ArtifactError)),
)
def test_workspace_rejects_dangling_symlink_output_path(
    tmp_path: Path,
    resume: bool,
    error_type: type[Exception],
) -> None:
    target = tmp_path / "missing-target"
    link = tmp_path / "run-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(error_type):
        prepare_run_workspace(
            link,
            _plan(),
            _identity(tmp_path / "data"),
            resume=resume,
            implementation=_IMPLEMENTATION,
        )

    assert link.is_symlink()
    assert not target.exists()


def test_stage_is_completed_last_validated_and_reused_without_writer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    result = _stage(workspace)

    assert result.reused is False
    assert result.path == workspace.path / "stages" / "voting" / result.fingerprint
    assert set(item.name for item in result.path.iterdir()) == {
        "fv.npy",
        "stage_manifest.json",
        "complete.json",
    }
    completion = json.loads((result.path / "complete.json").read_text(encoding="utf-8"))
    assert set(completion["files"]) == {"fv.npy", "stage_manifest.json"}
    assert completion["files"]["fv.npy"]["size"] == (result.path / "fv.npy").stat().st_size

    calls = 0

    def should_not_run(path: Path) -> None:
        nonlocal calls
        calls += 1

    reused = _stage(workspace, should_not_run)
    assert reused.reused is True
    assert calls == 0

    shutil.rmtree(result.path)
    assert not result.path.exists()


def test_forced_stage_recompute_failure_preserves_valid_stage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    result = _stage(workspace)
    original = {item.name: item.read_bytes() for item in result.path.iterdir() if item.is_file()}

    def fail_writer(path: Path) -> None:
        _write_array(path, np.ones((2, 3, 4), dtype=np.float32))
        raise RuntimeError("forced writer failed")

    with pytest.raises(RuntimeError, match="forced writer failed"):
        workspace.write_or_reuse_stage(
            "voting",
            parent_fingerprints=("0" * 64,),
            input_fingerprints={"scanner": "1" * 64},
            resolved_settings={"radius": 10, "weight": 0.5},
            artifacts=(F3StageArtifact("fv.npy", (2, 3, 4)),),
            writer=fail_writer,
            fingerprint=result.fingerprint,
            force_recompute=True,
        )

    assert {
        item.name: item.read_bytes() for item in result.path.iterdir() if item.is_file()
    } == original


@pytest.mark.parametrize("corruption", ("missing", "extra", "hash", "symlink", "shape", "dtype"))
def test_corrupt_stage_is_rejected_without_compute(
    tmp_path: Path,
    corruption: str,
) -> None:
    workspace = _workspace(tmp_path / "run")
    result = _stage(workspace)
    artifact = result.path / "fv.npy"
    if corruption == "missing":
        (result.path / "complete.json").unlink()
    elif corruption == "extra":
        (result.path / "extra.bin").write_bytes(b"extra")
    elif corruption == "hash":
        artifact.write_bytes(artifact.read_bytes() + b"changed")
    elif corruption == "symlink":
        artifact.unlink()
        artifact.symlink_to(result.path / "stage_manifest.json")
    elif corruption in {"shape", "dtype"}:
        values = np.zeros(
            (2, 3, 5) if corruption == "shape" else (2, 3, 4),
            dtype=np.float32 if corruption == "shape" else np.float64,
        )
        np.save(artifact, values)
        _rehash_stage_file(result.path, "fv.npy")
    else:
        raise AssertionError(corruption)
    before = sorted(item.name for item in result.path.iterdir())
    calls = 0

    def should_not_run(path: Path) -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(F3StageCorruptionError):
        _stage(workspace, should_not_run)
    assert calls == 0
    assert sorted(item.name for item in result.path.iterdir()) == before


def _rehash_stage_file(stage: Path, filename: str) -> None:
    payload = (stage / filename).read_bytes()
    metadata = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    manifest_path = stage / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = metadata
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_payload)
    completion_path = stage / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files"][filename] = metadata
    completion["files"]["stage_manifest.json"] = {
        "sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "size": len(manifest_payload),
    }
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")


@pytest.mark.parametrize("fault", ("write", "hash", "fsync", "rename"))
def test_stage_fault_does_not_publish_or_leave_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    workspace = _workspace(tmp_path / "run")
    parent = workspace.path / "stages" / "voting"

    if fault == "write":
        monkeypatch.setattr(
            artifacts_module,
            "_write_bytes",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write fault")),
        )
    elif fault == "hash":
        monkeypatch.setattr(
            artifacts_module,
            "_file_metadata",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("hash fault")),
        )
    elif fault == "fsync":
        monkeypatch.setattr(
            artifacts_module,
            "_fsync_directory",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fsync fault")),
        )
    elif fault == "rename":
        monkeypatch.setattr(
            artifacts_module,
            "_rename_noreplace",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("rename fault")),
        )

    with pytest.raises(OSError, match="fault"):
        _stage(workspace)
    assert list(parent.iterdir()) == []


def test_stage_fsyncs_callback_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "run")
    synced: list[str] = []
    original = artifacts_module._fsync_file

    def track_fsync(path: Path) -> None:
        synced.append(path.name)
        original(path)

    monkeypatch.setattr(artifacts_module, "_fsync_file", track_fsync)

    _stage(workspace)

    assert synced == ["fv.npy"]


def test_stage_artifact_fsync_failure_cleans_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "run")
    parent = workspace.path / "stages" / "voting"
    monkeypatch.setattr(
        artifacts_module,
        "_fsync_file",
        lambda path: (_ for _ in ()).throw(OSError("artifact fsync fault")),
    )

    with pytest.raises(OSError, match="artifact fsync fault"):
        _stage(workspace)

    assert list(parent.iterdir()) == []


def test_directory_fsync_failure_is_propagated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifacts_module.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError("directory fsync fault")),
    )

    with pytest.raises(OSError, match="directory fsync fault"):
        artifacts_module._fsync_directory(tmp_path)


def test_resume_cleans_only_owned_nonsymlink_temporary_directories(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    parent = workspace.path / "stages" / "scanner"
    owned = parent / ".pyosv-stage-tmp-deadbeef"
    owned.mkdir()
    unrelated = parent / ".other-temp"
    unrelated.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    symlink = parent / ".pyosv-stage-tmp-symlink"
    try:
        symlink.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    prepare_run_workspace(
        workspace.path,
        _plan(),
        _identity(tmp_path / "data"),
        resume=True,
        implementation=_IMPLEMENTATION,
    )

    assert not owned.exists()
    assert unrelated.exists()
    assert symlink.is_symlink()
    assert target.exists()


def test_existing_wrong_fingerprint_manifest_is_not_overwritten(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    manifest_path = workspace.path / "run_manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["run_fingerprint"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered = manifest_path.read_bytes()

    with pytest.raises(F3WorkspaceMismatchError, match="run_fingerprint"):
        prepare_run_workspace(
            workspace.path,
            _plan(),
            _identity(tmp_path / "data"),
            resume=True,
            implementation=_IMPLEMENTATION,
        )
    assert manifest_path.read_bytes() == tampered
    assert manifest_path.read_bytes() != original


def test_stage_writer_rejects_mixed_generation_files(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")

    def writer(path: Path) -> None:
        _write_array(path)
        (path / "old-generation.npy").write_bytes(b"old")

    with pytest.raises(F3StageCorruptionError, match="extra"):
        _stage(workspace, writer)
    assert list((workspace.path / "stages" / "voting").iterdir()) == []


def test_workspace_manifest_atomic_replace_fault_cleans_new_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "run"
    real_replace = os.replace

    def fail_replace(source: Path, destination: Path) -> None:
        if Path(destination).name == "run_manifest.json":
            raise OSError("replace fault")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace fault"):
        _workspace(path)
    assert not path.exists()


def test_config_change_changes_run_fingerprint(tmp_path: Path) -> None:
    baseline = _plan()
    changed = replace(
        baseline,
        boundary_diagnostic_margin=baseline.boundary_diagnostic_margin + 1,
    )

    assert run_fingerprint(
        baseline,
        _identity(tmp_path / "data"),
        implementation=_IMPLEMENTATION,
    ) != run_fingerprint(
        changed,
        _identity(tmp_path / "data"),
        implementation=_IMPLEMENTATION,
    )

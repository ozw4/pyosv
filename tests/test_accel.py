import builtins
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

MODULE_NAME = "pyosv._accel"
REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_OPERATIONS = """
import json
import sys

import numpy as np

from pyosv import dp
from pyosv._accel import NUMBA_AVAILABLE
from pyosv.skinner import FaultSkinner
from pyosv.voting2d import OptimalPathVoter
from pyosv.voting3d import OptimalSurfaceVoter

cost = np.random.default_rng(406).normal(size=(6, 5)).astype(np.float32)
path = dp.find_path_2d(cost, lmin=-2, bstrain=2, attribute_smoothing=1)
surface_cost = np.random.default_rng(407).normal(size=(3, 4, 5)).astype(np.float32)
surface = dp.find_surface_3d(
    surface_cost,
    lmin=-2,
    bstrain1=2,
    bstrain2=2,
    attribute_smoothing=1,
)

voter2d = OptimalPathVoter(ru=1, rv=3)
voter2d.set_attribute_smoothing(0)
voter2d.set_path_smoothing(0.0)
ft2d = np.zeros((15, 15), dtype=np.float32)
pt2d = np.zeros_like(ft2d)
ft2d[7, 3:12] = 0.9
voted2d = voter2d.apply_voting(d=3, fm=0.5, ft=ft2d, pt=pt2d)

voter3d = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
voter3d.set_attribute_smoothing(0)
voter3d.set_surface_smoothing(0.0, 0.0)
ft3d = np.zeros((11, 11, 11), dtype=np.float32)
pt3d = np.zeros_like(ft3d)
tt3d = np.full_like(ft3d, 90.0)
ft3d[3:8, 5, 3:8] = 0.8
voted3d = voter3d.apply_voting(d=3, fm=0.5, ft=ft3d, pt=pt3d, tt=tt3d)

fv = np.zeros((13, 13, 13), dtype=np.float32)
vp = np.zeros_like(fv)
vt = np.full_like(fv, 90.0)
fv[3:10, 6, 3:10] = 0.9
skins = FaultSkinner(min_skin_size=1).find_skins(
    fv,
    vp,
    vt,
    min_likelihood=0.5,
    ru=5,
    rv=6,
    rw=6,
    max_steps=2,
    reskin=False,
)

print(json.dumps({
    "numba_available": NUMBA_AVAILABLE,
    "numba_loaded": any(
        name == "numba" or name.startswith("numba.") for name in sys.modules
    ),
    "dp": [path.tolist(), surface.tolist()],
    "voting2d": [array.tolist() for array in voted2d],
    "voting3d": [array.tolist() for array in voted3d],
    "skins": [[list(cell.index) for cell in skin] for skin in skins],
}))
"""


def _run_accel_subprocess(
    code: str,
    *,
    mode: str | None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, env.get("PYTHONPATH", "")) if part
    )
    if mode is None:
        env.pop("PYOSV_ACCEL", None)
    else:
        env["PYOSV_ACCEL"] = mode
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def fresh_accel_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("PYOSV_ACCEL", raising=False)
    sys.modules.pop(MODULE_NAME, None)
    importlib.invalidate_caches()
    yield
    sys.modules.pop(MODULE_NAME, None)
    importlib.invalidate_caches()


def test_njit_is_noop_when_numba_import_fails(
    monkeypatch: pytest.MonkeyPatch,
    fresh_accel_module: None,
) -> None:
    original_import = builtins.__import__

    def import_without_numba(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba intentionally unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_numba)

    accel = importlib.import_module(MODULE_NAME)

    def add(left: int, right: int) -> int:
        return left + right

    assert accel.NUMBA_AVAILABLE is False
    assert accel.njit(add) is add
    assert accel.njit(cache=True)(add) is add
    assert accel.njit("int64(int64, int64)")(add) is add
    assert accel.njit(cache=True)(add)(1, 2) == 3


def test_njit_delegates_to_numba_when_available(
    monkeypatch: pytest.MonkeyPatch,
    fresh_accel_module: None,
) -> None:
    fake_numba = types.ModuleType("numba")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_njit(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return ("compiled", args[0])

        def decorate(func: Any) -> tuple[str, Any]:
            return ("compiled-with-options", func)

        return decorate

    fake_numba.njit = fake_njit
    monkeypatch.setitem(sys.modules, "numba", fake_numba)

    accel = importlib.import_module(MODULE_NAME)

    def add(left: int, right: int) -> int:
        return left + right

    assert accel.NUMBA_AVAILABLE is True
    assert accel.njit(add) == ("compiled", add)
    assert accel.njit(cache=True)(add) == ("compiled-with-options", add)
    assert calls == [((add,), {}), ((), {"cache": True})]


def test_unset_and_auto_modes_preserve_default_import_behavior() -> None:
    code = """
    import json
    from pyosv._accel import NUMBA_AVAILABLE
    print(json.dumps(NUMBA_AVAILABLE))
    """

    unset = _run_accel_subprocess(code, mode=None).stdout
    auto = _run_accel_subprocess(code, mode="auto").stdout

    assert json.loads(unset) is json.loads(auto)


@pytest.mark.parametrize("mode", [" OFF ", "Off"])
def test_mode_is_case_insensitive_and_ignores_surrounding_whitespace(mode: str) -> None:
    result = _run_accel_subprocess(
        """
        from pyosv._accel import NUMBA_AVAILABLE
        assert NUMBA_AVAILABLE is False
        """,
        mode=mode,
    )

    assert result.returncode == 0


@pytest.mark.parametrize("mode", ["", "sometimes"])
def test_invalid_mode_raises_value_error(mode: str) -> None:
    result = _run_accel_subprocess("import pyosv._accel", mode=mode, check=False)

    assert result.returncode != 0
    assert "ValueError: PYOSV_ACCEL must be one of" in result.stderr


def test_required_mode_uses_numba_when_available() -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("Numba is not installed")

    result = _run_accel_subprocess(
        """
        from pyosv._accel import NUMBA_AVAILABLE
        assert NUMBA_AVAILABLE is True
        """,
        mode="required",
    )

    assert result.returncode == 0


def test_required_mode_preserves_failed_numba_import_as_cause() -> None:
    result = _run_accel_subprocess(
        """
        import builtins

        original_import = builtins.__import__

        def import_without_numba(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "numba" or name.startswith("numba."):
                raise ImportError("simulated numba import failure")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_without_numba
        try:
            import pyosv._accel
        except ImportError as exc:
            assert "Numba is required" in str(exc)
            assert "PYOSV_ACCEL=auto/off" in str(exc)
            assert "simulated numba import failure" in str(exc)
            assert isinstance(exc.__cause__, ImportError)
            assert str(exc.__cause__) == "simulated numba import failure"
        else:
            raise AssertionError("required mode silently fell back")
        """,
        mode="required",
    )

    assert result.returncode == 0


def test_off_mode_does_not_import_numba_and_core_results_match_auto() -> None:
    off = json.loads(_run_accel_subprocess(CORE_OPERATIONS, mode="off").stdout)
    auto = json.loads(_run_accel_subprocess(CORE_OPERATIONS, mode="auto").stdout)

    assert off["numba_available"] is False
    assert off["numba_loaded"] is False
    assert auto["numba_loaded"] is auto["numba_available"]
    assert off["skins"] == auto["skins"]
    for operation in ("dp", "voting2d", "voting3d"):
        for off_array, auto_array in zip(off[operation], auto[operation]):
            np.testing.assert_allclose(
                np.asarray(off_array),
                np.asarray(auto_array),
                rtol=1.0e-6,
                atol=1.0e-6,
            )

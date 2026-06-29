import builtins
import importlib
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

MODULE_NAME = "pyosv._accel"


@pytest.fixture
def fresh_accel_module() -> Iterator[None]:
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

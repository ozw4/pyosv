import pyosv


def test_package_imports() -> None:
    assert isinstance(pyosv.__version__, str)
    assert pyosv.__version__

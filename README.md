# pyosv

`pyosv` is a Python package scaffold for reimplementing practical OSV (Optimal Surface Voting / Optimal Path Voting) workflows.

The project uses the local `reference_osv/` directory as a read-only reference implementation. That directory is expected to be a bind mount and is not part of this repository.

## Status

This repository is scaffold-only. OSV algorithms, I/O helpers, interpolation adapters, filters, voting kernels, scanners, and skinning will be implemented in later issues.

## Setup

```bash
python -m pip install -e ".[dev]"
```

Verify the package import:

```bash
python -c "import pyosv; print(pyosv.__version__)"
```

## Checks

Run all default checks with:

```bash
./.issue_forge/checks/run_changed.sh
```

The script runs:

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

## Development Notes

- Runtime dependencies are limited to NumPy and SciPy at this stage.
- Runtime must not depend on JVM, Jython, Mines JTK, or Gradle.
- Practical equivalence with `reference_osv` is the goal; bitwise equivalence is not.
- `vendor/issue_forge` is an external symlink or bind mount and must not be committed.

## ライセンス

`reference_osv` のライセンスと、Python 再実装としての `pyosv` の配布ライセンスは別途確認・決定してください。

# pyosv

`pyosv` is a Python package scaffold for reimplementing practical OSV (Optimal Surface Voting / Optimal Path Voting) workflows.

The project uses the local `reference_osv/` directory as a read-only reference implementation. That directory is expected to be a bind mount and is not part of this repository.

## Status

This repository has the package scaffold plus the initial DAT I/O and reference dataset metadata. OSV algorithms, interpolation adapters, filters, voting kernels, scanners, and skinning will be implemented in later issues.

## DAT I/O

`pyosv.io.read_dat` and `pyosv.io.write_dat` read and write raw scalar `.dat` files. The array shape convention is:

- 2D arrays: `(n2, n1)`
- 3D arrays: `(n3, n2, n1)`

`reference_osv` `.dat` files are treated as big-endian `float32` by default. Use the reference metadata helpers to keep paths, shapes, and endian settings aligned:

```python
from pyosv.io import read_dat
from pyosv.reference import REFERENCE_DATASETS_2D, resolve_reference_file

dataset = REFERENCE_DATASETS_2D["f3d2d"]
path = resolve_reference_file(dataset, "ft.dat")
ft = read_dat(path, dataset.shape, endian=dataset.endian)
```

The local `reference_osv/` directory is a read-only bind mount and is not committed. Set `PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master` if the mount is not located at `./reference_osv`.

See `docs/dat_io.md` for detailed I/O behavior and reference fixture test policy.

## Equivalence Policy

`pyosv` targets practical equivalence for fault interpretation workflows, not
bit-exact comparison with Java, Jython, or Mines JTK outputs.

Mines JTK `SincInterpolator` behavior is approximated with SciPy interpolation
primitives such as `scipy.ndimage.map_coordinates`. Mines JTK
`RecursiveExponentialFilter` and `RecursiveGaussianFilterP` behavior is
approximated with SciPy Gaussian smoothing. These approximations may differ in
kernel details, boundary handling, and floating-point accumulation order.

The shape convention is 2D `(n2, n1)` and 3D `(n3, n2, n1)`. The
`reference_osv/` directory is a read-only bind mount for reference only; it is
not part of the package and is not distributed.

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

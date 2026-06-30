# pyosv

`pyosv` is a Python package scaffold for reimplementing practical OSV (Optimal Surface Voting / Optimal Path Voting) workflows.

The project uses the local `reference_osv/` directory as a read-only reference implementation. That directory is expected to be a bind mount and is not part of this repository.

## Status

This repository has the package scaffold plus DAT I/O, reference dataset
metadata, the implemented 2D orientation scanner and optimal-path voting
workflow, an approximate 3D orientation scanner, and a synthetic-test-covered
3D voting/thinning MVP. A minimal connected-component skinning layer is also
available for thinned 3D vote volumes.

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

## 2D Voting

Install the package in development mode before running examples:

```bash
python -m pip install -e ".[dev]"
```

The 2D workflow can run from existing reference `ft.dat` and `pt.dat` files.
`reference_osv/` is a read-only bind mount for reference inputs only; it is
optional for normal tests and must not be used for generated outputs.

```python
from pyosv.io import read_dat
from pyosv.reference import REFERENCE_DATASETS_2D, resolve_reference_file
from pyosv.voting2d import OptimalPathVoter

dataset = REFERENCE_DATASETS_2D["f3d2d"]
ft = read_dat(resolve_reference_file(dataset, "ft.dat"), dataset.shape, endian=dataset.endian)
pt = read_dat(resolve_reference_file(dataset, "pt.dat"), dataset.shape, endian=dataset.endian)

voter = OptimalPathVoter(15, 30)
voter.set_strain_max(0.25)
voter.set_path_smoothing(2)
fv, w1, w2 = voter.apply_voting(d=4, fm=0.3, ft=ft, pt=pt)
fvt = voter.thin(fv, w1, w2)
```

`OptimalPathVoter.apply_voting` runs deterministic 2D optimal-path voting over
the selected seeds and returns `(fv, w1, w2)` arrays with the same `(n2, n1)`
shape. `fv` is the normalized float32 vote image, and `w1`/`w2` are the vector
components associated with the strongest local vote at each image sample.

`OptimalPathVoter.thin` keeps local maxima from the vote image along the
returned vector field and returns a thinned float32 vote image with the same
shape. The thinning interpolation uses the package SciPy adapter
(`scipy.ndimage.map_coordinates` through `pyosv.interp.sample2`) rather than
Mines JTK sinc interpolation.

Run the `f3d2d` reference workflow from the command line with an explicit output
directory:

```bash
python examples/run_2d_f3d2d.py --output-dir outputs/f3d2d
```

For other supported 2D reference datasets, use:

```bash
python examples/run_2d_reference.py --dataset campos --output-dir outputs/campos
```

The scanner-to-voting workflow can also run without external data:

```bash
python examples/run_2d_synthetic_scan_vote.py
```

Pass `--output-dir` to that synthetic example only when generated DAT outputs
should be written.

The approximate 3D scanner is documented in `docs/orient3d.md`, with a small
self-contained example:

```bash
python examples/run_3d_synthetic_scan_vote.py
```

Public F3 3D reference-data validation is documented in
`docs/f3d_validation.md`, including the external data layout, smoke checks,
crop validation, and the manual full-volume pipeline.

Reference-like 3D thinning is documented in
`docs/reference_like_thinning.md`. It explains the default `normal` thinning
mode, the opt-in `reference` mode, and the F3 crop, multi-crop, and ablation
validation workflow.

F3 figure-based diagnostics and interpretation order are documented in
`docs/f3d_visual_diagnostics.md`.

Optional static visualization helpers are documented in `docs/visualization.md`.
Install `pyosv[viz]` only when PNG diagnostics such as slice panels, ridge
overlays, MIPs, or value histograms are needed.

Minimal connected-component skinning is documented in `docs/skinning.md`, with
a small self-contained example:

```bash
python examples/run_3d_synthetic_skinning.py
```

The reference example scripts read `ft.dat` and `pt.dat` from `reference_osv/`
or `PYOSV_REFERENCE_OSV`, then write generated files such as `fv_py.dat` and
`fvt_py.dat` under `--output-dir`. Keep that directory outside `reference_osv/`.

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
not part of the package and is not distributed. Default tests skip optional
reference cases clearly when the mount or required `.dat` files are absent.

## Setup

```bash
python -m pip install -e ".[dev]"
```

Visualization dependencies are optional and are not required for the core
package or default tests:

```bash
python -m pip install -e ".[dev,viz]"
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

- Core runtime dependencies are limited to NumPy and SciPy at this stage.
- Runtime must not depend on JVM, Jython, Mines JTK, or Gradle.
- Practical equivalence with `reference_osv` is the goal; bitwise equivalence is not.
- `vendor/issue_forge` is an external symlink or bind mount and must not be committed.

## ライセンス

`reference_osv` のライセンスと、Python 再実装としての `pyosv` の配布ライセンスは別途確認・決定してください。

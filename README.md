# pyosv

`pyosv` is a Python package scaffold for reimplementing reference-first OSV (Optimal Surface Voting / Optimal Path Voting) workflows.

The project uses the local `reference_osv/` directory as a read-only reference implementation. That directory is expected to be a bind mount and is not part of this repository.

## Status

This repository has the package scaffold plus DAT I/O, reference dataset
metadata, the implemented 2D orientation scanner and optimal-path voting
workflow, an approximate 3D orientation scanner, and a synthetic-test-covered
3D voting MVP with current 3D thinning helpers. Reference-like skinning is the
default for thinned 3D vote volumes, with connected-component grouping
available only as an explicit fallback or diagnostic path.

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

## 2D Orientation Scanning And Voting

Install the package in development mode before running examples:

```bash
python -m pip install -e ".[dev]"
```

The normal 2D workflow scans a finite image shaped `(n2, n1)` and passes the
result directly to optimal-path voting. `FaultOrientScanner2.scan()` is the
reference-like default: it approximates the reference rotate, separable smooth,
unrotate, and likelihood-scoring flow with NumPy and SciPy. The output `pt`
angle convention is compatible with `FaultCell2` and `OptimalPathVoter`.
Nonconstant finite inputs are linearly scaled to `[0, 1]` before the
reference-like likelihood score is computed.

```python
from pyosv.orient2d import FaultOrientScanner2
from pyosv.voting2d import OptimalPathVoter

scanner = FaultOrientScanner2(sigma1=2.0)
ft, pt = scanner.scan(-75.0, 75.0, image)

voter = OptimalPathVoter(ru=2, rv=5)
fv, w1, w2 = voter.apply_voting(d=3, fm=0.45, ft=ft, pt=pt)
fvt = voter.thin(fv, w1, w2)
```

`scan_fast()` exposes the older derivative-bank backend only as an explicit
fallback or diagnostic path. Use `scan()` for reference-first examples and
normal scanner-to-voting workflows. `scan_dip()` follows the reference API
shape by running the two dip-angle scan branches and keeping the stronger
sample at each location.

The 2D voting workflow can also run from existing reference `ft.dat` and
`pt.dat` files. `reference_osv/` is a read-only bind mount for reference inputs
only; it is optional for normal tests and must not be used for generated
outputs.

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
should be written. Detailed 2D scanner behavior is documented in
`docs/orient2d.md`.

The approximate 3D scanner is documented in `docs/orient3d.md`, with a small
self-contained example:

```bash
python examples/run_3d_synthetic_scan_vote.py
```

`FaultOrientScanner3.scan()` is the reference-like default and uses the
rotate/shear scanner path with Java-style strike and dip sampling for normal
scanner-to-voting workflows. `scan_reference_like()` remains as an explicit
compatible alias and exposes `backend="directional"` for the previous
fault-parallel smoothing approximation, while `scan_fast()` exposes the older
derivative-bank backend for diagnostics or practical comparisons.

`scan_quality()` selects the quality scanner backend: it uses the same scoring
path with a refined reference-like sampling grid. Scanner backend selection is
independent of the synthetic report's downstream workflow selection, so neither
`scan()` nor `scan_quality()` implicitly selects `--workflow-mode quality`.
See [Scanner backends, workflow modes, thinning modes, and reference
targets](docs/mode_comparison.md) for the canonical distinction.

Publication-facing F3 3D reference comparison uses the complete
`(420, 400, 100)` volume as one evaluation unit. See
[F3 3D Reference Data Validation](docs/f3d_validation.md) for that protocol and
[Scanner, Workflow, Thinning, and F3 Reference Comparison](docs/mode_comparison.md)
for the canonical scanner/workflow matrix and terminology. Existing crop and
multi-crop commands are optional legacy/internal diagnostics and historical
validation paths; crops are not publication samples or statistical replicates.

The current [`examples/run_3d_f3d_full.py`](examples/run_3d_f3d_full.py) command
is a reference-like baseline full-volume scan/vote runner. It does not implement
the quality scanner backend, workflow profiles, or the planned full-volume 2×2
scanner-backend/workflow matrix. That matrix and its comparison runner remain
future work.

Controlled synthetic 3D truth checks are documented in
`docs/synthetic_quality.md`, including the default `minimal` case set and the
`geometry` and `extended` case sets for vertical, dipping, curved, parallel,
crossing, boundary, and weak/noisy faults, voter variants, and skinning
metrics. That document also explains downstream workflow modes, independently
of scanner backend selection: `reference` selects the default reference
workflow, while `quality` defaults voter
thinning to the truth-quality-favored `hybrid_v2` path with support-aware surface
voting inactive, the quality skinner v2 profile, and the empty-primary boundary
skinner fallback. Degraded-primary fallback variants remain diagnostic after
the legacy 49^3 scanner-inclusive boundary benchmark using the quality scanner
backend: v2
over-includes fallback components, filtered v3 did not reach the boundary skin
F1 promotion target or the non-boundary regression tolerances, and skeletonized
v4 still missed the scanner boundary skin target while regressing oracle
boundary skin.

The separate reference-like scanner backend 49^3 scanner-thinning policy
evaluation has
been completed. `quality_reference_like_scanner_thin_normal_v1` passed the
`scanner-boundary-reference-like` gate against
`quality_reference_like_scanner_thin_reference_v1`, with all 14 required rows,
no missing evidence rows, and a passing configuration contract. Compact values
and hashes are recorded in
`tests/fixtures/synthetic_quality_refactor/reference_like_scanner_thinning_49_evidence.json`.
This is synthetic candidate evidence only. A historical 64^3-by-3 F3 crop
diagnostic later failed one conservative external-smoke check: crop 1 worsened
public-FVT sparse-distance p95 by `6.193637` samples against the allowed `5.0`.
The other seven historical diagnostic checks passed, including shared scanning,
finite/nonempty stages, density, edge, crop stability, and configuration
contracts. The prerequisite large-crop diagnostic was therefore not run and
human geological review remains pending. This failed crop check is preserved as
historical diagnostic evidence, not a current publication gate or a set of
statistical replicates. The quality-workflow scanner-thinning default and all
public defaults remain unchanged. Compact failed-run evidence is recorded in
`tests/fixtures/f3d_scanner_thinning_policy/quality_reference_like_normal_v1_evidence.json`.

The legacy quality scanner backend promotion-candidate flow for
`boundary_edge_thin_v1`, `boundary_seed_retention_v1`, and
`quality_boundary_skinner_fallback_v5` is reported with
`scripts/compare_quality_reports.py` or
`scripts/check_synthetic_quality_promotion_gate.py`; no new 49^3
`promotion_candidates_49` result is recorded for that separate flow.
The diagnostic workflow keeps reference-workflow defaults while enabling
reference-vs-normal thinning diagnostics.
The current quality workflow profile and guardrails are summarized in
`docs/quality_mode.md`. A typical extended report run is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 33,33,33 \
  --variant-preset quality-matrix \
  --output-dir outputs/3d/synthetic_quality/extended_001 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

The default report path remains oracle input. To compare the downstream oracle
upper-bound path with the scanner-inclusive end-to-end path on the same
synthetic truth geometry, use `--input-mode both`:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set geometry \
  --shape 33,33,33 \
  --input-mode both \
  --output-dir outputs/3d/synthetic_quality/scanner_inclusive_001 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

Reference-like 3D thinning is documented in
`docs/reference_like_thinning.md`. `FaultOrientScanner3.thin()` now defaults
to reference-like strike-binned thinning with scanner edge-effect removal; pass
`remove_edge_effects=False` only for diagnostics, or `mode="normal"` for the
legacy fault-normal scanner path. `OptimalSurfaceVoter.thin()` also defaults
to reference-like strike-binned thinning, with voter-specific retained-sample
reinforcement and no scanner edge-effect cleanup. Use `mode="normal"` for the
legacy fault-normal voter path. Voter diagnostics can also use `mode="hybrid"`,
`mode="hybrid_v2"`, or `mode="normal_plateau"` to compare reference-like,
fault-normal, and plateau-aware thinning behavior without changing defaults.

Backward-compatible 3D thinning calls are explicit:

```python
fet, fpt, ftt = scanner.thin(ft, pt, tt, mode="normal")
fvt = voter.thin(fv, vp, vt, mode="normal")

# Diagnostic opt-out for scanner edge cleanup:
fet, fpt, ftt = scanner.thin(ft, pt, tt, remove_edge_effects=False)
```

`OptimalSurfaceVoter` smooths extracted local surfaces before recomputing vote
strike/dip, matching the reference-first surface-orientation path. Use
`set_surface_orientation_smoothing(0.0)` only for diagnostics that require the
older raw-surface behavior.

`OptimalSurfaceVoter.apply_voting()` now uses reference-style final vote-map
normalization by default: subtract the global minimum, divide by the global
maximum when it is positive, then apply `1 - (1 - x) ** 8` without final
vote-map smoothing. This is separate from input fault-likelihood smoothing and
surface-orientation smoothing. Reference-first workflows should leave it
unset. Use `set_final_normalization_smoothing(1.0)` to opt into the older
practical smoothed final-normalization behavior, or pass
`--final-normalization-smoothing 1.0` in the F3 validation examples.

Migration note: code that relied on the older derivative-bank scanner should
call `scan_fast()` explicitly. Code that relied on old 3D scanner
fault-normal thinning should pass `mode="normal"` explicitly, or use
`--scanner-thin-mode normal` in F3 validation examples. Code that relied on
old 3D voter fault-normal thinning should pass `mode="normal"` explicitly, or
use `--voter-thin-mode normal` in F3 validation examples. Code that relied on
old final vote-map smoothing should call
`set_final_normalization_smoothing(1.0)` explicitly, or use
`--final-normalization-smoothing 1.0` in F3 validation examples. Code following
the reference-first default should not configure final normalization smoothing.

F3 figure-based diagnostics and interpretation order are documented in
`docs/f3d_visual_diagnostics.md`.

Optional static visualization helpers are documented in `docs/visualization.md`.
Install `pyosv[viz]` only when PNG diagnostics such as slice panels, ridge
overlays, MIPs, or value histograms are needed.

Reference-like skinning is documented in `docs/skinning.md`, with a small
self-contained example:

```bash
python examples/run_3d_synthetic_skinning.py
```

Normal skinning workflows should use `FaultSkinner()` or module-level
`pyosv.skinner.find_skins(...)`, both of which use the reference-like backend.
Use `FaultSkinner(method="connected_component")` or
`pyosv.skinner.find_connected_component_skins(...)` only for fallback or
diagnostic connected-component grouping.

The reference example scripts read `ft.dat` and `pt.dat` from `reference_osv/`
or `PYOSV_REFERENCE_OSV`, then write generated files such as `fv_py.dat` and
`fvt_py.dat` under `--output-dir`. Keep that directory outside `reference_osv/`.

## Equivalence Policy

`pyosv` follows a reference-first policy for fault interpretation workflows:
Python implementations should preserve the reference control flow and geometric
semantics where practical. Bit-exact comparison with Java, Jython, or Mines JTK
outputs is not a goal.

Mines JTK `SincInterpolator` behavior is approximated with SciPy interpolation
primitives such as `scipy.ndimage.map_coordinates`. Mines JTK
`RecursiveExponentialFilter` and `RecursiveGaussianFilterP` behavior is
approximated with SciPy Gaussian smoothing. These approximations may differ in
kernel details, boundary handling, and floating-point accumulation order.
Faster, simpler, or more robust variants should be explicit opt-in modes rather
than silent replacements for reference-like defaults.

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
- Reference-first alignment with `reference_osv` is the goal; bitwise equivalence is not.
- `vendor/issue_forge` is an external symlink or bind mount and must not be committed.

## ライセンス

`reference_osv` のライセンスと、Python 再実装としての `pyosv` の配布ライセンスは別途確認・決定してください。

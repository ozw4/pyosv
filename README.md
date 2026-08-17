# pyosv

`pyosv` is a Python implementation of reference-first Optimal Surface Voting
and Optimal Path Voting workflows for seismic fault interpretation.

The external `reference_osv/` directory is used only as a read-only comparison
reference. It is not part of the package, is not committed, and is not a runtime
dependency.

## Capabilities

The package provides:

- raw scalar DAT I/O and reference dataset metadata;
- 2D fault-orientation scanning, optimal-path voting, and thinning;
- 3D fault-orientation scanning, optimal-surface voting, scanner/voter thinning,
  and fault skinning;
- controlled synthetic quality evaluation and canonical scanner/workflow mode
  comparison;
- full-volume F3 mode comparison and validated publication-bundle generation;
- typed numerical contracts for fault apparent-shift estimation across a known
  fault surface.

## Installation

Python 3.10 or later is required. Core dependencies are NumPy 1.x, SciPy, and
threadpoolctl.

```bash
python -m pip install -e .
```

Install development, optional Numba acceleration, or visualization support as
needed:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[accel]"
python -m pip install -e ".[dev,accel,viz]"
```

Verify the package import:

```bash
python -c "import pyosv; print(pyosv.__version__)"
```

PyOSV supports the NumPy 1.x line through the dependency bound `numpy<2`.
Byte-level fixtures and numerical regression contracts must be evaluated in a
supported NumPy 1.x environment.

## Data and numerical conventions

Global arrays use these shapes:

- 2D: `(n2, n1)`, indexed as `array[i2, i1]`;
- 3D: `(n3, n2, n1)`, indexed as `array[i3, i2, i1]`.

Vector components retain OSV coordinate order: `(x1, x2)` in 2D and
`(x1, x2, x3)` in 3D. Numerical arrays are generally represented as
`np.float32`.

Reference DAT files are interpreted as big-endian `float32` unless their
metadata specifies otherwise. Generated outputs must be written outside the
external reference directory.

```python
from pyosv.io import read_dat
from pyosv.reference import REFERENCE_DATASETS_2D, resolve_reference_file

dataset = REFERENCE_DATASETS_2D["f3d2d"]
path = resolve_reference_file(dataset, "ft.dat")
ft = read_dat(path, dataset.shape, endian=dataset.endian)
```

Set `PYOSV_REFERENCE_OSV` when the read-only reference checkout is not mounted
at `./reference_osv`:

```bash
export PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master
```

See [DAT I/O](docs/dat_io.md) for storage and metadata details.

## 2D orientation scanning and voting

`FaultOrientScanner2.scan()` is the reference-like 2D scanner. It approximates
the reference rotate, separable-smooth, unrotate, and likelihood-scoring flow
with NumPy and SciPy. `scan_fast()` selects the derivative-bank backend, and
`scan_dip()` evaluates the two reference-style dip-angle branches.

```python
from pyosv.orient2d import FaultOrientScanner2
from pyosv.voting2d import OptimalPathVoter

scanner = FaultOrientScanner2(sigma1=2.0)
ft, pt = scanner.scan(-75.0, 75.0, image)

voter = OptimalPathVoter(ru=2, rv=5)
fv, w1, w2 = voter.apply_voting(d=3, fm=0.45, ft=ft, pt=pt)
fvt = voter.thin(fv, w1, w2)
```

`apply_voting()` returns normalized `float32` vote evidence and its strongest
local vector field. `thin()` retains strict local maxima along that field using
the SciPy-backed interpolation adapter.

Reference-input and self-contained examples are available through:

```bash
python examples/run_2d_f3d2d.py --output-dir outputs/f3d2d
python examples/run_2d_reference.py --dataset campos --output-dir outputs/campos
python examples/run_2d_synthetic_scan_vote.py
```

Detailed contracts are documented in [2D Orientation Scanning](docs/orient2d.md)
and [2D Voter Reference Mapping](docs/reference_mapping_voting2d.md).

## 3D scanning, voting, thinning, and skinning

`FaultOrientScanner3.scan()` uses the reference-like rotate/shear scanner with
Java-style strike and dip sampling. `scan_reference_like()` exposes the same
scanner explicitly and permits the `directional` scoring backend.
`scan_quality()` refines the reference-like orientation grid.
`scan_fast()` selects the derivative-bank backend.

Scanner backend selection is independent of downstream workflow selection. A
quality scanner does not select the quality workflow, and a quality workflow
does not select a scanner backend.

```python
from pyosv.orient3d import FaultOrientScanner3
from pyosv.skinner import FaultSkinner
from pyosv.voting3d import OptimalSurfaceVoter

scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
ft, pt, tt = scanner.scan(
    phi_min=0.0,
    phi_max=90.0,
    theta_min=45.0,
    theta_max=90.0,
    g=image,
)

fet, fpt, ftt = scanner.thin(ft, pt, tt)

voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=fet, pt=fpt, tt=ftt)
fvt = voter.thin(fv, vp, vt)

skinner = FaultSkinner(min_likelihood=0.7, min_skin_size=20)
skins = skinner.find_skins(
    fvt,
    vp,
    vt,
    ep=fvt,
    ft=fvt,
    pt=vp,
    tt=vt,
)
```

Scanner thinning defaults to reference-like strike-binned suppression with
scanner edge-effect removal. Voter thinning defaults to reference-like
strike-binned suppression with voter-specific retained-sample reinforcement and
without scanner edge cleanup. `mode="normal"` selects fault-normal thinning.
Voter diagnostics also expose `hybrid`, `hybrid_v2`, and `normal_plateau`.

```python
fet, fpt, ftt = scanner.thin(ft, pt, tt, mode="normal")
fvt = voter.thin(fv, vp, vt, mode="normal")
```

`OptimalSurfaceVoter.apply_voting()` normalizes accumulated vote evidence by
subtracting its minimum, dividing by its positive maximum, and applying
`1 - (1 - x) ** 8`. Final vote-map smoothing is disabled unless
`set_final_normalization_smoothing(...)` is called explicitly. Extracted local
surfaces are smoothed before vote strike and dip are recomputed; use
`set_surface_orientation_smoothing(0.0)` only when unsmoothed surface
orientation is required.

`FaultSkinner()` uses the reference-like skinning backend. `method="quality"`
selects the quality skinning profile. `method="connected_component"` and
`find_connected_component_skins(...)` select connected-component grouping
explicitly.

Self-contained 3D examples are available through:

```bash
python examples/run_3d_synthetic_scan_vote.py
python examples/run_3d_synthetic_skinning.py
```

See [3D Orientation Scanning](docs/orient3d.md),
[3D Voting Conventions](docs/3d_voting.md),
[Reference-Like 3D Thinning](docs/reference_like_thinning.md), and
[Skinning](docs/skinning.md).

## Controlled synthetic evaluation

Controlled synthetic cases provide independent truth geometry for evaluating
scanner, voting, thinning, and skinning behavior. The input modes are:

- `oracle`: evaluates downstream stages from truth-derived attributes;
- `scanner`: evaluates scanner and downstream stages end to end;
- `both`: evaluates both paths on the same truth geometry.

A diagnostic quality matrix can be generated with:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 33,33,33 \
  --variant-preset quality-matrix \
  --input-mode both \
  --workflow-mode diagnostic \
  --output-dir outputs/3d/synthetic_quality/quality_matrix_001 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

The canonical scanner-backend × workflow comparison has a separate command and
bundle contract. A small execution check is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_mode_comparison.py \
  --case-set minimal \
  --shape 9,9,9 \
  --skip-skinning \
  --output-dir outputs/3d/synthetic_mode_comparison/smoke_9
```

See [Controlled Synthetic Quality](docs/synthetic_quality.md) and
[Synthetic Mode Comparison](docs/synthetic_mode_comparison.md) for case sets,
metrics, artifact schemas, validation, and runtime attribution.

## F3 full-volume mode comparison

Publication-facing F3 comparison uses the complete `(420, 400, 100)` volume as
one evaluation unit. The public `fl.dat`, `fv.dat`, and `fvt.dat` files are
comparison targets, not independent geological truth. Crops and regional
partitions are diagnostics within the same volume and are not statistical
replicates.

The F3 data root must contain the official big-endian `float32` volumes. Run the
canonical four-cell matrix with the publication runtime controls set before
Python starts:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
python -m pyosv.cli.f3d_mode_comparison \
  --output-dir outputs/3d/f3d/mode_comparison_001
```

The matrix contains `RL-REF`, `RL-QUAL`, `Q-REF`, and `Q-QUAL`. Scanner output
and scanner thinning are shared between workflows for the same scanner backend.
Official bundles require enabled Numba JIT and a publication-valid recorded
runtime identity.

Validate an existing bundle without recomputation:

```bash
python -m pyosv.cli.f3d_mode_comparison \
  --output-dir outputs/3d/f3d/mode_comparison_001 \
  --validate-only
```

Deep validation requires a matching publication runtime and rechecks persisted
numerical evidence:

```bash
python -m pyosv.cli.f3d_mode_comparison \
  --output-dir outputs/3d/f3d/mode_comparison_001 \
  --validate-only \
  --deep-validate
```

Resume a matching interrupted workspace with:

```bash
python -m pyosv.cli.f3d_mode_comparison \
  --data-root /path/to/external/reference_osv \
  --output-dir outputs/3d/f3d/mode_comparison_001 \
  --resume
```

See [F3 3D Reference Data Validation](docs/f3d_validation.md) for dataset
identity, runtime identity, stage reuse, artifact validation, metrics, and
regional diagnostics. Figure interpretation rules are documented in
[F3 Visual Diagnostics](docs/f3d_visual_diagnostics.md).

## Mode comparison publication bundle

The publication command derives tables, figures, and a report from completed
Synthetic and F3 source bundles. It does not rerun scanner, voting, thinning, or
skinning stages. Synthetic results retain known-truth semantics; F3 results
retain public-reference-agreement semantics.

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --synthetic-bundle <completed-synthetic-bundle> \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --environment-lock uv.lock \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

Validate a completed publication directory from its recorded files:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --validate-only \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

`publication_manifest.json` defines artifact integrity and provenance identity.
It is not a cryptographic signature or an end-to-end experiment replay. See
[Mode Comparison Publication Bundle](docs/mode_comparison_publication.md).

## F3 compact publication

Generate the focused `PUBLIC-REF` versus `Q-QUAL` F3 publication from a
completed F3 source bundle:

```bash
PYTHONPATH=src python -m pyosv.cli.f3_compact_publication \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --environment-lock <environment-lock> \
  --output-dir <new-output-dir>
```

The command requires the eight publication environment controls documented in
[F3 Compact Publication](docs/f3_compact_publication.md), which also defines
the four time slices, four inline sections, six atlases, summary, output layout,
and validate-only command. Attribute display thresholds use half the source
value for every stage; only `fvt` attribute panels receive the fixed one-pixel
red halo.

## Fault-warping numerical contract

`pyosv.fault_warping` defines typed, Atlas-independent input, configuration,
result, and estimator protocol contracts for apparent sample-axis shift
estimation across a known fault surface. The package contains no concrete
estimator, artifact writer, workflow integration, or physical slip conversion.

See [Fault-warping contract](docs/fault_warping.md) for coordinate, side,
validity, topology, slope, and result semantics.

## Reference alignment policy

PyOSV preserves reference control flow and geometric semantics where practical.
Bit-exact output matching with Java, Jython, or Mines JTK is not required.

Mines JTK interpolation and recursive filters are represented by SciPy-backed
interpolation and Gaussian or separable smoothing. Differences in kernels,
boundary behavior, and floating-point accumulation are expected and are
assessed with deterministic Python regression tests and practical localization
or agreement metrics.

The external `reference_osv/` directory remains read-only. Default tests do not
require it or the F3 data root. See
[Reference-First Equivalence Policy](docs/equivalence_policy.md) and
[3D Reference Alignment](docs/reference_alignment_3d.md).

## Development checks

Run the repository check wrapper:

```bash
./.issue_forge/checks/run_changed.sh
```

The underlying default checks are:

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

Core numerical runtime code must not depend on JVM, Jython, Mines JTK, Gradle,
Atlas workflow packages, viewers, artifact publication, or job-management
systems.

## Documentation index

- [Architecture](docs/architecture.md)
- [DAT I/O](docs/dat_io.md)
- [2D Orientation Scanning](docs/orient2d.md)
- [2D Voter Reference Mapping](docs/reference_mapping_voting2d.md)
- [3D Orientation Scanning](docs/orient3d.md)
- [3D Scanner Reference Mapping](docs/reference_mapping_orient3d.md)
- [3D Voting Conventions](docs/3d_voting.md)
- [3D Voter Reference Mapping](docs/reference_mapping_voting3d.md)
- [Reference-Like 3D Thinning](docs/reference_like_thinning.md)
- [Skinning](docs/skinning.md)
- [Mode Comparison Contract](docs/mode_comparison.md)
- [Controlled Synthetic Quality](docs/synthetic_quality.md)
- [Synthetic Mode Comparison](docs/synthetic_mode_comparison.md)
- [F3 3D Reference Data Validation](docs/f3d_validation.md)
- [F3 Visual Diagnostics](docs/f3d_visual_diagnostics.md)
- [Mode Comparison Publication Bundle](docs/mode_comparison_publication.md)
- [F3 Compact Publication](docs/f3_compact_publication.md)
- [Fault-warping contract](docs/fault_warping.md)
- [Reference-First Equivalence Policy](docs/equivalence_policy.md)

## License

The distribution license for the Python implementation and the license of the
external `reference_osv` implementation must be evaluated independently.

# F3 3D Reference Data Validation

This workflow validates the current Python 3D scanner and optimal-surface
voting pipeline against the public F3 reference volumes. The F3 `.dat` files
are external data, not repository files. Do not copy them into git; repository
`.gitignore` rules ignore generated `.dat` files and `outputs/`.

## Data Layout

Use an external data root such as the local shared copy:

```text
/home/dcuser/public_data/field/F3/reference_osv/
  ep.dat
  fl.dat
  fv.dat
  fvt.dat
  xs.dat
```

Point repository commands at that root with:

```bash
export PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv
```

The expected format for each file is:

```text
shape = (420, 400, 100)  # (n3, n2, n1)
dtype = big-endian float32
expected bytes per file = 67,200,000
```

Files:

- `xs.dat`: seismic input.
- `ep.dat`: input planarity; start OSV validation from this file.
- `fl.dat`: reference fault likelihood.
- `fv.dat`: reference OSV fault volume.
- `fvt.dat`: reference thinned OSV fault volume.

The current OSV validation starts from `ep.dat`; reproducing `xs.dat -> ep.dat`
is out of scope for this workflow.

## Fast Smoke Validation

Check that the external files are present, have the expected byte size, and can
be read as `(420, 400, 100)` big-endian float32 volumes:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python -m pytest -q tests/test_f3d_reference_data.py -s
```

Generate a summary report for the reference volumes:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/report_3d_f3d_reference.py --pretty
```

On the local shared F3 copy, that report shows `fv.dat` max around `1.0`,
`fvt.dat` max around `0.99`, and `fvt.dat` much sparser than `fv.dat`. Treat
the report output as the source of truth for the exact local values.

## Small Crop Practical-Equivalence Validation

Run one deterministic crop validation and write metrics under `outputs/`:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/crop_001
```

Run the opt-in pytest wrapper for the crop pipeline:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
PYOSV_RUN_F3D_CROP_PIPELINE=1 \
python -m pytest -q tests/test_f3d_reference_crop_pipeline.py -s
```

By default, the crop example writes `metrics.json` only. Add `--save-volumes`
when crop-level Python `.dat` outputs are needed.

Default crop selection is margin-aware: when a crop shape is used to pick
centers, candidates too close to the volume boundary are skipped instead of
being silently shifted by `crop_slices()`. Pass `--center i3,i2,i1` to validate
a specific manual crop.

## Reference-Like Thinning Validation

The 3D scanner and voter thinning steps support two modes:

- `normal`: existing pyosv behavior. It uses 3D normal-vector interpolation for
  non-maximum suppression and remains the default.
- `reference`: opt-in reference-like behavior. It smooths the comparison volume,
  bins samples by strike angle, and compares local maxima in the `i2-i3` plane.

Use `reference` mode only when running diagnostics against reference-style
outputs. It is closer to the Java thinning workflow, but it is still Pythonic
and not bit-exact Mines JTK. Existing pyosv behavior and tests remain unchanged
unless the mode is selected explicitly. See `docs/reference_like_thinning.md`
for the API-level details.

Run one crop with reference-like scanner and voter thinning:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --output-dir outputs/3d/f3d/crop_reference_thin_001 \
  --scanner-thin-mode reference \
  --voter-thin-mode reference \
  --reference-thin-sigma 1.0 \
  --pretty \
  --save-figures
```

Run a multi-crop report with reference-like thinning:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/report_3d_f3d_multicrop.py \
  --output-json outputs/3d/f3d/multicrop_reference_thin_001/metrics.json \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --scanner-thin-mode reference \
  --voter-thin-mode reference \
  --reference-thin-sigma 1.0 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

Run the thinning ablation report to compare current/current, mixed, and
reference/reference thinning cases:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/report_3d_f3d_thinning_ablation.py \
  --output-json outputs/3d/f3d/thinning_ablation_001/metrics.json \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

Do not document generated report outputs as committed artifacts. Generated
JSON, PNG, markdown, and `.dat` outputs belong under `outputs/` or another
ignored working directory.

## Large Crop Manual Validation

The `(128, 128, 100)` crop preset is an explicit long-running manual validation,
not part of regular checks or CI. It runs the scanner, thinning, voting, and
voter thinning on a substantially larger crop:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --large-crop-preset \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/large_crop_001
```

The opt-in pytest wrapper is also manual:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
PYOSV_RUN_F3D_LARGE_CROP_PIPELINE=1 \
python -m pytest -q tests/test_f3d_large_crop_validation.py -s
```

If a large crop run is interrupted, rerun the command into a fresh output
directory. The crop validation script does not currently reuse partial scanner
or voting stages.

## Full F3 Pipeline

The full F3 run is manual and can be slow because it scans and votes over the
entire `(420, 400, 100)` volume. It is separate from the smoke, small-crop, and
large-crop validations. Run it explicitly with an output directory:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_full.py \
  --output-dir outputs/3d/f3d/full_001
```

The full script writes `run_config.json`, `metrics.json`, and generated Python
volumes such as `fv_py.dat` and `fvt_py.dat` under `--output-dir`. Use
`--skip-save-intermediates` when only final vote volumes and reports are needed.
Use `--reuse-existing` to reuse complete stage outputs already present in that
directory; incomplete stage output sets are rejected with a clear error.

## Output Policy

- Never write into `PYOSV_F3D_DATA_ROOT`.
- Write generated reports and volumes under `outputs/` or another ignored
  working directory.
- Do not commit reference `.dat` files or generated `.dat` outputs.

The F3 scripts reject output paths inside the data root.

## Interpretation

`pyosv` uses practical approximations for Mines JTK interpolation and filtering.
For example, interpolation is based on SciPy primitives rather than JTK sinc
interpolation, and smoothing may use SciPy Gaussian-style filters rather than
JTK recursive filters.

Reference comparisons should use practical metrics and visual review:

- finite-value summaries
- normalized correlation
- top-percentile overlap
- sparsity checks
- visual checks of fault ridges and thinned volumes

For an operational figure-first workflow, including crop PNGs, ridge overlays,
MIPs, histograms, and multi-crop markdown indexes, see
`docs/f3d_visual_diagnostics.md`.

For reference-like thinning runs, the first expected improvements are not
necessarily high voxel-wise correlation. The main checks are:

- `fvt` `nonzero_fraction` moving closer to the reference.
- `buffered_ridge_overlap.interior.fvt.buffered_f1` improving.
- sparse ridge distance medians decreasing.
- ridge overlay figures showing fewer far-away candidate-only ridges.
- exact overlap remaining interpretable even when it is low for sparse ridges.

Previous current/current baseline context:

```text
normalized_correlation.interior.fvt.mean ~= 0.224
buffered_ridge_overlap.interior.fvt.buffered_f1.mean ~= 0.075
exact fvt ridge overlap F1/Jaccard = 0.0
```

Do not claim success until actual ablation results are generated and reviewed.

No bitwise equality with Java, Jython, or Mines JTK output is expected.

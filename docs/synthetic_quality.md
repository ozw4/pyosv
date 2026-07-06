# Controlled Synthetic Quality

Controlled synthetic cases measure truth quality, not agreement with the
reference implementation. They are for checking whether a known fault geometry
is recovered by the Python workflow without scanner or external-data
confounds.

These validation modes answer different questions:

```text
reference comparison: is the implementation close to the original?
controlled synthetic: is the result correct against known truth?
F3 visual/multicrop: does the workflow avoid obvious failures on real data?
```

## Current Scope

The controlled synthetic API includes:

- `SyntheticPlaneSpec`
- `Synthetic3DCase`
- `generate_single_plane_case`
- `make_single_vertical_plane_case`
- `ft` / `pt` / `tt` oracle attributes
- top-k / truth-count masks
- buffered surface overlap
- surface distance metrics
- masked orientation error
- minimal oracle pipeline smoke test
- `examples/report_3d_synthetic_quality.py`

The current report CLI includes the `minimal` case set, which contains only
`single_vertical_plane`. It runs oracle `ft` / `pt` / `tt` attributes through
`OptimalSurfaceVoter`, applies thinning, and writes truth-quality report files.

The current scope does not include:

- extended cases: dipping, curved, crossing, boundary, weak noisy
- skin topology metrics
- synthetic seismic generation
- scanner-inclusive synthetic path
- FaultSeg3D loader

## Shape And Convention

Controlled synthetic 3D arrays use the repository 3D shape convention:
`(n3, n2, n1)`.

Coordinates are expressed as `(x1, x2, x3)`, matching the OSV geometry helpers.
Orientation follows the same convention as `pyosv.geometry`, `FaultCell`, and
`OptimalSurfaceVoter`.

The public API is exposed through module imports such as `pyosv.synthetic3d`
and `pyosv.synthetic_metrics`; these helpers are not re-exported from
`pyosv.__init__`.

## Minimal Usage

```python
from pyosv.synthetic3d import make_single_vertical_plane_case
from pyosv.synthetic_metrics import top_truth_count_mask, buffered_surface_overlap

case = make_single_vertical_plane_case(shape=(33, 33, 33))
mask = top_truth_count_mask(case.ft_oracle, case.truth_fault_mask)
metrics = buffered_surface_overlap(mask, case.truth_fault_mask, radius=2.0)
```

## Report CLI

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set minimal \
  --output-dir outputs/3d/synthetic_quality/minimal_001 \
  --variants current_default \
  --truth-surface-half-width 0.5 \
  --buffer-radius 2.0 \
  --pretty
```

The CLI writes these files under `--output-dir`:

```text
metrics.json
summary.csv
```

With optional visual-report outputs enabled, each case also gets a case
directory. For the `minimal` case set this is:

```text
single_vertical_plane/
  truth_fault_mask.dat
  truth_distance.dat
  truth_strike.dat
  truth_dip.dat
  ft_oracle.dat
  pt_oracle.dat
  tt_oracle.dat
  fv_py.dat
  vp_py.dat
  vt_py.dat
  fvt_py.dat
  figures/
    ft_oracle_i3_center.png
    fv_py_i3_center.png
    fvt_py_i3_center.png
    truth_vs_fvt_overlay_i3_center.png
```

The stable minimum JSON contract is:

```json
{
  "format_version": 1,
  "config": {
    "case_set": "minimal",
    "shape": [33, 33, 33],
    "variants": ["current_default"]
  },
  "cases": [
    {
      "case_id": "single_vertical_plane",
      "shape": [33, 33, 33],
      "truth": {
        "fault_voxel_count": 2277,
        "surface_voxel_count": 1089
      },
      "variants": {
        "current_default": {
          "pyosv": {
            "fv": {},
            "fvt": {}
          },
          "quality": {
            "fv_top_truth_count": {
              "buffered_overlap_radius2": {},
              "surface_distance": {}
            },
            "fvt_top_truth_count": {
              "buffered_overlap_radius2": {},
              "surface_distance": {},
              "orientation_error": {}
            }
          }
        }
      }
    }
  ]
}
```

Each case stores per-variant metrics under `cases[].variants`. For backward
compatibility, `current_default` is also duplicated at the case top level when
that variant is present. `quality.*.buffered_overlap_radius2` uses the wider
`truth_fault_mask` band as the truth target. `quality.*.surface_distance` uses
the thin truth surface mask defined by
`abs(truth_distance) <= --truth-surface-half-width`.

`--variants` accepts a comma-separated list:

```text
current_default
no_surface_orientation_smoothing
final_norm_smoothing_1
voter_thin_normal
```

The default is `current_default`. Diagnostic variants do not add pass/fail
judgments; they make the same truth metrics comparable across voter settings.
`summary.csv` writes one row per `(case_id, variant)` and includes the variant
column, buffered F1, candidate-to-truth p95 distance, and fvt median
orientation error columns. `--save-volumes` writes float32 big-endian DAT
volumes under each case directory, with `truth_fault_mask.dat` stored as 0/1
float32 values. With more than one variant, volumes and figures are written
under `case_id/variant/`. `--save-figures` writes static center-slice PNGs.
`--write-markdown-index` writes `visual_report.md` with relative links to the
case figures.

## Test Commands

```bash
PYTHONPATH=src python -m pytest -q tests/test_synthetic3d.py tests/test_synthetic_metrics.py tests/test_synthetic_oracle_pipeline.py tests/test_report_3d_synthetic_quality.py
```

The broader synthetic acceptance checks are:

```bash
PYTHONPATH=src python -m pytest -q tests/test_voting3d.py
PYTHONPATH=src python -m ruff check src tests examples
PYTHONPATH=src python -m ruff format --check src tests examples
```

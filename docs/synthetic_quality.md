# Controlled Synthetic Quality

Controlled synthetic cases measure truth quality, not reference agreement. They
are for checking whether a known fault geometry is recovered by the Python
workflow without scanner or external-data confounds.

These validation modes answer different questions:

```text
reference comparison: is the implementation close to the original?
controlled synthetic: is the result correct against known truth?
F3 visual/multicrop: does the workflow avoid obvious failures on real data?
```

## Current Scope

The controlled synthetic API includes:

- `SyntheticPlaneSpec`
- `SyntheticCurvedSurfaceSpec`
- `Synthetic3DCase`
- `generate_single_plane_case`
- `generate_curved_surface_case`
- `make_single_dipping_plane_case`
- `make_single_vertical_plane_case`
- `make_curved_surface_case`
- `ft` / `pt` / `tt` oracle attributes
- top-k / truth-count masks
- buffered surface overlap
- surface distance metrics
- masked orientation error
- skin metrics, including skin topology metrics
- minimal oracle pipeline smoke test
- `examples/report_3d_synthetic_quality.py`

The current report CLI includes these case sets:

- `minimal`: the default PR2-compatible smoke set containing only
  `single_vertical_plane`.
- `geometry`: the PR3 geometry set containing `single_vertical_plane`,
  `single_dipping_plane`, and `curved_surface`.

The individual cases are:

- `single_vertical_plane`: a planar fault centered near constant `x2`, with
  constant strike and dip truth orientation.
- `single_dipping_plane`: a planar fault generated from a strike/dip normal,
  with constant truth orientation and nonzero dip geometry.
- `curved_surface`: an analytic surface whose `x1` position varies with `x2`
  and `x3`; truth strike and dip vary spatially with the local normal.

The controlled synthetic tests cover the oracle `ft` / `pt` / `tt` path for
vertical and dipping single-plane cases and the analytic curved surface: they
run those controlled attributes through `OptimalSurfaceVoter`, apply thinning,
and check truth-quality metrics.

The current scope does not include:

- extended cases: crossing, boundary, weak noisy
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
  --case-set geometry \
  --shape 33,33,33 \
  --variants current_default,no_surface_orientation_smoothing,final_norm_smoothing_1,voter_thin_normal \
  --output-dir outputs/3d/synthetic_quality/geometry_001 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

The CLI writes these files under `--output-dir`:

```text
metrics.json
summary.csv
visual_report.md  # only with --write-markdown-index
```

With optional visual outputs enabled, each case also gets a case directory. For
the `minimal` case set this is:

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
    "variants": ["current_default"],
    "skinning": {
      "enabled": true,
      "min_likelihood": 0.5,
      "min_skin_size": 1,
      "d": 1,
      "ru": 10,
      "rv": null,
      "rw": null,
      "max_steps": 10,
      "du": 5.0,
      "max_delta_strike": 30.0,
      "reskin": true,
      "small_skin_size": 10
    }
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
            "fvt": {},
            "skins": {
              "skin_count": 1,
              "cell_count": 1089,
              "unique_cell_count": 1089,
              "largest_skin_size": 1089
            }
          },
          "skinning": {
            "enabled": true
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
            },
            "skin": {
              "topology": {},
              "buffered_overlap_radius2": {},
              "surface_distance": {},
              "orientation_error": {}
            }
          }
        }
      },
      "variant_comparison": {
        "baseline_variant": "current_default",
        "variants": {
          "current_default": {
            "fvt_buffered_f1_r2_delta_vs_current": 0.0,
            "fvt_candidate_to_truth_p95_delta_vs_current": 0.0,
            "fvt_strike_median_error_delta_vs_current": 0.0,
            "fvt_dip_median_error_delta_vs_current": 0.0,
            "fv_buffered_f1_r2_delta_vs_current": 0.0
          }
        }
      }
    }
  ]
}
```

Each case stores per-variant metrics under `cases[].variants`. For backward
compatibility, `current_default` is also duplicated at the case top level when
that variant is present. `cases[].variant_comparison` stores per-variant deltas
against `current_default` when that baseline variant is present; when it is not
present, `baseline_variant` is `null` and the comparison map is empty.
`quality.*.buffered_overlap_radius2` uses the wider `truth_fault_mask` band as
the truth target. `quality.*.surface_distance` uses the thin truth surface mask
defined by
`abs(truth_distance) <= --truth-surface-half-width`.
`quality.skin.buffered_overlap_radius2` and `quality.skin.surface_distance` use
the same truth targets as the `fv` and `fvt` truth-count metrics. With
`--skip-skinning`, each variant stores `"skinning": {"enabled": false}`,
`pyosv.skins` is a zero-count topology summary, and `quality.skin` is `null`.

The synthetic report default skinning configuration is intentionally small for
controlled synthetic volumes: `--skinner-ru 10`, `--skinner-rv none`,
`--skinner-rw none`, and `--skinner-max-steps 10`. These are report defaults,
not the general `FaultSkinner` API defaults.

The `geometry` case set keeps the same top-level JSON contract and writes one
`cases[]` entry plus one `summary.csv` row per `(case_id, variant)`. Optional
volumes and figures are split by case directory, for example
`single_dipping_plane/` and `curved_surface/`.

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
column, baseline variant, buffered F1, candidate-to-truth p95 distance, fvt
median orientation error columns, skin topology and truth metric columns, and
fvt delta columns against the baseline.

Read diagnostic variant comparison as "same case, same truth, different voter
setting." JSON delta fields under `variant_comparison.variants.*` and CSV delta
fields ending in `_delta_vs_baseline` are only populated when `current_default`
is included. If the baseline is omitted, `baseline_variant` is `null` in JSON,
empty in CSV, and the delta fields are empty.

Delta signs use `variant_value - current_default_value`. That means positive
buffered-F1 deltas are improvements, while negative distance and
orientation-error deltas are improvements. The report does not encode this
good/bad direction; consumers should interpret each metric family explicitly.

`--save-volumes` writes float32 big-endian DAT volumes under each case
directory, with `truth_fault_mask.dat` stored as 0/1 float32 values. With more
than one variant, volumes and figures are written under `case_id/variant/`.
`--save-figures` writes static center-slice PNGs. `--write-markdown-index`
writes `visual_report.md` with relative links to the case figures.

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

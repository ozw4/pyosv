# Controlled Synthetic Quality

Controlled synthetic cases measure truth quality, not reference agreement. They
are for checking whether a known fault geometry is recovered by the Python
workflow without external-data confounds. The oracle path remains the main
report path; scanner-inclusive checks start from a controlled synthetic
planarity input contract.

These validation modes answer different questions:

```text
reference comparison: is the implementation close to the original?
controlled synthetic: is the result correct against known truth?
F3 visual/multicrop: does the workflow avoid obvious failures on real data?
```

## Input Paths

The report has three public input paths:

```text
oracle:
  voting/thinning/skinning upper-bound and isolated downstream evaluation.

scanner:
  scanner + voting + thinning + skinning end-to-end synthetic evaluation.

both:
  diagnostic mode that compares oracle and scanner on the same truth geometry.
```

Oracle mode is the default and remains the backward-compatible report shape.
It starts from synthetic truth attributes (`ft_oracle`, `pt_oracle`,
`tt_oracle`) and is best read as the practical upper bound for the downstream
voting, thinning, and skinning stages when the fault likelihood and orientation
attributes are already controlled.

Scanner mode is intentionally harder. It first converts the synthetic truth
likelihood into a low-planarity surrogate, runs `FaultOrientScanner3`, and then
passes scanner-derived attributes into the same downstream stages. Its metrics
therefore include scanner errors as well as downstream voting/thinning/skinning
errors and should not be expected to match oracle-mode scores.

Both mode runs both paths for each `(case_id, variant)` using the same truth
geometry. It is a diagnostic comparison mode, not a new pass/fail contract.
Top-level `pyosv` and `quality` alias the oracle pipeline in both mode, while
the canonical pipeline reports live under
`cases[].pipelines.<pipeline>.variants.<variant>`.

## Current Scope

The controlled synthetic API includes:

- `SyntheticPlaneSpec`
- `SyntheticCurvedSurfaceSpec`
- `SyntheticScannerInputConfig`
- `Synthetic3DCase`
- `generate_single_plane_case`
- `generate_curved_surface_case`
- `make_scanner_input_from_case`
- `make_boundary_plane_case`
- `make_crossing_planes_case`
- `make_single_dipping_plane_case`
- `make_single_vertical_plane_case`
- `make_curved_surface_case`
- `make_parallel_planes_case`
- `make_weak_noisy_plane_case`
- `ft` / `pt` / `tt` oracle attributes
- top-k / truth-count masks
- buffered surface overlap
- edge false-positive ratio
- surface distance metrics
- masked orientation error
- skin metrics, including skin topology metrics
- minimal oracle pipeline smoke test
- scanner-inclusive report/CLI pipeline
- `examples/report_3d_synthetic_quality.py`

The controlled synthetic oracle path is:

```text
synthetic truth geometry
  -> ft_oracle / pt_oracle / tt_oracle
  -> OptimalSurfaceVoter
  -> voter thin
  -> FaultSkinner
  -> truth metrics
```

The scanner-inclusive path is:

```text
synthetic truth geometry
  -> synthetic scanner input
  -> FaultOrientScanner3
  -> ft_scan / pt_scan / tt_scan
  -> optional scanner thin
  -> OptimalSurfaceVoter
  -> voter thin
  -> FaultSkinner
  -> truth metrics
```

Scanner-inclusive experiments can also generate a controlled
`scanner_input` / `ep_synthetic` volume from any `Synthetic3DCase`:

```text
truth geometry
  -> ft_oracle
  -> scanner_input / ep_synthetic
  -> FaultOrientScanner3.scan() or scan_fast()
  -> scanner ft/pt/tt
```

`make_scanner_input_from_case(case, config)` converts high-on-fault
`case.ft_oracle` into low-on-fault planarity-like input using:

```text
scanner_input = background - fault_contrast * ft_oracle
scanner_input += normal(0, noise_sigma)  # only when noise_sigma > 0
scanner_input = clip(scanner_input, clip_min, clip_max)
```

The default `SyntheticScannerInputConfig` uses `background=1.0`,
`fault_contrast=0.85`, `noise_sigma=0.0`, `seed=20260706`, `clip_min=0.0`,
and `clip_max=1.0`. This mirrors the F3 scanner convention where background
planarity is high and fault-adjacent planarity is low.

The current report CLI includes these case sets:

- `minimal`: the default PR2-compatible smoke set containing only
  `single_vertical_plane`.
- `geometry`: the PR3 geometry set containing `single_vertical_plane`,
  `single_dipping_plane`, and `curved_surface`.
- `extended`: the geometry cases plus `parallel_planes`, `crossing_planes`,
  `boundary_plane`, and `weak_noisy_plane`.

Scanner mode is most useful on `geometry` and `extended` when read as a stress
test of the scanner contract. Curved, crossing, boundary, and weak/noisy cases
can expose orientation ambiguity, edge effects, fragmented skins, or weak
scanner response before the voter sees the data. Compare those rows with oracle
mode on the same case before treating a scanner-mode drop as a downstream voter
or skinning regression.

The public factory cases are:

- `single_vertical_plane`: a planar fault centered near constant `x2`, with
  constant strike and dip truth orientation.
- `single_dipping_plane`: a planar fault generated from a strike/dip normal,
  with constant truth orientation and nonzero dip geometry.
- `curved_surface`: an analytic surface whose `x1` position varies with `x2`
  and `x3`; truth strike and dip vary spatially with the local normal.
- `parallel_planes`: close but distinct faults; tests separation and skin
  fragmentation.
- `crossing_planes`: intersecting faults; tests crossing robustness and
  orientation ambiguity.
- `boundary_plane`: a fault near the `i2` boundary; tests edge artifact
  behavior and boundary handling.
- `weak_noisy_plane`: degraded likelihood with deterministic noise; tests
  robustness under weak contrast.

## Case Classification And Findings

Use controlled synthetic results to separate basic recovery checks from
diagnostic stress signals. The current classifications are:

```text
Basic pass / sanity cases:
  - single_vertical_plane
  - single_dipping_plane
  - weak_noisy_plane

Diagnostic / stress cases:
  - curved_surface: model-limit / thinning-sensitivity diagnostic
  - boundary_plane: edge/boundary stress diagnostic
  - parallel_planes: skin separation / topology diagnostic
  - crossing_planes: crossing / over-merge / over-split diagnostic
```

Observed numbers from `oracle_extended_001` are examples of the current
implementation behavior, not fixed acceptance thresholds. In that run,
`single_vertical_plane` was a clean basic pass (`fvt_buffered_f1_r2=1.0`,
`fvt_distance_p95=0.0`, and zero orientation error). `single_dipping_plane`
was also a basic pass (`fvt_buffered_f1_r2` about `0.9993`,
`fvt_distance_p95=1.0`, and near-zero orientation error). `weak_noisy_plane`
remained a robustness sanity case (`fvt_buffered_f1_r2` about `0.9937`) with
good skin quality.

`curved_surface` should not be read as a simple CI failure when current
defaults score poorly. `OptimalSurfaceVoter` is not an arbitrary global curved
surface tracker. It is expected to handle surfaces that remain single-valued
and moderately varying in the seed-local coordinate system, while strong
curvature or large orientation variation is a stress case. In the observed
`oracle_extended_001` run, current defaults had `fvt_buffered_f1_r2` about
`0.6546`, `fvt_distance_p95` about `14.66`, `strike_median_error` about
`37.4` deg, and `skin_buffered_f1_r2` about `0.2915`. The
`voter_thin_normal` variant improved the same truth case substantially
(`fvt_buffered_f1_r2` about `0.9879`, `fvt_distance_p95=1.0`, and
`skin_buffered_f1_r2` about `0.9753`). Treat this as a model-limit and
thinning-sensitivity signal unless a narrower regression is demonstrated.

## Curved Surface Thinning Diagnostic

`curved_surface` is not a basic pass/fail case for reference-first OSV. It is a
model-limit and thinning-sensitivity diagnostic for comparing how voter
thinning behaves on a gently curved analytic surface with spatially varying
truth strike and dip.

The diagnostic compares `reference` and `normal` voter thinning from the same
pre-thin `fv` / `vp` / `vt` volumes. `reference` thinning is closer to the
Java-reference style and remains the default path. `normal` thinning can score
better against analytic truth on curved synthetic surfaces because it thins
along the local fault-normal field instead of the reference-like strike-binned
path. A better `normal` result on `curved_surface` does not automatically mean
the default should change; read it as evidence about curved-surface
truth-quality behavior, not as a reference-agreement requirement.

Use `--thinning-diagnostics` to compute both thinning modes from the same
pre-thin voter output:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set geometry \
  --shape 33,33,33 \
  --input-mode oracle \
  --variants current_default \
  --thinning-diagnostics \
  --save-volumes \
  --save-figures \
  --write-markdown-index \
  --output-dir outputs/3d/synthetic_quality/curved_thinning_diag_001 \
  --pretty
```

Inspect these outputs:

```text
summary.csv:
  thinning_diag_* columns

metrics.json:
  cases[].variants.<variant>.thinning_diagnostic
  cases[].pipelines.<pipeline>.variants.<variant>.thinning_diagnostic
    for --input-mode both

visual_report.md:
  reference vs normal thinning overlays
```

The diagnostic `summary.csv` columns are written in this stable order:

```text
thinning_diag_reference_fvt_buffered_f1_r2
thinning_diag_normal_fvt_buffered_f1_r2
thinning_diag_normal_minus_reference_fvt_buffered_f1_r2
thinning_diag_reference_fvt_distance_p95
thinning_diag_normal_fvt_distance_p95
thinning_diag_normal_minus_reference_fvt_distance_p95
thinning_diag_reference_count
thinning_diag_normal_count
thinning_diag_intersection_count
thinning_diag_reference_only_count
thinning_diag_normal_only_count
thinning_diag_jaccard
```

Diagnostic JSON uses the mode names `reference` and `normal`, and stores
deltas under `delta.normal_minus_reference`. The DAT and PNG artifacts are
written under `case_id[/variant][/pipeline]/thinning_diagnostic/` when
`--save-volumes` or `--save-figures` is enabled. Diagnostic artifact names use
`reference`, `normal`, `reference_only`, and `normal_only` consistently.

`boundary_plane` places the true fault near a volume boundary. Poor edge
metrics are not automatically evidence that candidates should be removed by a
single edge-cleanup rule, because that could also erase true boundary faults.
False-positive suppression and boundary truth preservation need to be evaluated
separately. In the observed run, all variants had `fvt_buffered_f1_r2` about
`0.1136`, `fvt_distance_p95=30`, `skin_count=0`, and
`fvt_edge_false_positive_fraction` about `0.8788`; read this as an
edge/boundary stress diagnostic.

For `parallel_planes` and `crossing_planes`, FVT overlap alone is not enough.
These cases are intended to expose skin topology behavior: separation of nearby
truth faults for `parallel_planes`, and over-merge or over-split behavior near
intersections for `crossing_planes`. Future metrics should report coverage,
over-merge, and over-split against truth fault IDs rather than relying only on
global overlap scores or raw skin counts.

The controlled synthetic tests cover the oracle `ft` / `pt` / `tt` path for
vertical and dipping single-plane cases and the analytic curved surface. They
also cover the boundary, parallel, crossing, and weak/noisy generator contracts,
and the edge false-positive metric for boundary-local candidate artifacts.
Pipeline tests run controlled attributes through `OptimalSurfaceVoter`, apply
thinning, and check truth-quality metrics.

The current scope does not include:

- synthetic seismic generation
- scanner algorithm changes
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

`edge_false_positive_ratio(candidate_mask, truth_mask, edge_margin=...,
truth_buffer_radius=...)` reports boundary-local candidate counts and false
positive fractions outside a buffered truth mask. The edge region is the union
of samples within `edge_margin` voxels of any volume face. A candidate sample
in that edge region is counted as an edge false positive when it is outside the
truth buffer, where the buffer is the truth mask dilated by
`truth_buffer_radius` using Euclidean distance. The main fraction,
`edge_false_positive_fraction_of_candidates`, is the share of all candidates
that are edge-local false positives; the companion
`edge_false_positive_fraction_of_edge_candidates` normalizes by only edge
candidates. This is a controlled synthetic truth metric for cases such as
`boundary_plane`, not an F3 reference agreement metric.

## Report CLI

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 33,33,33 \
  --variants current_default,no_surface_orientation_smoothing,final_norm_smoothing_1,voter_thin_normal \
  --output-dir outputs/3d/synthetic_quality/extended_001 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

To compare the oracle upper-bound path with the scanner-inclusive path in one
report, run `--input-mode both`:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set geometry \
  --shape 33,33,33 \
  --input-mode both \
  --scanner-backend reference-like \
  --scanner-thin-mode reference \
  --variants current_default \
  --output-dir outputs/3d/synthetic_quality/both_geometry_reference_like_001 \
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

## Review Workflow

Read controlled synthetic results from the stable summary outward. The normal
review order is:

```text
1. Read summary.csv first.
2. Check fvt truth quality.
3. Check skin quality.
4. Check edge false-positive fraction.
5. Check variant delta columns.
6. Check visual_report.md and PNG overlays visually.
7. Use metrics.json only for detailed drill-down.
```

Start with the tabular view:

```bash
column -s, -t < outputs/3d/synthetic_quality/oracle_extended_001/summary.csv | less -S
```

For a smaller comparison table, extract the important review columns:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
import csv

p = Path("outputs/3d/synthetic_quality/oracle_extended_001/summary.csv")
cols = [
    "case_id",
    "input_mode",
    "variant",
    "fvt_buffered_f1_r2",
    "fvt_distance_p95",
    "fvt_strike_median_error",
    "fvt_dip_median_error",
    "skin_count",
    "skin_cell_count",
    "skin_largest_fraction",
    "skin_buffered_f1_r2",
    "skin_distance_p95",
    "fv_edge_false_positive_fraction",
    "fvt_edge_false_positive_fraction",
    "scanner_ft_buffered_f1_r2",
    "scanner_ft_distance_p95",
    "scanner_strike_median_error",
    "scanner_dip_median_error",
]
with p.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"no rows in {p}")
available = [c for c in cols if c in rows[0]]
print("\t".join(available))
for r in rows:
    print("\t".join(r.get(c, "") for c in available))
PY
```

For `--input-mode both`, compare the oracle and scanner rows by grouping on
`case_id` and reading the stable `pipeline` column:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
import csv

p = Path("outputs/3d/synthetic_quality/both_geometry_reference_like_001/summary.csv")
metrics = [
    "fvt_buffered_f1_r2",
    "fvt_distance_p95",
    "skin_count",
    "skin_buffered_f1_r2",
]
rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))

by_case = {}
for row in rows:
    by_case.setdefault(row["case_id"], {})[row["pipeline"]] = row

print("case_id\tmetric\toracle\tscanner\tscanner_minus_oracle")
for case_id, pipelines in sorted(by_case.items()):
    oracle = pipelines.get("oracle", {})
    scanner = pipelines.get("scanner", {})
    for metric in metrics:
        if metric in oracle and metric in scanner:
            try:
                a = float(oracle[metric])
                b = float(scanner[metric])
            except ValueError:
                continue
            print(f"{case_id}\t{metric}\t{a:.6g}\t{b:.6g}\t{b-a:.6g}")
PY
```

The current `summary.csv` format writes one row per
`(case_id, pipeline, variant)`. The `pipeline` column is always present:
`oracle` for oracle-only runs, `scanner` for scanner-only runs, and both values
for `--input-mode both`. Pipeline-specific scanner information is represented by
`scanner_backend`, `scanner_thin_mode`, and `scanner_*` quality columns. In
`--input-mode both`, compare the `oracle` and `scanner` rows, then inspect the
split `oracle/` and `scanner/` artifact directories when visual or volume-level
drill-down is needed.

Important summary columns are:

```text
case_id
pipeline
input_mode
variant
fvt_buffered_f1_r2
fvt_distance_p95
fvt_strike_median_error
fvt_dip_median_error
skin_count
skin_cell_count
skin_largest_fraction
skin_buffered_f1_r2
skin_distance_p95
fv_edge_false_positive_fraction
fvt_edge_false_positive_fraction
scanner_ft_buffered_f1_r2
scanner_ft_distance_p95
scanner_strike_median_error
scanner_dip_median_error
```

Metric direction is:

```text
Higher is better:
  - fvt_buffered_f1_r2
  - skin_buffered_f1_r2
  - scanner_ft_buffered_f1_r2
  - skin_largest_fraction, but only within the same expected topology

Lower is better:
  - fvt_distance_p95
  - skin_distance_p95
  - fvt_strike_median_error and fvt_dip_median_error
  - scanner_strike_median_error and scanner_dip_median_error
  - edge_false_positive_fraction columns
  - skin_small_cell_fraction
```

`skin_count` and `skin_cell_count` are topology context. Do not judge them
alone: the right direction depends on whether the truth case expects one
connected skin, multiple separated skins, crossing geometry, or no accepted
skin.

Read diagnostic variant comparison as "same case, same truth, different voter
setting." CSV delta fields ending in `_delta_vs_baseline` use
`variant_value - current_default_value`. Positive
`*_buffered_f1_delta_vs_baseline` is better. Negative
`*_distance_p95_delta_vs_baseline` is better. Negative
`*_orientation_error_delta_vs_baseline`, including the strike and dip median
error delta columns, is better. Delta fields are populated only when the
`current_default` baseline variant is included.

After the CSV pass, open `visual_report.md` when `--write-markdown-index` was
used. It links the center-slice PNG overlays, including truth-vs-FVT,
truth-vs-skin, and scanner overlays when scanner output is present. Use these
overlays to confirm that high or low summary metrics correspond to plausible
fault geometry rather than a misleading aggregate score.

`metrics.json` is for detailed drill-down and scripted diagnostics. Prefer
`summary.csv` for routine review, especially for `--input-mode both`. If a
script hard-codes internal `metrics.json` paths, first check `format_version`
and the documented schema section below. The canonical direct JSON access path
for pipeline metrics is
`cases[].pipelines.<pipeline>.variants.<variant>`.

`--input-mode` controls the report input path:

```text
oracle   # default: oracle ft/pt/tt; downstream upper-bound/isolated evaluation
scanner  # scanner_input -> FaultOrientScanner3 -> optional scanner thin -> voting
both     # compare oracle and scanner pipelines on the same truth case/variant
```

Scanner mode is configured with `--scanner-backend reference-like|fast`,
`--scanner-phi-min`, `--scanner-phi-max`, `--scanner-theta-min`,
`--scanner-theta-max`, `--scanner-sigma1`, `--scanner-sigma2`,
`--scanner-thin-mode none|reference|normal`, and
`--keep-scanner-edge-effects`. The scanner defaults are reference-like backend,
strike range `0..180`, dip range `45..90`, `sigma1=sigma2=2`, and reference
scanner thinning with edge-effect removal. `--scanner-backend reference-like`
uses the reference-like scan path; `fast` uses the accelerated scanner path.
`--scanner-thin-mode none` passes raw scanner attributes to voting,
`reference` applies strike-binned scanner thinning, and `normal` uses the
legacy fault-normal scanner thinning path.

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
  skin_mask_py.dat
  skins.json
  figures/
    ft_oracle_i3_center.png
    fv_py_i3_center.png
    fvt_py_i3_center.png
    truth_vs_fvt_overlay_i3_center.png
    skin_mask_py_i3_center.png
    truth_vs_skin_overlay_i3_center.png
```

The stable minimum JSON contract stores canonical metrics under
`cases[].pipelines.<pipeline>.variants.<variant>`. `cases[].variants`,
`cases[].pyosv`, and `cases[].quality` remain compatibility aliases for the
active pipeline. In single-pipeline modes, `cases[].variant_comparison` is also
an active-pipeline alias; in `--input-mode both`, it is a `pipelines` map.

```json
{
  "format_version": 1,
  "config": {
    "case_set": "minimal",
    "shape": [33, 33, 33],
    "variants": ["current_default"],
    "input_mode": "both",
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
      "pipelines": {
        "oracle": {
          "variants": {
            "current_default": {
              "pyosv": {
                "fv": {},
                "fvt": {},
                "skins": {
                  "skin_count": 1,
                  "cell_count": 1089,
                  "unique_cell_count": 1089,
                  "duplicate_cell_count": 0,
                  "largest_skin_size": 1089,
                  "largest_skin_fraction": 1.0,
                  "small_skin_size": 10,
                  "small_skin_count": 0,
                  "small_skin_cell_count": 0,
                  "small_skin_cell_fraction": 0.0
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
                "edge_false_positive": {
                  "fv_top_truth_count": {
                    "edge_false_positive_fraction_of_candidates": 0.0
                  },
                  "fvt_top_truth_count": {
                    "edge_false_positive_fraction_of_candidates": 0.0
                  },
                  "skin": {
                    "edge_false_positive_fraction_of_candidates": 0.0
                  }
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
        },
        "scanner": {
          "variants": {
            "current_default": {
              "pyosv": {
                "fv": {},
                "fvt": {},
                "skins": {}
              },
              "scanner": {
                "input": {},
                "ft": {},
                "pt": {},
                "tt": {},
                "fet": {},
                "fpt": {},
                "ftt": {}
              },
              "skinning": {
                "enabled": true
              },
              "quality": {
                "fv_top_truth_count": {},
                "fvt_top_truth_count": {},
                "edge_false_positive": {},
                "skin": {}
              },
              "scanner_quality": {
                "ft_top_truth_count": {},
                "orientation_error": {},
                "input_association": {}
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
      },
      "active_pipeline": "oracle",
      "pyosv": {
        "fv": {},
        "fvt": {},
        "skins": {}
      },
      "skinning": {
        "enabled": true
      },
      "quality": {
        "fv_top_truth_count": {},
        "fvt_top_truth_count": {},
        "edge_false_positive": {},
        "skin": {}
      },
      "variants": {
        "current_default": {
          "active_pipeline": "oracle",
          "pyosv": {},
          "skinning": {},
          "quality": {},
          "pipelines": {
            "oracle": {},
            "scanner": {}
          }
        }
      },
      "variant_comparison": {
        "pipelines": {
          "oracle": {
            "baseline_variant": "current_default",
            "variants": {}
          },
          "scanner": {
            "baseline_variant": "current_default",
            "variants": {}
          }
        }
      }
    }
  ]
}
```

Each case stores per-variant metrics under `cases[].pipelines.*.variants`.
For backward compatibility, the active pipeline is also exposed through
`cases[].variants`; `current_default` is duplicated at the case top level when
that variant is present. `cases[].pipelines.*.variant_comparison` stores
per-variant deltas against `current_default` when that baseline variant is
present; when it is not present, `baseline_variant` is `null` and the
comparison map is empty. In `--input-mode both`, the top-level comparison is a
`pipelines` map to avoid an ambiguous active-pipeline comparison.

The canonical pipeline schema is:

```text
cases[].pipelines.oracle.variants.<variant>.pyosv
cases[].pipelines.oracle.variants.<variant>.quality
cases[].pipelines.oracle.variant_comparison
cases[].pipelines.scanner.variants.<variant>.pyosv
cases[].pipelines.scanner.variants.<variant>.quality
cases[].pipelines.scanner.variants.<variant>.scanner_quality
cases[].pipelines.scanner.variant_comparison
```

Oracle-only runs include only `pipelines.oracle`; scanner-only runs include only
`pipelines.scanner`; `--input-mode both` includes both. Existing
`cases[].variants` entries remain as compatibility aliases. In scanner-only
mode, top-level `pyosv` and `quality` alias the scanner pipeline; in both mode
they alias the oracle pipeline. Scanner pipeline reports include
`scanner.input`, raw scanner `ft`/`pt`/`tt`, and used scanner
`fet`/`fpt`/`ftt` summaries; the corresponding volume artifacts are named
`ft_used`, `pt_used`, and `tt_used`.

Scanner pipeline reports also include `scanner_quality`, which measures scanner
outputs before voting/skinning: raw scanner `ft` top-truth-count overlap and
surface distance, raw and used scanner `pt`/`tt` orientation errors, and
scanner-input association with the truth surface. These metrics are separate
from downstream `quality.*` so scanner failures can be distinguished from voter,
thinning, or skinning failures. The input association uses
`abs(truth_distance) <= --truth-surface-half-width` as the near-surface mask and
`abs(truth_distance) >= max(3.0, --truth-surface-half-width + 2.0)` as the far
mask; positive contrast means the low-on-fault scanner input is lower near truth
than far from truth.
`quality.*.buffered_overlap_radius2` uses the wider `truth_fault_mask` band as
the truth target. `quality.*.surface_distance` uses the thin truth surface mask
defined by
`abs(truth_distance) <= --truth-surface-half-width`.
`quality.edge_false_positive` stores edge false-positive metrics for
`fv_top_truth_count`, `fvt_top_truth_count`, and, when skinning is enabled,
`skin`.
`quality.skin.buffered_overlap_radius2` and `quality.skin.surface_distance` use
the same truth targets as the `fv` and `fvt` truth-count metrics. With
`--skip-skinning`, each variant stores `"skinning": {"enabled": false}`,
`pyosv.skins` is a zero-count topology summary, and `quality.skin` is `null`.

When skinning is enabled, `quality.skin` has this stable structure:

```json
{
  "topology": {
    "skin_count": 1,
    "cell_count": 1089,
    "unique_cell_count": 1089,
    "duplicate_cell_count": 0,
    "largest_skin_size": 1089,
    "largest_skin_fraction": 1.0,
    "small_skin_size": 10,
    "small_skin_count": 0,
    "small_skin_cell_count": 0,
    "small_skin_cell_fraction": 0.0
  },
  "buffered_overlap_radius2": {
    "buffered_precision": 1.0,
    "buffered_recall": 1.0,
    "buffered_f1": 1.0
  },
  "surface_distance": {
    "candidate_to_truth_p95": 0.0,
    "truth_to_candidate_p95": 0.0,
    "hausdorff_p95": 0.0
  },
  "orientation_error": {
    "strike_median": 0.0,
    "dip_median": 0.0
  }
}
```

The synthetic report default skinning configuration is intentionally small for
controlled synthetic volumes: `--skinner-ru 10`, `--skinner-rv none`,
`--skinner-rw none`, and `--skinner-max-steps 10`. These are report defaults,
not the general `FaultSkinner` API defaults.

Skinning can be disabled with `--skip-skinning`. Skin extraction is configured
with `--skinner-min-likelihood`, `--skinner-min-skin-size`, `--skinner-d`,
`--skinner-ru`, `--skinner-rv`, `--skinner-rw`, `--skinner-max-steps`,
`--skinner-du`, `--skinner-max-delta-strike`, `--no-skinner-reskin`, and
`--small-skin-size`.

The `geometry` and `extended` case sets keep the same top-level JSON contract
and write one `cases[]` entry per case plus one `summary.csv` row per
`(case_id, pipeline, variant)`. Optional volumes and figures are split by case
directory, for example `single_dipping_plane/`, `curved_surface/`,
`parallel_planes/`, `crossing_planes/`, `boundary_plane/`, and
`weak_noisy_plane/`.

`--variants` accepts a comma-separated list:

```text
current_default
no_surface_orientation_smoothing
final_norm_smoothing_1
voter_thin_normal
```

The default is `current_default`. Diagnostic variants do not add pass/fail
judgments; they make the same truth metrics comparable across voter settings.
`summary.csv` writes one row per `(case_id, pipeline, variant)` and includes the
pipeline column, variant column, baseline variant, input mode, buffered F1,
candidate-to-truth p95 distance, fvt median orientation error columns,
`fv_edge_false_positive_fraction`, `fvt_edge_false_positive_fraction`, skin
topology and truth metric columns, and fvt and skin delta columns against the
baseline. Scanner columns are always in the header; they are populated for
scanner pipeline rows and empty for oracle pipeline rows:

```text
pipeline
input_mode
scanner_backend
scanner_thin_mode
scanner_ft_buffered_f1_r2
scanner_ft_distance_p95
scanner_strike_median_error
scanner_dip_median_error
scanner_input_contrast
```

The skin columns are written in deterministic order:

```text
skinning_enabled
skin_enabled
skin_count
skin_cell_count
skin_unique_cell_count
skin_duplicate_cell_count
skin_largest_size
skin_largest_fraction
skin_small_count
skin_small_cell_fraction
skin_buffered_f1_r2
skin_buffered_precision_r2
skin_buffered_recall_r2
skin_distance_p95
skin_distance_candidate_to_truth_p95
skin_distance_truth_to_candidate_p95
skin_distance_hausdorff_p95
skin_strike_median_error
skin_dip_median_error
skin_buffered_f1_delta_vs_baseline
skin_distance_p95_delta_vs_baseline
skin_strike_median_error_delta_vs_baseline
skin_dip_median_error_delta_vs_baseline
skin_count_delta_vs_baseline
```

JSON delta fields under `variant_comparison.variants.*` and CSV delta fields
ending in `_delta_vs_baseline` are only populated when `current_default` is
included. If the baseline is omitted, `baseline_variant` is `null` in JSON,
empty in CSV, and the delta fields are empty. Delta signs use
`variant_value - current_default_value`.

Metric direction is:

```text
buffered F1: higher is better
distance p95: lower is better
orientation error: lower is better
edge false-positive fraction: lower is better
small skin cell fraction: lower is better
skin_count: depends on truth topology; do not judge it alone
largest_skin_fraction: for single-fault cases, higher means less fragmentation
```

For skin deltas, `skin_buffered_f1_delta_vs_baseline` follows the same
positive-is-better interpretation, while `skin_distance_p95_delta_vs_baseline`,
`skin_strike_median_error_delta_vs_baseline`, and
`skin_dip_median_error_delta_vs_baseline` are better when negative.
`skin_count_delta_vs_baseline` is useful topology context only. The underlying
JSON comparison fields keep the older `_delta_vs_current` names, but the sign
definition is the same.

`--save-volumes` writes float32 big-endian DAT volumes under each case
directory, with `truth_fault_mask.dat` and `skin_mask_py.dat` stored as 0/1
float32 values. It also writes `skins.json` with deterministic skin cell
records, or a disabled zero-count object when `--skip-skinning` is used. With
more than one variant, volumes, `skins.json`, and figures are written under
`case_id/variant/`.

For `--input-mode scanner`, the same case or variant directory also includes
scanner artifacts:

```text
scanner_input.dat
ft_scan.dat
pt_scan.dat
tt_scan.dat
ft_used.dat
pt_used.dat
tt_used.dat
```

`ft_scan` / `pt_scan` / `tt_scan` are raw scanner outputs. `ft_used` /
`pt_used` / `tt_used` are the attributes passed to voting after scanner
thinning; with `--scanner-thin-mode none`, used attributes match raw scanner
outputs. For `--input-mode both`, artifacts are split into pipeline
subdirectories:

```text
case_id[/variant]/oracle/
case_id[/variant]/scanner/
```

`--save-figures` writes static center-slice PNGs, including skin mask and
truth-vs-skin overlays. Scanner mode also writes scanner input, `ft_scan`,
`ft_used`, truth-vs-`ft_scan`, truth-vs-`ft_used`, and scanner FVT overlays.
`--write-markdown-index` writes `visual_report.md` with relative links to the
case figures, scanner overlays, scanner input/attribute metrics, FVT metrics,
and skin metrics.

## Test Commands

The PR6 minimum validation command for the synthetic scanner-inclusive suite is:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_synthetic3d.py \
  tests/test_synthetic_metrics.py \
  tests/test_synthetic_oracle_pipeline.py \
  tests/test_report_3d_synthetic_quality.py
```

The broader synthetic voting/skinning checks are:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_voting3d.py \
  tests/test_skinner.py
```

If Ruff is installed:

```bash
PYTHONPATH=src python -m ruff check src tests examples
PYTHONPATH=src python -m ruff format --check src tests examples
```

# Controlled Synthetic Quality

Controlled synthetic evaluation measures PyOSV output against known 3D fault
geometry. It isolates numerical and geological behavior without requiring F3
volumes, `reference_osv/`, Java, Jython, or Mines JTK.

Synthetic truth and reference agreement are different evaluation domains:

- controlled synthetic metrics measure recovery against generated truth;
- Java/reference reports measure agreement with an external implementation;
- F3 reports measure agreement and structural differences on one real-data
  volume without independent geological truth labels.

Do not combine those meanings into one score or describe F3 public-reference
agreement as synthetic truth accuracy.

## Processing contract

The stage order is:

```text
synthetic truth
  -> input selection
  -> scanner
  -> scanner thinning
  -> voting
  -> voter thinning
  -> skinning
  -> truth metrics and diagnostics
```

The oracle path bypasses the scanner. The scanner path exercises every stage.
Scanner backend, scanner thinning, workflow mode, voter thinning, and skinning
are separate configuration choices.

## Public APIs

Synthetic geometry is defined in `pyosv.synthetic3d`:

```python
from pyosv.synthetic3d import (
    Synthetic3DCase,
    SyntheticCurvedSurfaceSpec,
    SyntheticPlaneSpec,
    SyntheticScannerInputConfig,
    generate_curved_surface_case,
    generate_single_plane_case,
    make_boundary_plane_case,
    make_crossing_planes_case,
    make_curved_surface_case,
    make_parallel_planes_case,
    make_scanner_input_from_case,
    make_single_dipping_plane_case,
    make_single_vertical_plane_case,
    make_weak_noisy_plane_case,
)
```

Reusable evaluation entry points are exposed from
`pyosv.evaluation.synthetic_quality`:

```python
from pyosv.evaluation.synthetic_quality import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
    build_report,
    resolve_workflow_settings,
    run_case,
)
```

`build_report(...)` evaluates a registered case set and returns a detached
`format_version=1` report mapping without writing files. `run_case(...)`
evaluates one `SyntheticQualityCaseDefinition` and returns its report payload
and volume payloads.

The typed report models and current JSON/CSV serializers are exposed from
`pyosv.evaluation.reporting`:

```python
from pyosv.evaluation.reporting import (
    CaseReport,
    PipelineReport,
    Report,
    ReportConfig,
    SUMMARY_CSV_V1_FIELDS,
    VariantComparison,
    VariantReport,
    report_to_json,
    summary_csv_text,
    write_metrics_json,
    write_summary_csv,
)
```

These APIs are module-level interfaces and are not re-exported from
`pyosv.__init__`.

## Array, coordinate, and truth contracts

All controlled 3D arrays use shape `(n3, n2, n1)` and are indexed as
`array[i3, i2, i1]`. Coordinates and vectors use OSV component order
`(x1, x2, x3)`.

`Synthetic3DCase` contains matching arrays with these contracts:

| Field | dtype | Meaning |
| --- | --- | --- |
| `truth_fault_mask` | `bool` | Fault-band truth support. |
| `truth_fault_id` | `int32` | Positive component ID on fault support and zero elsewhere. |
| `truth_distance` | `float32` | Signed distance to the nearest truth surface. |
| `truth_strike` | `float32` | Truth strike in the PyOSV angle convention. |
| `truth_dip` | `float32` | Truth dip in the PyOSV angle convention. |
| `ft_oracle` | `float32` | Truth-derived fault likelihood supplied to the oracle path. |
| `pt_oracle` | `float32` | Truth-derived strike supplied to the oracle path. |
| `tt_oracle` | `float32` | Truth-derived dip supplied to the oracle path. |

Every array must have the declared case shape. Floating arrays must be finite.

Two truth supports have distinct metric roles:

- `truth_fault_mask` is the fault band used for buffered overlap and edge
  false-positive interpretation;
- the thin truth surface is
  `abs(truth_distance) <= truth_surface_half_width` and is used for surface
  distance and truth-count selection.

`SyntheticTruthMetricConfig` defaults to:

```text
truth_surface_half_width = 0.5
buffer_radius = 2.0
```

`truth_fault_id` supports component-aware topology metrics for multi-fault and
crossing cases.

## Registered case sets

The report registry provides three case sets:

| Case set | Cases |
| --- | --- |
| `minimal` | `single_vertical_plane` |
| `geometry` | `single_vertical_plane`, `single_dipping_plane`, `curved_surface` |
| `extended` | all geometry cases plus `parallel_planes`, `crossing_planes`, `boundary_plane`, `weak_noisy_plane` |

The cases exercise different contracts:

| Case | Intended diagnostic role |
| --- | --- |
| `single_vertical_plane` | Basic planar localization and orientation. |
| `single_dipping_plane` | Dipping-plane localization and orientation. |
| `curved_surface` | Spatially varying orientation and thinning sensitivity. |
| `parallel_planes` | Separation, fragmentation, and component topology. |
| `crossing_planes` | Intersection behavior, over-merge, and over-split. |
| `boundary_plane` | Boundary support, edge behavior, and fallback paths. |
| `weak_noisy_plane` | Deterministic weak-contrast and noise robustness. |

`weak_noisy_plane` has a seed-aware factory for explicit deterministic
realizations. Other registered cases are deterministic for a given shape.

A case definition validates that its factory returns the registered `case_id`.
Case IDs and case sequences must be nonempty, registered, and unique.

## Input modes

The public input modes are:

| Mode | Evaluation path |
| --- | --- |
| `oracle` | `ft_oracle` / `pt_oracle` / `tt_oracle` -> voting -> voter thinning -> skinning |
| `scanner` | synthetic scanner input -> scanner -> scanner thinning -> voting -> voter thinning -> skinning |
| `both` | Runs the oracle and scanner paths on the same truth case and variant. |

The oracle path evaluates downstream behavior from controlled attributes. The
scanner path includes scanner localization and orientation error. Compare the
two paths by `case_id`, pipeline, variant, and effective configuration; a drop
in scanner mode is not automatically attributable to voting or skinning.

### Scanner input

`make_scanner_input_from_case(case, config)` builds a low-on-fault,
high-background planarity-like volume:

```text
scanner_input = background - fault_contrast * ft_oracle
scanner_input += normal(0, noise_sigma)  # only when noise_sigma > 0
scanner_input = clip(scanner_input, clip_min, clip_max)
```

`SyntheticScannerInputConfig` defaults to:

```text
background = 1.0
fault_contrast = 0.85
noise_sigma = 0.0
seed = 20260706
clip_min = 0.0
clip_max = 1.0
```

### Scanner configuration

`SyntheticScannerConfig` defaults to:

| Setting | Default |
| --- | --- |
| backend | `reference-like` |
| strike range | `0.0` through `180.0` degrees |
| dip range | `45.0` through `90.0` degrees |
| `sigma1`, `sigma2` | `2.0`, `2.0` |
| refinement factor | `2` |
| scanner thinning | `reference` |
| requested edge cleanup | `true` |

The accepted scanner backends are `reference-like`, `quality`, `fast`, and
`ensemble`. `quality` uses the configured refinement factor. `ensemble` is an
explicit diagnostic backend that selects component attributes voxel by voxel.
No workflow profile selects a scanner backend.

Scanner thinning accepts:

- `none`: pass raw scanner attributes to voting;
- `reference`: apply strike-binned scanner thinning;
- `normal`: apply fault-normal scanner thinning.

Scanner edge cleanup is effective only for `reference` thinning. Reports use
`effective_remove_edge_effects=null` when the requested setting is not
applicable.

`--scanner-backend-matrix` evaluates `reference-like`, `quality`, and `fast`
under the same case, scanner-side controls, workflow, and variant. It is active
only for `scanner` or `both` input modes.

## Workflow and variant resolution

Workflow profiles accept `reference`, `quality`, and `diagnostic`. Resolution
follows this order:

1. Select workflow defaults.
2. Apply explicitly supplied voting and skinning configuration.
3. Apply the selected variant's declarative patch.

Scanner configuration is resolved independently. A supplied
`SyntheticVotingConfig`, including its default constructor, is preserved as
supplied. Explicit skinning flags prevent the corresponding workflow default
from replacing that field.

The effective workflow defaults, variant registry, scanner-thinning comparison
profiles, and promotion-gate thresholds are specified in
[Quality Workflow Mode](quality_mode.md). The canonical distinction between
scanner backend, workflow, thinning, and reference target is specified in
[Mode Comparison Contract](mode_comparison.md).

`current_default` applies no variant patch and represents the resolved workflow
configuration. The `default` variant preset contains only `current_default`.
`--variants` overrides `--variant-preset` and requires a nonempty, unique list
of registered variant names.

## Execution ownership and reuse

The application layer builds each case, prepares its requested inputs, resolves
effective configuration, runs variants, and constructs immutable report
models. Artifact writers are separate from numerical evaluation.

Prepared scanner inputs and scanner outputs may be shared only within the same
case, shape, and scanner configuration. Voting, thinning, seed selection, and
skinning reuse is controlled by case-local semantic keys. Cached arrays are
read-only and the cache is discarded after the case's variant loop. Array
identity alone is not a reusable stage identity.

See [Architecture](architecture.md) for the complete ownership and cache
contract.

## Truth metric contract

### Candidate selection

Reports use deterministic candidate masks:

- `top_truth_count_mask(values, truth_surface_mask)` selects exactly the largest
  `k` samples, where `k` is the number of thin truth-surface voxels;
- positive truth-count selection restricts that top-k operation to values
  strictly greater than `1e-6` and may therefore select fewer than `k` samples;
- positive candidate counts and floating-point `nonzero_fraction` use the same
  strict `1e-6` threshold.

Equal values are ordered deterministically by flat array index.

### Surface metrics

For `fv`, `fvt`, scanner likelihood, and skin occupancy, reports can include:

- exact intersection, union, precision, recall, F1, and Jaccard;
- buffered precision, recall, and F1 at the configured radius;
- directional candidate-to-truth and truth-to-candidate surface distances;
- mean, median, p90, and p95 distance summaries;
- masked strike and dip error summaries;
- edge candidate and edge false-positive counts and fractions.

Buffered overlap compares candidate support with `truth_fault_mask`. Surface
distance compares candidate support with the thin truth surface. The default
edge shell includes samples within two voxels of any volume face. An edge
candidate is a false positive when it lies outside the configured truth buffer.

Higher buffered F1 is better. Lower distance, orientation error, and edge
false-positive fraction are better.

### Scanner metrics

Scanner pipeline reports separate scanner quality from downstream quality:

- raw and used scanner-likelihood overlap and distance;
- raw and used strike/dip error;
- scanner-input association near and far from truth;
- optional scanner-to-voting and scanner-to-thinning retention diagnostics.

Scanner-input association defines near truth using the configured thin truth
surface and far support using
`abs(truth_distance) >= max(3.0, truth_surface_half_width + 2.0)`. Positive
contrast means the low-on-fault input is lower near truth than far from truth.

### Skin metrics

When skinning is enabled, reports include:

- skin count, total and unique cell counts, and duplicate counts;
- largest-skin and small-skin summaries;
- skin overlap, distance, orientation, and edge metrics;
- component-aware truth/skin incidence;
- over-merge and over-split counts;
- purity and truth-component recall;
- link and reskin diagnostics where applicable.

`skin_count` and `skin_cell_count` have no universal better direction. Interpret
them with the truth topology for the case.

### Variant deltas

When `current_default` is present, each pipeline records selected variant
metrics as `variant_value - current_default_value`. Positive buffered-F1 deltas
are improvements; negative distance and orientation-error deltas are
improvements. No baseline deltas are emitted when `current_default` is absent.

## Report schema

The JSON root is:

```text
format_version: 1
config: resolved report configuration
cases: ordered case reports
```

The canonical case paths are:

```text
cases[].pipelines.oracle.variants.<variant>
cases[].pipelines.oracle.variant_comparison
cases[].pipelines.scanner.variants.<variant>
cases[].pipelines.scanner.variant_comparison
```

Oracle-only reports contain `pipelines.oracle`. Scanner-only reports contain
`pipelines.scanner`. `both` reports contain both.

Each variant pipeline report contains the applicable groups:

```text
pyosv
pyosv.voting
quality
skinning
scanner
scanner_quality
scanner_downstream
scanner_boundary_stage_diagnostics
thinning_diagnostic
scanner_backend_matrix
```

`cases[].variants`, top-level case `pyosv` / `quality` / `skinning`, and
case-level `variant_comparison` are current aliases for the active pipeline.
For `both`, the active alias is the oracle pipeline and the case-level
comparison is partitioned by pipeline. New readers should use
`cases[].pipelines` as the canonical path.

`summary.csv` writes one row per `(case_id, pipeline, variant)` and uses the
fixed `SUMMARY_CSV_V1_FIELDS` column order. Use the `pipeline`, `input_mode`,
`workflow_mode`, scanner, thinning, and skinning columns to confirm that rows
are comparable before interpreting metric differences.

## Artifact output

The CLI always writes:

```text
metrics.json
summary.csv
```

Optional flags add:

```text
--write-markdown-index  -> visual_report.md
--save-volumes          -> DAT volumes and skins.json
--save-figures          -> center-slice PNG figures and overlays
```

For one variant, case artifacts are written directly under the case directory.
For multiple variants, each variant receives a subdirectory. With
`--input-mode both`, oracle and scanner artifacts are separated into `oracle/`
and `scanner/` subdirectories.

Core volume names include:

```text
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
```

Scanner outputs add:

```text
scanner_input.dat
ft_scan.dat
pt_scan.dat
tt_scan.dat
ft_used.dat
pt_used.dat
tt_used.dat
scanner_confidence.dat  # when supplied by the selected backend
```

Thinning diagnostics are written under `thinning_diagnostic/`. Scanner boundary
stage volumes are written under `scanner_boundary_stage_diagnostics/`.

Visualization requires the optional `viz` dependencies. Figures use fixed
center slices and include truth-versus-FVT, truth-versus-skin, and scanner
likelihood overlays when the corresponding volumes exist.

## CLI usage

A small oracle report using the resolved default is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set minimal \
  --shape 33,33,33 \
  --workflow-mode reference \
  --variants current_default \
  --input-mode oracle \
  --output-dir outputs/3d/synthetic_quality/minimal_oracle_33 \
  --pretty
```

A diagnostic matrix over oracle and scanner pipelines is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 33,33,33 \
  --variant-preset quality-matrix \
  --input-mode both \
  --workflow-mode diagnostic \
  --scanner-backend reference-like \
  --scanner-thin-mode reference \
  --output-dir outputs/3d/synthetic_quality/quality_matrix_33 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

A scanner-inclusive quality-workflow report is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 49,49,49 \
  --workflow-mode quality \
  --variants current_default \
  --input-mode scanner \
  --scanner-backend quality \
  --scanner-refinement-factor 2 \
  --scanner-downstream-diagnostics \
  --scanner-boundary-stage-diagnostics \
  --output-dir outputs/3d/synthetic_quality/scanner_quality_current_49 \
  --pretty
```

Add `--save-volumes` when DAT and `skins.json` artifacts are required. Add
`--skip-skinning` when only scanner, voting, and thinning behavior should be
evaluated.

## Voter thinning diagnostics

`--thinning-diagnostics` (also accepted as
`--include-thinning-diagnostic`) computes `reference` and `normal` voter
thinning from the same pre-thin vote output for the selected diagnostic cases.
The default diagnostic case is `curved_surface`.

Diagnostic JSON stores both modes and `delta.normal_minus_reference`. Optional
artifacts include reference, normal, shared, and mode-only masks under
`thinning_diagnostic/`.

Use `--thinning-diagnostic-cases` to select a nonempty, unique list of
registered case IDs.

## Scanner boundary stage diagnostics

`--scanner-boundary-stage-diagnostics` is active only for scanner pipelines. It
records this fixed stage order:

```text
scanner_ft_positive
-> scanner_fet_positive
-> seed_candidate
-> seed_selected
-> fv_positive
-> fvt_positive
-> primary_skin
-> fallback_skin
-> final_skin
```

Stage reports include counts, boundary/interior profiles, distance-to-face
profiles, connected components, truth comparisons, and adjacent-stage
transitions. Transitions record applicability explicitly; fallback transitions
remain present but are marked not applicable when fallback is unused.

When `--save-volumes` is enabled, the stage masks and the boundary shell are
written under `scanner_boundary_stage_diagnostics/`.

A deterministic scalar summary can be generated from a completed report:

```bash
PYTHONPATH=src python scripts/summarize_scanner_boundary_stages.py \
  outputs/3d/synthetic_quality/scanner_quality_current_49/metrics.json \
  --case-id boundary_plane \
  --variant current_default \
  --retention-threshold 0.80 \
  --output-json outputs/3d/synthetic_quality/scanner_quality_current_49/stage_summary.json \
  --output-markdown outputs/3d/synthetic_quality/scanner_quality_current_49/stage_summary.md
```

The retention threshold belongs to this diagnostic summary and is not a
promotion-gate threshold.

## Reading results

Use this review order:

1. Confirm the effective row configuration in `summary.csv`.
2. Review scanner metrics for scanner pipelines.
3. Review FVT overlap, distance, orientation, and edge metrics.
4. Review skin overlap and component topology.
5. Review variant deltas only against the matching `current_default` row.
6. Inspect figures and `metrics.json` for spatial and diagnostic detail.

Do not compare rows that differ in case, shape, input pipeline, scanner backend,
scanner thinning, workflow, voter thinning, skinning, truth-metric settings, or
variant scope unless that difference is the declared comparison axis.

The scanner-thinning policy comparison profiles and their promotion gates use
paired `summary.csv` and `metrics.json` evidence. Their current commands and
configuration contracts are documented in
[Quality Workflow Mode](quality_mode.md).

## Separate canonical mode-comparison experiment

`examples/report_3d_synthetic_mode_comparison.py` runs the canonical scanner
backend × workflow experiment and writes its own validated scalar bundle. It is
not an alias for this report CLI and does not combine Synthetic and F3 results.
See [Synthetic Mode Comparison](synthetic_mode_comparison.md).

## F3 boundary

Controlled synthetic evaluation does not read F3 files. F3 publication
comparison uses the complete `(420, 400, 100)` volume as one evaluation unit.
Crops and regional partitions are diagnostics within that volume and are not
synthetic cases or statistical replicates.

Use [F3 3D Reference Data Validation](f3d_validation.md) for the F3 runner and
artifact contract. Use [F3 Visual Diagnostics](f3d_visual_diagnostics.md) for
display and interpretation rules.

## Related specifications

- [Architecture](architecture.md)
- [Quality Workflow Mode](quality_mode.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Synthetic Mode Comparison](synthetic_mode_comparison.md)
- [3D Orientation Scanning](orient3d.md)
- [3D Voting Conventions](3d_voting.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Skinning](skinning.md)
- [F3 3D Reference Data Validation](f3d_validation.md)

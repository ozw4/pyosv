# Synthetic Mode Comparison

The synthetic mode-comparison command runs the canonical scanner and workflow
comparison against controlled 3D truth. It needs no F3 data or network access.
It fixes the scanner thinning, edge policy, quality refinement factor,
workflows, and `current_default` variant; the CLI only selects cases, shape,
trial seeds, oracle isolation, skinning, output formatting, and output location.

## Comparison cells

The command separates three questions:

- Scanner-only cells `RL-SCAN` and `Q-SCAN` compare reference-like and quality
  scanner outputs before a downstream workflow can affect the result.
- Oracle-isolation cells `ORACLE-REF` and `ORACLE-QUAL` compare the reference
  and quality workflows from the same truth-derived oracle attributes. Use
  `--no-oracle-workflow-isolation` only when these isolation cells are not
  needed.
- End-to-end cells form the canonical 2×2: `RL-REF`, `RL-QUAL`, `Q-REF`, and
  `Q-QUAL`. The first part of each label selects the scanner backend and the
  second part selects the downstream workflow.

Registered deterministic cases run once with `seed=None`, regardless of the
number of `--trial-seeds`. Only registered stochastic cases, currently
`weak_noisy_plane`, expand into one trial per seed. Seeds control case
generation; they do not replace the fixed scanner-input seed.

## Reading the results

`metrics_long.csv` contains finite per-trial values with their stage,
selection, unit, and direction (`higher`, `lower`, or `neutral`). Read a metric
in that declared direction and compare like-for-like rows.
`array_nonzero_fraction` uses the synthetic-quality report definition: a
floating-point value counts as nonzero only when its magnitude is strictly
greater than `1e-6` (integer values use ordinary nonzero counting).

`contrasts.csv` contains paired, within-trial linear differences. Scanner and
workflow effects compare their named cell pairs. `scanner_main_effect` and
`workflow_main_effect` average an axis effect across the other 2×2 axis.
`scanner_workflow_interaction` measures whether one axis's effect changes with
the other axis; it is not an additional quality score. For directional
metrics, positive `improvement_value` always means improvement; neutral metrics
have no improvement value.

`metric_aggregates.csv` and `contrast_aggregates.csv` are descriptive
statistics (`n`, mean, median, standard deviation, range, and quartiles) within
each case and metric grouping. They are summaries, not significance tests.
Deterministic cases therefore have `n=1`, even in a five-seed experiment.

## Commands

A small development smoke run is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_mode_comparison.py \
  --case-set minimal \
  --shape 9,9,9 \
  --skip-skinning \
  --output-dir outputs/3d/synthetic_mode_comparison/smoke_9
```

An explicit public experiment command is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_mode_comparison.py \
  --case-set extended \
  --shape 49,49,49 \
  --trial-seeds 20260707,20260708,20260709,20260710,20260711 \
  --output-dir outputs/3d/synthetic_mode_comparison/extended_49_five_seed \
  --pretty
```

These are reproduction instructions, not claims that either experiment has
already run. In the public command every deterministic case still has `n=1`;
only the stochastic case has five seed trials. The output directory must not
already exist.

## Artifact bundle and validation

The successful command writes exactly these eight files atomically:

- `manifest.json`: artifact schema v3 plus independent scalar-evidence and
  runtime contract versions, requested configuration, resolved canonical plan,
  case and trial order, software versions, cache statistics, and source
  provenance. The current scalar-evidence and runtime contract versions are
  both 1.
- `cell_reports.json`: ordered scalar cell reports. Artifact schema v3 records
  the complete registry-ordered `scanner_metric_evidence` in every scanner-only
  and end-to-end scanner cell. Each scanner-stage candidate-count entry also
  carries the canonical overlap, distance, orientation, and edge source report
  needed to validate its publication metrics algebraically. Scanner publication
  metrics are joined totally and one-to-one to this persisted evidence; a
  missing applicable evidence entry or metric row is invalid.
- `metrics_long.csv` and `metric_aggregates.csv`: trial metrics and descriptive
  summaries.
- `contrasts.csv` and `contrast_aggregates.csv`: paired contrasts and their
  descriptive summaries.
- `runtime.csv`: stage timing and shared-stage call counts.
- `completion.json`: the required file list plus size and SHA-256 records.

After each case is generated, the experiment rejects an empty truth-surface
support mask before constructing scanner input or starting a scanner. The CLI
validates `completion.json`, hashes, schemas, the complete file set, and
cross-file semantic consistency before printing the output path. This includes
the canonical plan and trial coverage, scalar cell-report/metric agreement,
paired contrasts, aggregates, runtime rows, and cache statistics. Invalid,
non-finite, or negative truth-metric scalars are rejected while constructing
the configuration, before experiment timing or case generation begins. Cache
counters are checked from the resolved semantic stage keys in canonical cell
execution order. Validation also enforces the mathematical constraints on
array summaries and report scalars and compares shared scanner, voting, and
(when their resolved keys match) thinning evidence across cells. Programmatic
readers can call
`validate_completed_bundle(path)` from
`pyosv.evaluation.synthetic_mode_comparison`. A failed run does not publish a
partial final bundle. Successful completion establishes that the recorded
scalar evidence is internally and cross-cell consistent. Overlap ratios,
distance symmetric summaries, orientation percentile order, and edge
false-positive fractions are checked algebraically with one strict numeric
tolerance. A `top_truth_count` candidate count must equal the truth-surface
support count, while a `positive_top_truth_count` candidate count may not
exceed it. Empty candidate/truth masks cannot report buffered hits, radius-zero
buffered numerators must equal the exact intersection, and every distance
summary is bounded by the volume diagonal; empty distance reports use that
same diagonal convention. Skin largest/small summaries are recomputed from the
per-skin arrays and the effective `small_skin_size`, while component-topology
summaries are checked against their per-truth and per-skin arrays. Prepared
scanner scalar evidence is built once per trial and backend, reused by the
scanner-only and end-to-end cells, and recorded as a shared runtime stage.
Validation does not rerun any volume calculation, independently prove that its
computation was correct, or provide a tamper-prevention signature. Schema-v1
bundles do not contain complete scanner evidence, while schema-v2 bundles do
not uniquely identify their runtime coverage. Both must be regenerated with
the current schema-v3 writer; validation does not implicitly upgrade them. The
scalar-evidence contract version identifies the persisted cell-report evidence
structure independently of the artifact schema. The runtime contract version
independently identifies required runtime stage coverage, including the shared
`scanner_scalar_evidence` row in version 1.

F3 reference agreement, F3 full-volume 2×2 execution, figure generation, and
mode tuning are outside this command's scope. Use the scalar bundle to inspect
controlled synthetic truth behavior; do not interpret it as F3 agreement or as
an automatic default-selection decision.

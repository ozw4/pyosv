# Synthetic Mode Comparison

`pyosv.cli.synthetic_mode_comparison` runs the canonical scanner-backend ×
workflow experiment against controlled 3D truth. The experiment requires no F3
data, Java runtime, or network access.

The result is a scalar-only, self-validating bundle. It contains resolved
configuration, cell reports, metric rows, paired contrasts, descriptive
aggregates, cache statistics, runtime attribution, provenance, and completion
hashes. It does not contain DAT volumes, skins files, PNG figures, or a spatial
replay of the experiment.

Synthetic and F3 evaluation have different semantics:

- Synthetic metrics measure recovery against generated known truth.
- F3 metrics measure public-reference agreement on one real-data volume.
- The derived publication bundle preserves those domains separately and does
  not combine them into one quality score.

The canonical terminology and condition labels are defined in
[Mode Comparison Contract](mode_comparison.md).

## Processing and evaluation boundary

The experiment evaluates this stage model:

```text
synthetic truth geometry
  -> scanner input
  -> scanner
  -> scanner thinning
  -> voting
  -> voter thinning
  -> skinning
  -> scalar metrics, contrasts, and runtime attribution
```

Three scopes isolate different questions:

| Scope | Cells | Question |
| --- | --- | --- |
| scanner-only | `RL-SCAN`, `Q-SCAN` | How do the two scanner backends recover likelihood and orientation before a downstream workflow acts? |
| oracle workflow isolation | `ORACLE-REF`, `ORACLE-QUAL` | How do the two workflows behave when supplied with the same truth-derived attributes? |
| scanner-inclusive end to end | `RL-REF`, `RL-QUAL`, `Q-REF`, `Q-QUAL` | How do scanner backend and downstream workflow combine in the complete pipeline? |

Oracle-isolation cells are included by default and can be omitted with
`--no-oracle-workflow-isolation`. Scanner-only and end-to-end cells remain part
of the canonical plan.

## CLI

A small execution check is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_mode_comparison.py \
  --case-set minimal \
  --shape 9,9,9 \
  --skip-skinning \
  --output-dir outputs/3d/synthetic_mode_comparison/smoke_9
```

A complete extended-case command is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_mode_comparison.py \
  --case-set extended \
  --shape 49,49,49 \
  --trial-seeds 20260707,20260708,20260709,20260710,20260711 \
  --output-dir outputs/3d/synthetic_mode_comparison/extended_49_five_seed \
  --pretty
```

The command-line controls are:

| Option | Contract |
| --- | --- |
| `--output-dir` | Required new directory. The path must not already exist. |
| `--case-set` | Selects `minimal`, `geometry`, or `extended`. The default is `minimal`. |
| `--case-ids` | Selects an ordered, nonempty, unique list of registered case IDs. It is mutually exclusive with `--case-set`. |
| `--shape` | Sets the volume shape in `(n3, n2, n1)` order. Reproducible commands should set it explicitly. |
| `--trial-seeds` | Sets a nonempty, unique list of nonnegative seeds for seed-aware cases. |
| `--no-oracle-workflow-isolation` | Omits `ORACLE-REF` and `ORACLE-QUAL`. |
| `--skip-skinning` | Disables skin extraction and skin metrics while retaining the canonical cells. |
| `--pretty` | Indents JSON files. CSV schemas and numerical semantics are unchanged. |

The CLI executes the experiment, writes the bundle, validates the completed
bundle, and prints its path. A failed post-write validation removes the invalid
bundle rather than leaving a completed directory.

## Programmatic API

The public API is exposed from
`pyosv.evaluation.synthetic_mode_comparison`:

```python
from pathlib import Path

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    run_mode_comparison,
    validate_completed_bundle,
    write_artifact_bundle,
)

config = SyntheticModeComparisonConfig(
    case_set="extended",
    shape=(49, 49, 49),
    trial_seeds=(20260707, 20260708, 20260709),
)
result = run_mode_comparison(config)
bundle = write_artifact_bundle(
    result,
    Path("outputs/3d/synthetic_mode_comparison/extended_49"),
    config=config,
    pretty=True,
)
validate_completed_bundle(bundle)
```

`run_mode_comparison(...)` returns a `SyntheticModeComparisonResult` containing
only JSON-safe scalar evidence and typed metric, contrast, aggregate, and
runtime rows. `write_artifact_bundle(...)` validates that result before writing
it. `validate_completed_bundle(...)` returns `True` for a valid bundle and
raises `ValueError` for an invalid one.

## Canonical plan

`SyntheticModeComparisonConfig` accepts either a registered `case_set` or an
ordered `case_ids` sequence. It validates shape, trial seeds, scanner template,
voting configuration, skinning configuration, truth-metric configuration, and
explicit-setting flags before a plan is built.

The plan fixes these comparison constraints:

```text
comparison_variant = current_default
scanner_template.backend = reference-like
scanner_template.scanner_thin_mode = reference
scanner_template.remove_edge_effects = true
scanner_template.refinement_factor = 2
```

The quality scanner configuration is produced from the same scanner template by
changing only `backend` to `quality`. All angle bounds, sigmas, scanner-input
settings, scanner thinning, edge policy, and other scanner controls are held
constant across the scanner axis.

Programmatic callers may supply common scanner, voting, skinning, and
truth-metric settings. The plan records every resolved value. Scanner backend
and workflow remain the declared comparison axes; a common override is not an
additional condition axis.

The canonical plan does not use the `diagnostic` workflow and does not enable
thinning, scanner-downstream, scanner-boundary, or variant-matrix diagnostics.

## Workflow resolution

When `voting_config=None`, the two workflows resolve these defaults:

| Setting | `reference` workflow | `quality` workflow |
| --- | --- | --- |
| voter thinning | `reference` | `hybrid_v2` |
| surface-support minimum fraction | `0.0` | `0.0` |
| surface-support exponent | `0.0` | `0.0` |
| skinner method | `reference` | `quality` |
| skinner minimum likelihood | `0.5` | `None` / adaptive |
| seed planarity threshold | `0.8` | `0.5` |
| skin growth source | `thinned` | `pre_thin` |
| configured accepted-occupancy radius | `None` | `1` |
| effective accepted-occupancy radius | `5` | `1` |
| boundary fallback | disabled | enabled |
| boundary fallback policy | `empty_primary` | `empty_primary` |

An explicitly supplied `SyntheticVotingConfig` is preserved as supplied for
both workflow resolvers. Quality skinning defaults are filled only for fields
that were not explicitly selected by the caller. The fixed
`current_default` variant applies no additional patch.

Disabling skinning changes the effective skinning configuration in every
workflow cell. It does not remove cells, scanner stages, voting stages, or FVT
metrics.

## Canonical cells and order

The complete cell order is:

```text
RL-SCAN
Q-SCAN
ORACLE-REF
ORACLE-QUAL
RL-REF
RL-QUAL
Q-REF
Q-QUAL
```

With oracle workflow isolation disabled, the order is:

```text
RL-SCAN
Q-SCAN
RL-REF
RL-QUAL
Q-REF
Q-QUAL
```

The labels have these resolved meanings:

| Cell | Scope | Input | Scanner backend | Workflow |
| --- | --- | --- | --- | --- |
| `RL-SCAN` | scanner-only | scanner | `reference-like` | none |
| `Q-SCAN` | scanner-only | scanner | `quality` | none |
| `ORACLE-REF` | workflow isolation | oracle | none | `reference` |
| `ORACLE-QUAL` | workflow isolation | oracle | none | `quality` |
| `RL-REF` | end to end | scanner | `reference-like` | `reference` |
| `RL-QUAL` | end to end | scanner | `reference-like` | `quality` |
| `Q-REF` | end to end | scanner | `quality` | `reference` |
| `Q-QUAL` | end to end | scanner | `quality` | `quality` |

Cell labels are accepted only when they agree with scope, input mode, scanner
backend, and workflow. The resolved plan and the persisted cell order are part
of bundle validation.

## Trial expansion

Registered cases are expanded in selected case order.

- A deterministic case produces one trial with `seed=None`.
- A seed-aware case produces one trial for each configured trial seed.
- The registered seed-aware case is `weak_noisy_plane`.
- Deterministic cases remain one observation even when several trial seeds are
  supplied.

Trial IDs are deterministic:

```text
<case_id>                  # deterministic case
<case_id>__seed_<seed>     # seed-aware realization
```

Trial seeds control case realization. They do not replace the scanner-input
seed stored in `SyntheticScannerInputConfig`.

Each trial persists one truth-evidence object with exactly these fields and
order:

```text
fault_voxel_count
surface_voxel_count
```

Both counts must be integers from `1` through the total volume voxel count. An
empty thin truth surface is rejected before scanner input or scanner execution.

## Shared execution and caches

One trial is generated once and evaluated through a shared input and cache
context.

The runner performs these shared scanner operations once per trial:

```text
case generation
scanner-input generation
reference-like scan and scanner thinning
quality scan and scanner thinning
reference-like scanner scalar evidence
quality scanner scalar evidence
```

Prepared scanner arrays and scanner scalar evidence are reused by scanner-only
and end-to-end cells with the same backend.

Downstream work uses semantic stage keys. Seed selection, voting volume, base
thinning, primary skinning, voting scalar evidence, and thinning scalar
evidence are shared only when their complete resolved keys match. A key
referenced by at least two canonical cells is a shared stage. A key referenced
by one cell is cell-owned. A cache hit adds no call and no elapsed time.

The per-trial cache counters are:

```text
seed_hits
seed_misses
voting_hits
voting_misses
thinning_hits
thinning_misses
primary_skinning_hits
primary_skinning_misses
```

Validator expectations are derived from the canonical semantic keys and cell
execution order rather than accepted from the recorded counters alone.

## Metric registry

`metrics_long.csv` contains one finite `MetricRow` for every metric applicable
to every trial and cell. Each row records:

```text
case and trial identity
scope and cell label
input mode
scanner backend, refinement, and thinning metadata
workflow, voter thinning, and skinner metadata
variant
stage, selection, metric, value, unit, direction, and contrast eligibility
```

The current metric schema version is `1`.

Metric applicability is stage-specific:

| Cell scope | Metric stages |
| --- | --- |
| `RL-SCAN` | `scanner_raw`, `scanner_thinned` |
| `Q-SCAN` | `scanner_raw`, `scanner_thinned`, `scanner_confidence` |
| oracle and end-to-end cells | `fv`, `fvt`, and `skin` when skinning is enabled |

End-to-end scanner cells retain scanner metric evidence in `cell_reports.json`
so shared scanner evidence can be verified. Canonical scanner metric rows are
emitted by the scanner-only cells.

The main selections are:

| Selection | Contract |
| --- | --- |
| `all` | Whole-array scalar summaries, including nonzero fraction. |
| `top_truth_count` | Deterministic top-k selection with `k` equal to thin truth-surface support. |
| `positive_top_truth_count` | Positive-only top-k selection with count no greater than thin truth-surface support. |
| `skin_cells` | Metrics over the serialized scalar skin evidence when skinning is enabled. |

A floating array value counts as nonzero only when its absolute magnitude is
strictly greater than `1e-6`. Scanner-confidence rows are neutral diagnostics
and are not eligible for paired contrasts.

The registry includes:

- candidate count;
- buffered precision, recall, and F1;
- directional median and p95 surface distances;
- Hausdorff p95;
- strike and dip median and p95 errors;
- edge false-positive fraction;
- skin count, size distribution, and duplicate-cell metrics;
- component coverage, over-merge, over-split, purity, and recall metrics.

Every metric declares one direction:

```text
higher
lower
neutral
```

Direction is part of the metric registry and is validated together with unit
and contrast eligibility.

## Contrast definitions

`contrasts.csv` contains paired linear contrasts within the same exact trial,
stage, selection, and metric definition.

| Contrast | Formula |
| --- | --- |
| `scanner_only_effect` | `Q-SCAN - RL-SCAN` |
| `oracle_workflow_effect` | `ORACLE-QUAL - ORACLE-REF` |
| `scanner_effect_ref` | `Q-REF - RL-REF` |
| `scanner_effect_qual` | `Q-QUAL - RL-QUAL` |
| `workflow_effect_rl` | `RL-QUAL - RL-REF` |
| `workflow_effect_q` | `Q-QUAL - Q-REF` |
| `end_to_end_delta` | `Q-QUAL - RL-REF` |
| `scanner_main_effect` | `0.5 * [(Q-REF - RL-REF) + (Q-QUAL - RL-QUAL)]` |
| `workflow_main_effect` | `0.5 * [(RL-QUAL - RL-REF) + (Q-QUAL - Q-REF)]` |
| `scanner_workflow_interaction` | `(Q-QUAL - Q-REF) - (RL-QUAL - RL-REF)` |

A contrast is emitted only when all required cells are present and the metric
is contrast eligible. Omitting oracle workflow isolation therefore omits
`oracle_workflow_effect` while leaving the scanner-only and end-to-end
contrasts intact.

`raw_value` follows the formula. For directional metrics,
`improvement_value` normalizes the sign so that positive means improvement.
Neutral metrics have `improvement_value=None`.

Main effects and interaction are descriptive comparisons of resolved
configurations. They are not causal estimates or significance tests.

## Descriptive aggregates

`metric_aggregates.csv` and `contrast_aggregates.csv` contain descriptive
statistics grouped by case and metric identity:

```text
n
mean
median
standard deviation
minimum
maximum
q25
q75
```

A deterministic case has `n=1`. A seed-aware case has one row contribution per
configured seed. Aggregates do not convert deterministic cases into repeated
observations and do not perform a hypothesis test.

## Runtime attribution

`runtime.csv` records a deterministic stage order for each trial and one final
experiment total.

The runtime stages are:

```text
case_generation
scanner_input_generation
scanner_scan_thinning                 # once per scanner backend
scanner_scalar_evidence               # once per scanner backend
seed_selection
voting_volume
base_thinning
primary_skinning
voting_scalar_evidence
thinning_scalar_evidence
cell_execution                        # once per canonical cell
metric_extraction
contrast_extraction
trial_total
experiment_total
```

The six cacheable downstream stages may have shared rows, cell-owned rows, or a
zero-call shared row when no semantic key is applicable. A zero-call shared row
must have zero elapsed time.

`cell_execution` is the residual cell time after nested cache-miss build time is
removed. A cell's mode-owned runtime is its residual plus its cell-owned stage
rows. Shared rows are not divided among consumer cells.

The disjoint per-trial stage sum may not exceed `trial_total` beyond the fixed
runtime tolerance. The sum of all `trial_total` values may not exceed
`experiment_total`. Runtime rows are within-experiment attribution, not
isolated-process benchmarks.

## Bundle layout

A completed bundle contains exactly eight regular, non-symlink files:

```text
manifest.json
cell_reports.json
metrics_long.csv
metric_aggregates.csv
contrasts.csv
contrast_aggregates.csv
runtime.csv
completion.json
```

No additional file or directory is accepted by bundle validation.

The current contracts are:

| Contract | Version |
| --- | ---: |
| artifact schema | `3` |
| scalar-evidence contract | `7` |
| runtime contract | `4` |
| metric schema | `1` |
| completion schema | `1` |

The validator requires these exact versions.

### `manifest.json`

The manifest records:

- the contract versions;
- canonical cells;
- requested input configuration;
- resolved plan;
- case and trial order;
- shape and fixed variant;
- oracle-isolation state;
- metric-registry and contrast-definition identities;
- per-trial cache statistics;
- software versions and availability status;
- source provenance.

The resolved plan must be reproducible exactly from the recorded input
configuration under the current canonical builder.

### `cell_reports.json`

The file contains one ordered trial report per canonical trial. Each trial
contains:

```text
case_id
trial_id
seed
truth_evidence
cells
```

`cells` follows the canonical plan order. Reports contain scalar array
summaries, quality reports, scanner evidence, voting diagnostics, skinning
diagnostics, skin topology, and component-topology evidence as applicable.
NumPy arrays are prohibited from the scalar result.

### CSV files

CSV field order is defined by the typed row models. Numeric values must be
finite. Nullable fields, booleans, tuples, units, directions, and metadata are
parsed according to each file's exact schema.

### `completion.json`

`completion.json` is written after the seven hashed files. It records:

```text
schema_version = 1
status = complete
required_files
size and lowercase SHA-256 for every hashed file
```

The required-file order and metadata set must match the bundle contract
exactly.

## Atomic publication behavior

`write_artifact_bundle(...)` validates the in-memory result, writes every file
to a private sibling directory, writes `completion.json`, synchronizes the
directory, and atomically finalizes the requested output path.

The output path must not already exist. A write failure removes the temporary
state. The CLI also removes a just-written bundle when its mandatory
post-write validation fails.

## Validation contract

`validate_completed_bundle(...)` performs file-integrity, schema, semantic, and
cross-file checks.

It verifies:

1. The bundle is a non-symlink directory with exactly the eight required
   regular files.
2. `completion.json` has the exact schema, status, required-file list, metadata
   set, sizes, and SHA-256 values.
3. `manifest.json` uses the exact contract versions and field set.
4. The recorded input configuration rebuilds the exact resolved plan, cases,
   trials, cell order, metric registry, and contrast definition.
5. `cell_reports.json` has one report per trial, one payload per canonical cell,
   the exact recursive scalar schema, and valid trial truth evidence.
6. Scanner, voting, thinning, and skinning scalar reports satisfy their numeric
   and topology algebra.
7. Metric rows have exact applicability, coverage, order, metadata, units,
   directions, value constraints, and one-to-one joins to persisted evidence.
8. Shared scanner, voting, and thinning evidence is identical wherever semantic
   stage keys are shared.
9. Paired contrasts are exactly reproducible from metric rows.
10. Metric and contrast aggregates are exactly reproducible from their source
    rows.
11. Cache counters match the canonical semantic-key execution order.
12. Runtime rows match the canonical stage coverage and order, including
    shared/cell-owned attribution and elapsed upper-bound algebra.
13. Truth-derived and mask-derived counts remain within volume capacity.
14. Buffered overlap, distance, orientation, edge, skin-size, duplicate-cell,
    and component-topology relationships are mathematically consistent.

Validation reconstructs scalar objects from the bundle and re-evaluates their
algebra. It does not rerun the case generator, scanner, voting, thinning,
skinning, or any volume calculation. It does not independently prove the
numerical computation, establish tamper prevention, or provide a cryptographic
signature.

A direct validation command is:

```bash
PYTHONPATH=src python - <<'PY'
from pyosv.evaluation.synthetic_mode_comparison import validate_completed_bundle

validate_completed_bundle(
    "outputs/3d/synthetic_mode_comparison/extended_49_five_seed"
)
PY
```

## Reading the bundle

Use this review order:

1. Read `manifest.json` to confirm case order, trial identities, shape, cells,
   resolved scanner controls, workflow settings, truth metrics, skinning state,
   and provenance.
2. Read `metrics_long.csv` by matching case, trial, cell, stage, selection, and
   metric. Apply the declared direction.
3. Read `contrasts.csv` for paired within-trial differences.
4. Read the aggregate CSV files for descriptive summaries across stochastic
   realizations.
5. Read `cell_reports.json` when a metric requires source evidence or topology
   drill-down.
6. Read `runtime.csv` as stage attribution within this experiment.

Do not compare rows with different case, trial, shape, truth-metric settings,
stage, selection, or resolved configuration unless that difference is the
declared comparison axis.

The bundle produces no automatic winner, promotion decision, default change,
or cross-domain score. Scanner confidence and neutral metrics are diagnostics,
not directional quality claims.

## Publication and F3 boundary

This command does not read F3 files, compute F3 public-reference agreement, or
generate publication PNGs.

`pyosv.cli.mode_comparison_publication` consumes one completed Synthetic bundle
and one completed F3 bundle. It derives publication tables and figures without
rerunning numerical stages. Because the Synthetic source bundle is scalar-only,
Synthetic publication figures are metric, contrast, topology, and runtime
figures rather than spatial volume replay.

F3 execution and validation are documented in
[F3 3D Reference Data Validation](f3d_validation.md). Publication artifact and
figure contracts are documented in
[Mode Comparison Publication Bundle](mode_comparison_publication.md) and
[F3 Visual Diagnostics](f3d_visual_diagnostics.md).

## Related specifications

- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [Quality Workflow Mode](quality_mode.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
- [Mode Comparison Publication Bundle](mode_comparison_publication.md)
- [Architecture](architecture.md)
- [3D Orientation Scanning](orient3d.md)
- [3D Voting Conventions](3d_voting.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Skinning](skinning.md)

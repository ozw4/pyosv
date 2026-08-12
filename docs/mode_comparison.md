# Mode Comparison Contract

This document defines the canonical terminology, processing axes, condition
identifiers, and contrast formulas used by PyOSV mode-comparison experiments.
It covers controlled Synthetic evaluation, full-volume F3 comparison, and the
derived publication bundle built from completed source experiments.

Synthetic and F3 results have different meanings:

- Synthetic metrics measure recovery against generated known truth.
- F3 metrics measure agreement and structural differences against public F3
  processing outputs.
- The derived publication preserves those domains separately and does not
  combine them into one quality score.

The public F3 files `fl.dat`, `fv.dat`, and `fvt.dat` are comparison targets.
They are not an independent geological ground truth and are not processing
modes.

## Canonical terminology

| Term | Definition |
| --- | --- |
| scanner backend | The implementation that converts scanner input into fault likelihood, strike, and dip attributes. The canonical comparison values are `reference-like` and `quality`. |
| workflow mode | A downstream profile that resolves voting, voter thinning, skinning, and diagnostic defaults. Canonical comparison cells use `reference` or `quality`. |
| scanner thinning mode | The policy applied after scanner orientation estimation and before voting. It is independent of scanner backend and workflow mode. |
| voter thinning mode | The policy applied to the vote volume after voting and before skinning. It is independent of scanner thinning. |
| input mode | The source of pipeline attributes. Synthetic evaluation supports truth-derived `oracle` input and scanner-derived `scanner` input. F3 uses `ep.dat` as scanner input. |
| condition | One resolved processing configuration identified by a stable label such as `RL-REF`. |
| F3 public reference | Public `fl.dat`, `fv.dat`, and `fvt.dat` outputs used for stage-matched agreement metrics and figures. |
| synthetic truth | Generated fault masks, component IDs, distances, strike, and dip fields used as independent truth for controlled evaluation. |
| trial | One case realization. Deterministic cases have one trial; seed-aware stochastic cases may have multiple trials. |
| variant | A declarative patch used by the general Synthetic quality report. The canonical Synthetic mode comparison fixes the variant to `current_default`. |
| derived publication | A read-only report generated from completed Synthetic and F3 source bundles without rerunning numerical stages. |

Avoid the ambiguous phrases `reference mode` and `quality mode`. Use the
qualified names that identify the actual axis:

- `reference-like scanner backend`
- `quality scanner backend`
- `reference workflow`
- `quality workflow`
- `reference scanner thinning`
- `reference voter thinning`
- `F3 public reference`

Machine-facing identifiers retain their exact spelling. In particular,
`reference-like` is hyphenated.

## Processing stages and independent axes

The common stage model is:

```text
input
  -> scanner
  -> scanner thinning
  -> voting
  -> voter thinning
  -> skinning
  -> metrics, contrasts, diagnostics, and resources
```

| Stage | Primary controls |
| --- | --- |
| input | Synthetic case and input mode, scanner-input generation, or the official F3 `ep.dat` identity. |
| scanner | `scanner_backend`, angle bounds, scanner sigmas, sampling refinement, interpolation, normalization, and dtype. |
| scanner thinning | `scanner_thin_mode`, reference-thin sigma, and scanner edge cleanup. |
| voting | Voting radii, seed controls, strain limits, attribute and surface smoothing, surface-support policy, and boundary policy. |
| voter thinning | `voter_thin_mode` and voter reference-thin controls. |
| skinning | Skinner method, likelihood threshold, seed planarity threshold, growth source, occupancy, reskin policy, and boundary fallback. |
| evaluation | Truth metric settings for Synthetic, public-reference metric settings for F3, contrast definitions, regional diagnostics, and resource attribution. |

`workflow_mode` is a profile resolver. It does not select a scanner backend or
scanner thinning mode. A quality scanner does not imply a quality workflow,
and a quality workflow does not imply a quality scanner.

A condition label is meaningful only with its resolved plan. Reports and
manifests must retain the input identity, shape, scanner settings, workflow
settings, thinning policies, and relevant stage fingerprints. The label alone
is not evidence that held controls match.

## Scanner backends

### `reference-like`

The reference-like backend uses the rotate/shear scanner path exposed by
`FaultOrientScanner3.scan()` and `scan_reference_like()`. It follows the
reference control flow and geometry where practical, using Python/SciPy
interpolation and smoothing rather than a bit-exact Mines JTK implementation.

The backend name does not select scanner thinning. Canonical comparison plans
specify reference scanner thinning independently.

### `quality`

The quality backend uses `FaultOrientScanner3.scan_quality()`. It retains the
reference-like scoring path and refines the strike and dip sampling grids by the
resolved refinement factor. The canonical Synthetic and F3 plans use refinement
factor `2`.

Synthetic scanner reporting may retain the normalized response-gap confidence
volume produced by the quality scan. Confidence is a scanner diagnostic, not a
workflow setting or a truth label.

### Diagnostic scanner backends

The general Synthetic quality report also exposes `fast` and `ensemble` for
explicit diagnostics. They are not canonical mode-comparison axes and do not
receive `RL-*` or `Q-*` condition labels.

## Workflow profiles

The canonical workflows are resolved by
`pyosv.evaluation.synthetic_quality.resolve_workflow_settings(...)`.

The default resolved settings are:

| Setting | `reference` workflow | `quality` workflow |
| --- | --- | --- |
| voter thinning | `reference` | `hybrid_v2` |
| surface-support minimum fraction | `0.0` | `0.0` |
| surface-support exponent | `0.0` | `0.0` |
| surface-voting boundary policy | `reference` | `reference` |
| skinner method | `reference` | `quality` |
| skinner minimum likelihood | `0.5` | `None` / adaptive |
| seed planarity threshold | `0.8` | `0.5` |
| growth source | `thinned` | `pre_thin` |
| configured accepted-occupancy radius | `None` | `1` |
| effective accepted-occupancy radius | `5` | `1` |
| boundary fallback | disabled | enabled |
| boundary fallback policy | `empty_primary` | `empty_primary` |

An explicitly supplied `SyntheticVotingConfig` is preserved as supplied. The
quality skinning defaults are filled only for fields that the caller did not
select explicitly. A selected Synthetic quality-report variant applies its
narrow declarative patch after workflow resolution.

The `diagnostic` workflow uses reference workflow defaults and enables thinning
diagnostics. It is not a canonical comparison condition.

A pipeline that omits skinning cannot evaluate workflow-owned skinning
differences. Crop tools that end at `fvt` are therefore stage diagnostics, not
complete substitutes for the canonical end-to-end matrix.

## Thinning policies

Scanner thinning and voter thinning are separate stages.

Canonical Synthetic and F3 mode-comparison plans fix scanner thinning to:

```text
scanner_thin_mode = reference
remove_edge_effects = true
effective_remove_edge_effects = true
refinement_factor = 2
```

The scanner backend is therefore the scanner axis; scanner thinning is held
constant. A comparison that changes scanner thinning is a scanner-thinning
policy experiment or thinning ablation and must record that axis explicitly.
It must not be represented only by an `RL-*` or `Q-*` label.

The workflow default owns voter thinning: `reference` for the reference
workflow and `hybrid_v2` for the quality workflow. Explicit modes such as
`normal`, `hybrid`, and `normal_plateau` are diagnostic or ablation settings
unless a resolved comparison plan declares them as a common override.

In the F3 library configuration, `voter_thin_mode_override` applies one value to
both workflow branches. Such an override is a controlled common setting, not a
new condition axis.

## Condition identifiers

The label components are:

```text
RL   = reference-like scanner backend
Q    = quality scanner backend
REF  = reference workflow
QUAL = quality workflow
SCAN = scanner-only scope
ORACLE = truth-derived input with no scanner backend
```

The canonical identifiers are:

| ID | Scope | Scanner backend | Workflow | Input | Meaning |
| --- | --- | --- | --- | --- | --- |
| `RL-SCAN` | Synthetic scanner-only | `reference-like` | none | scanner | Scanner output and scanner-side truth metrics before downstream workflow processing. |
| `Q-SCAN` | Synthetic scanner-only | `quality` | none | scanner | Quality-scanner output and scanner-side truth metrics before downstream workflow processing. |
| `ORACLE-REF` | Synthetic workflow isolation | none | `reference` | oracle | Reference downstream workflow from truth-derived attributes. |
| `ORACLE-QUAL` | Synthetic workflow isolation | none | `quality` | oracle | Quality downstream workflow from the same truth-derived attributes. |
| `RL-REF` | End-to-end | `reference-like` | `reference` | scanner / F3 | Reference-like scanner with reference downstream workflow. |
| `RL-QUAL` | End-to-end | `reference-like` | `quality` | scanner / F3 | Reference-like scanner with quality downstream workflow. |
| `Q-REF` | End-to-end | `quality` | `reference` | scanner / F3 | Quality scanner with reference downstream workflow. |
| `Q-QUAL` | End-to-end | `quality` | `quality` | scanner / F3 | Quality scanner with quality downstream workflow. |
| `PUBLIC-REF` | F3 display/reference target | none | none | public output | Stage-matched public `fl.dat`, `fv.dat`, or `fvt.dat`; not a processing cell. |

The four end-to-end conditions are the canonical scanner-backend × workflow
matrix and use the fixed order:

```text
RL-REF, RL-QUAL, Q-REF, Q-QUAL
```

`PUBLIC-REF` is never encoded as a scanner backend, workflow, or fifth matrix
cell.

## Canonical Synthetic comparison

`pyosv.cli.synthetic_mode_comparison` runs the controlled-truth comparison.
Its plan fixes:

- scanner template backend `reference-like`;
- quality scanner configuration obtained by changing only the backend;
- scanner thinning `reference`;
- requested scanner edge cleanup enabled;
- refinement factor `2`;
- comparison variant `current_default`;
- reference and quality workflow resolution from one common configuration;
- one ordered case selection and one ordered trial specification.

The cell order is:

```text
RL-SCAN
Q-SCAN
ORACLE-REF        # omitted only with --no-oracle-workflow-isolation
ORACLE-QUAL       # omitted only with --no-oracle-workflow-isolation
RL-REF
RL-QUAL
Q-REF
Q-QUAL
```

Deterministic cases produce one trial with `seed=None`. Seed-aware stochastic
cases produce one trial for each configured trial seed. Trial seeds control case
realization; they do not redefine the fixed scanner-input seed.

A representative command is:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_mode_comparison.py \
  --case-set extended \
  --shape 49,49,49 \
  --trial-seeds 20260707,20260708,20260709,20260710,20260711 \
  --output-dir outputs/3d/synthetic_mode_comparison/extended_49_five_seed \
  --pretty
```

The command writes an atomic scalar bundle containing cell reports, long-form
metrics, paired contrasts, descriptive aggregates, runtime attribution, a
manifest, and completion hashes. It does not read F3 files.

The general `pyosv.cli.synthetic_quality` command is a separate report surface.
It evaluates one selected scanner/workflow configuration per invocation and may
apply explicit variants, scanner matrices, and diagnostics. It must not be
mistaken for the canonical mode-comparison bundle.

## Canonical F3 comparison

`pyosv.cli.f3d_mode_comparison` runs the complete official F3 volume as one
evaluation unit. It executes only the four end-to-end conditions:

```text
RL-REF, RL-QUAL, Q-REF, Q-QUAL
```

The official input identity, shape, scanner controls, scanner thinning, voting
controls, and common skinning template are fixed by the resolved F3 plan. The
two scanner configurations differ only in backend. For each backend, raw and
scanner-thinned attributes are computed once and shared between workflows.
Voting is also shared between workflows because voter thinning and skinning are
later stages.

With skinning enabled, the canonical stage graph contains:

```text
2 scanner stages
2 voting stages
4 thinning stages
4 skinning stages
```

A representative command is:

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

F3 primary metrics use the full reconstructed volume. Boundary and interior
rows are regional diagnostics within the same evaluation unit. F3 contains no
oracle-isolation cells and no independent public orientation or skin truth.

## Derived publication bundle

`pyosv.cli.mode_comparison_publication` consumes one completed Synthetic
mode-comparison bundle and one completed F3 mode-comparison bundle. It validates
those sources, reads existing scalar evidence and F3 stage artifacts, and
writes publication tables, figure-data CSV files, PNG figures, and `report.md`.
It does not rerun scanner, voting, thinning, or skinning.

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --synthetic-bundle <completed-synthetic-bundle> \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --environment-lock uv.lock \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

The publication condition order remains the four end-to-end cells. Selected
Synthetic scanner-only and oracle-isolation evidence may appear in dedicated
metrics or figures, but they do not redefine that order.

Synthetic known-truth rows and F3 public-reference-agreement rows remain in
separate tables and carry explicit evaluation semantics. The publication
produces no cross-domain aggregate score, automatic winner, or default-change
decision.

## Contrast definitions

Contrasts are paired within the same case/trial and metric definition. The raw
contrast is the stated linear combination. For directional metrics, the
Synthetic comparison also records `improvement_value`, whose sign is normalized
so positive always means improvement. Neutral metrics have no improvement
value.

| Contrast | Formula | Scope |
| --- | --- | --- |
| `scanner_only_effect` | `Q-SCAN - RL-SCAN` | Synthetic scanner-only evidence. |
| `oracle_workflow_effect` | `ORACLE-QUAL - ORACLE-REF` | Synthetic oracle workflow isolation. |
| `scanner_effect_ref` | `Q-REF - RL-REF` | Scanner effect at the reference workflow. |
| `scanner_effect_qual` | `Q-QUAL - RL-QUAL` | Scanner effect at the quality workflow. |
| `workflow_effect_rl` | `RL-QUAL - RL-REF` | Workflow effect at the reference-like scanner. |
| `workflow_effect_q` | `Q-QUAL - Q-REF` | Workflow effect at the quality scanner. |
| `end_to_end_delta` | `Q-QUAL - RL-REF` | End-to-end difference between the two diagonal conditions. |
| `scanner_main_effect` | `0.5 * [(Q-REF - RL-REF) + (Q-QUAL - RL-QUAL)]` | Average scanner effect across workflows. |
| `workflow_main_effect` | `0.5 * [(RL-QUAL - RL-REF) + (Q-QUAL - Q-REF)]` | Average workflow effect across scanners. |
| `scanner_workflow_interaction` | `(Q-QUAL - Q-REF) - (RL-QUAL - RL-REF)` | Change in workflow effect across scanner backends. |

Main effects and interaction are descriptive contrasts over resolved
configurations. They are not inferential causal effects. F3 has one full-volume
evaluation unit, and Synthetic aggregate rows are descriptive summaries rather
than significance tests.

A contrast is valid only when its component rows share:

- case and trial identity;
- shape and coordinate registration;
- stage, selection, metric, unit, and direction;
- truth or public-reference support contract;
- every held-constant resolved setting outside the declared contrast axes.

`PUBLIC-REF` is excluded from scanner/workflow contrast formulas. Comparisons
against it are stage-matched F3 agreement measurements.

## Stage sharing and provenance

Stage sharing follows semantic identity, not label similarity.

- Synthetic scanner evidence is prepared once per trial and scanner backend and
  reused by scanner-only and end-to-end cells.
- Synthetic seed, voting, thinning, and skinning work is shared only when the
  resolved semantic stage key is identical.
- F3 scanner and voting stages are content-addressed and shared between workflow
  cells for the same scanner backend.
- A cache hit contributes no new call or elapsed time; runtime reports separate
  shared stage rows, cell-owned rows, and residual cell execution.

A source bundle must record enough evidence to prove stage parents, settings,
implementation identity, input identity, and output hashes. Reusing a stage
from a different dataset, runtime contract, implementation, or resolved
configuration is invalid even when its condition label is the same.

Runtime rows are within-experiment attribution. They are not isolated-process
benchmarks and should not be added or apportioned to cells unless the recorded
runtime contract defines that attribution.

## Reporting rules

Use these rules in reports, captions, and review notes:

1. Name the axis explicitly: scanner backend, workflow, scanner thinning,
   voter thinning, skinning policy, or input mode.
2. Pair a condition ID with the resolved configuration and source identity.
3. Use `known-truth recovery`, `orientation error`, and `topology` for Synthetic
   results only when the metric definition supports the claim.
4. Use `public-reference agreement`, `difference`, `ridge displacement`,
   `density`, and `consistency` for F3 results.
5. Do not describe F3 public-reference agreement as geological accuracy.
6. Do not treat F3 slices, crops, regions, or tiles as independent replicates.
7. Do not interpret deterministic Synthetic cases as repeated observations when
   only stochastic cases expand across trial seeds.
8. Label scanner-thinning, voter-thinning, reskin-policy, boundary-policy, and
   other ablations as separate axes rather than reusing the four-cell labels
   alone.
9. Keep Synthetic and F3 metrics separate; do not average them into one score.
10. Treat main effects, interactions, aggregates, and runtime rows as
    descriptive unless a separate inferential design is defined.

## Execution surfaces

| Surface | Contract |
| --- | --- |
| `pyosv.cli.synthetic_quality` | One selected Synthetic configuration per run, with explicit variants and diagnostics. |
| `pyosv.cli.synthetic_mode_comparison` | Canonical controlled-truth scanner-only, optional oracle-isolation, and four-cell end-to-end comparison. |
| `pyosv.cli.f3d_mode_comparison` | Canonical full-volume F3 four-cell comparison with content-addressed stage reuse and validation. |
| `pyosv.cli.mode_comparison_publication` | Derived cross-domain publication report from completed source bundles; no numerical stage replay. |
| F3 crop and thinning scripts | Optional local diagnostics and stage-specific ablations; not canonical full-volume comparison units. |

## Related specifications

- [Architecture](architecture.md)
- [Quality Workflow Mode](quality_mode.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [Synthetic Mode Comparison](synthetic_mode_comparison.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
- [F3 Visual Diagnostics](f3d_visual_diagnostics.md)
- [Mode Comparison Publication Bundle](mode_comparison_publication.md)
- [3D Orientation Scanning](orient3d.md)
- [3D Voting Conventions](3d_voting.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Skinning](skinning.md)

The authoritative implementation contracts are defined in:

- `pyosv.evaluation.synthetic_mode_comparison.config`
- `pyosv.evaluation.synthetic_mode_comparison.models`
- `pyosv.evaluation.synthetic_mode_comparison.contrasts`
- `pyosv.evaluation.f3d_mode_comparison.config`
- `pyosv.evaluation.f3d_mode_comparison.models`
- `pyosv.evaluation.mode_comparison_publication.config`

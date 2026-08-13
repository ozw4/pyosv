# Architecture

`pyosv` separates numerical algorithms, evaluation orchestration, serialization,
and command-line concerns. Dependencies point from orchestration toward the
numerical core; numerical modules do not depend on evaluation commands or
artifact publication.

## Package boundary

Stable package-level modules such as `pyosv.dp`, `pyosv.orient2d`,
`pyosv.orient3d`, `pyosv.voting2d`, `pyosv.voting3d`, and `pyosv.skinner` are
public facades. Underscore-prefixed packages contain private implementation
units and do not create additional root-level API.

The package root exports only `__version__`. Subsystem APIs are imported from
their defining modules or subpackages.

The primary dependency direction is:

```text
command entry points
    -> evaluation applications and experiment runners
       -> numerical core and typed configuration
       -> report models, artifact writers, and validators
```

Examples used as canonical command entry points delegate to `pyosv.cli` rather
than owning numerical implementations.

## Numerical core

The numerical core includes orientation scanning, interpolation, filtering,
dynamic programming, voting, thinning, skinning, geometry, metrics, and DAT
I/O. Public facades validate the external contract and delegate focused work to
private implementation packages.

Core numerical APIs define their own:

- array shape, dtype, and coordinate conventions;
- parameter validation and defaults;
- rounding, boundary, and tie-breaking behavior;
- deterministic ordering;
- mutation and ownership rules; and
- optional acceleration boundary.

Numba-backed and Python fallback kernels implement the same public scientific
contract. Optional acceleration does not create a separate workflow or result
schema.

`pyosv.fault_warping` is an independent typed-contract subpackage. It exposes
NumPy-based input, configuration, result, and estimator protocol types without a
concrete estimator, evaluation runner, CLI, or artifact dependency.

## Evaluation families

The evaluation layer contains separate applications for distinct evidence and
artifact contracts.

| Package | Responsibility |
| --- | --- |
| `pyosv.evaluation.synthetic_quality` | Controlled Synthetic cases, configurable scanner and downstream execution, diagnostic variants, metrics, and report models. |
| `pyosv.evaluation.synthetic_mode_comparison` | Canonical Synthetic scanner-backend × workflow comparison, contrasts, runtime attribution, bundle writing, and validation. |
| `pyosv.evaluation.f3d_mode_comparison` | Canonical full-volume F3 comparison, content-addressed stages, runtime and dataset identity, bundle writing, resume, and validation. |
| `pyosv.evaluation.mode_comparison_publication` | Derived tables, figure data, figures, report generation, publication manifest creation, and validation from completed source bundles. |
| `pyosv.evaluation.reporting` | Immutable Synthetic-quality report models and versioned JSON/CSV serialization. |
| `pyosv.evaluation.promotion` | Promotion specifications, result rows, gate evaluation, and rendered summaries. |

Synthetic known-truth evidence and F3 public-reference evidence remain separate
through execution, reporting, and publication. The publication application
consumes completed, validated source bundles and does not rerun scanner, voting,
thinning, or skinning stages.

## CLI ownership

Canonical commands live under `pyosv.cli`:

```text
synthetic_quality
synthetic_mode_comparison
f3d_mode_comparison
mode_comparison_publication
```

CLI modules own argument parsing, command-specific path checks, process-level
runtime requirements, exit status, and output orchestration. Evaluation packages
own scientific configuration, execution, metrics, and validation. Numerical
core modules own stage algorithms.

## Synthetic-quality configuration resolution

Scanner configuration is resolved independently of the downstream workflow.
The scanner input, scanner backend, and scanner-thinning mode determine prepared
scanner attributes.

The downstream resolution contract is:

```text
workflow profile
    -> fills downstream values that are not explicitly fixed
explicit downstream configuration
    -> remains authoritative over profile defaults
variant specification
    -> applies the final narrowly scoped patch
```

Workflow profiles control voting, voter thinning, skinning, and diagnostic
defaults. They do not select a scanner backend or scanner-thinning mode.

The Synthetic-quality variant registry has one definition site:

```text
src/pyosv/evaluation/synthetic_quality/variants.py
```

A variant can patch only the settings represented by its typed voting,
thinning, post-thinning, seed, and skinning fields. Variant names do not imply
unrecorded changes to scanner or workflow configuration.

## Stage reuse and cache ownership

Synthetic-quality stage reuse is limited to one concrete Synthetic case
instance. `PipelineStageCache` is created for a case, shared across that case's
variant loop, and cleared when the loop ends.

Cache identity is semantic rather than based on array object identity. Typed
keys include the case identity, shape, attribute source, and every effective
setting that can change seed selection, voting, thinning, final thinning, or
primary skinning.

Only exact semantic matches are reusable. Cached numerical arrays are marked
read-only. Diagnostics are copied when retrieved. Cached skinning results are
stored as immutable cell and link snapshots and cloned into independent
variant-local `FaultSkin` objects.

Prepared scanner inputs can be shared only within the same case, shape, and
scanner configuration. A cache bound to one case instance rejects reuse for a
different case instance.

## Reports and artifacts

Immutable report models are the handoff between Synthetic-quality evaluation and
serialization. `LegacyReportV1Adapter` emits the JSON version-1 report contract,
and the CSV version-1 writer emits the fixed summary schema. Serialization
contracts define field names, ordering, missing-value representation, and output
layout independently of internal model organization.

Canonical mode-comparison experiments use experiment-specific models, writers,
completion records, and validators. Validators check the schema and numerical or
cross-file evidence defined by their bundle contract; they do not infer missing
scientific identity from paths or mutable runtime state.

Publication artifacts distinguish primary scientific/provenance inputs from
derived presentation files. Both tiers receive file-integrity validation, while
only the primary tier participates in publication identity as defined by the
publication manifest contract.

## Experimental and promotion boundaries

Modules under `pyosv.experimental` are explicit diagnostic implementations.
Importing or selecting one in a named variant does not make it a package default.
Defaults are determined by public API defaults, workflow resolution, and the
variant registry.

Promotion thresholds and required coverage have one definition site:

```text
src/pyosv/evaluation/promotion/specifications.py
```

Promotion scripts call the package library rather than defining independent
thresholds.

## Structural non-regression

A responsibility-preserving refactor must retain the contracted numerical
meaning, deterministic ordering, boundary behavior, public defaults, CLI exit
status, versioned report output, and artifact bytes or semantics covered by the
non-regression fixtures.

The Synthetic-quality refactoring checker defines the executable fixture
contract. Fixture replacement changes that contract and is not part of a
responsibility-preserving refactor.

See [Refactoring Non-Regression Contract](refactoring_contract.md).

## Related specifications

- [Reference-First Equivalence Policy](equivalence_policy.md)
- [Quality Workflow and Variants](quality_mode.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [Synthetic Mode Comparison](synthetic_mode_comparison.md)
- [Mode Comparison Contract](mode_comparison.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
- [Mode Comparison Publication Bundle](mode_comparison_publication.md)
- [Fault-Warping Contract](fault_warping.md)

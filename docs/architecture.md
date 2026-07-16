# Architecture

`pyosv` keeps stable compatibility modules at the package boundary and places
implementation details in focused internal packages. Existing imports such as
`pyosv.dp`, `pyosv.voting3d`, `pyosv.skinner`, and `pyosv.orient3d` are public
facades. Their underscore-prefixed implementation packages are private and may
change without becoming new root-level exports. The package root intentionally
exports only `__version__`.

## Evaluation pipeline

```text
CLI arguments
    -> config validation
    -> case construction
       scanner input/backend/thinning configuration -> scanner preparation
       workflow profile -> explicit downstream overrides
                        -> voting/voter-thinning/skinning configuration
       variant registry patch -> narrowly scoped effective variant
    -> voting -> thinning -> skinning
    -> metrics and diagnostics
    -> immutable report model
    -> JSON/CSV v1 adapters and artifact writers
```

`pyosv.cli.synthetic_quality` owns argument parsing and output orchestration.
`pyosv.evaluation.synthetic_quality` owns cases, configuration, profiles,
variant execution, scanner preparation, metrics, diagnostics, and runners.
`pyosv.evaluation.reporting` owns report models and serialization. The example
script is only a command entry point to the package CLI; tests import package
modules, never the example.

## Resolution order and reuse

Scanner input, backend, and scanner-thinning configuration are resolved for
scanner preparation independently of the downstream workflow profile. The
workflow profile fills downstream voting, voter-thinning, skinning, and
diagnostic values that were not explicitly supplied; explicit configuration
values win over those defaults. A variant is then resolved from the single
registry in `synthetic_quality/variants.py` and its narrowly scoped patch is
applied to the effective configuration:

```text
scanner configuration --------------------> scanner preparation
workflow profile -> explicit overrides ---> downstream effective configuration
variant patch ----------------------------> narrowly scoped effective variant
```

The workflow profile does not select a scanner backend or scanner thinning.
See [scanner backends, workflow modes, thinning modes, and reference
targets](mode_comparison.md) for the canonical distinction. This split does not
change the existing resolution order within downstream configuration: explicit
values still win before the variant patch is applied.

Scanner results are reusable only within one case, shape, and scanner
configuration. Within that scope, a prepared scanner input may be shared by
variants and backend-matrix evaluation. Scanner arrays must not be reused
across a different case, shape, or scanner configuration.

Seed selection and completed voting outputs (`fv`, `vp`, and `vt`) use the same
case-local lifetime. Their immutable keys contain the semantic oracle/scanner
attribute source and every effective setting that affects the stage output;
array identity alone is never a key. Only fully identical completed voting
outputs are shared. Cached voting arrays are read-only, diagnostics are copied
on retrieval, and the cache is discarded after each case's variant loop. A
cache is bound to the exact case instance and rejects reuse for another case.
Prepared attributes retain these keys only when produced by the case preparation
stage; caller-supplied replacements and custom post-thinning targets bypass the
dependent cache stages unless they have a validated semantic identity.

## Reports and experimental code

The immutable reporting models are the internal handoff between evaluation and
serialization. `LegacyReportV1Adapter` and the CSV v1 writer preserve the
existing `format_version=1`, field order, missing-value representation, and
artifact layout. Model changes therefore do not imply a report-v2 change.

Modules under `pyosv.experimental` are explicit diagnostic candidates. They
are not package-root exports and do not become defaults merely by being used by
a named variant. Promotion thresholds and coverage requirements live only in
`evaluation/promotion/specifications.py`; scripts call the promotion library.

Numerical changes and structural changes belong in separate pull requests. A
structural refactor must retain numerical meaning, ordering, boundary behavior,
CLI defaults, exit codes, and v1 outputs. See
[Refactoring non-regression contract](refactoring_contract.md) for the fixture
procedure.

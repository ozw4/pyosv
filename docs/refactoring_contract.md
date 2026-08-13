# Refactoring Non-Regression Contract

The synthetic-quality refactoring checker verifies that responsibility-preserving
code changes retain one canonical Python report contract. It compares a fixed
17³ report with committed fixtures and fails on any contracted difference.

This checker protects PyOSV output semantics and artifact bytes. It does not
establish Java or Mines JTK equivalence, scientific generalization, or a
performance requirement.

## Canonical report configuration

Run the checker from the repository root:

```bash
python scripts/check_synthetic_quality_refactor_contract.py
```

Without `--existing-output`, the checker creates a temporary output directory
and invokes the current Python interpreter with this report configuration:

```text
entry point                    examples/report_3d_synthetic_quality.py
case set                       extended
shape                          17,17,17
workflow mode                  quality
variants                       current_default,boundary_aware_voter_v1
input mode                     both
scanner backend                quality
scanner refinement factor      2
scanner downstream diagnostics enabled
pretty JSON                    enabled
volume artifacts               enabled
```

The canonical command therefore covers both oracle and scanner inputs for both
listed variants under one resolved quality-workflow configuration.

## Execution environment

The checker executes the report from the repository root with
`PYTHONPATH=<repository>/src`. It inherits the caller's environment and
overrides these reproducibility controls for the report subprocess:

```text
PYTHONHASHSEED=0
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

All other environment state is inherited. The checker does not independently
validate the Python version, NumPy version, BLAS implementation, CPU identity,
Numba mode, or dependency lock. Run it in the repository-supported environment,
including the authoritative `numpy<2` dependency bound.

## Contract fixtures

The checker reads exactly these fixtures:

```text
tests/fixtures/synthetic_quality_refactor/
  17_quality_ref2_metrics.json
  17_quality_ref2_summary.csv
  17_quality_ref2_artifact_sha256.json
```

Other files in that directory, including 49³ evidence files, are not read,
validated, regenerated, or updated by this checker.

## Comparison contract

A check passes only when all three comparison layers pass.

### `metrics.json`

The checker parses the fixture and generated files as JSON and recursively
compares their complete structures.

The default comparison is exact:

- JSON value types must match;
- object keys must match;
- array lengths and item order must match;
- scalar values must match exactly;
- numeric values have no tolerance;
- formatting, indentation, and object-key order do not affect the result.

Only the following unexpected fields in the generated file are accepted as
additive fields:

| JSON location | Accepted generated-only fields |
| --- | --- |
| any object whose path ends in `buffered_overlap_radius2` | `candidate_in_truth_buffer_count`, `truth_in_candidate_buffer_count` |
| `component_topology` | `qualification_min_fraction` |
| each direct item of `component_topology.truth_components` | `qualifying_skin_count`, `skin_cell_counts` |
| each direct item of `component_topology.skins` | `qualifying_truth_component_count`, `truth_component_cell_counts` |

These exceptions apply only at the listed object levels. They do not permit
arbitrary fields inside nested arrays or objects. Missing fixture fields,
changed values, changed types, reordered arrays, and all other generated-only
fields are differences.

JSON differences are reported by JSON path.

### `summary.csv`

The generated `summary.csv` must be byte-identical to
`17_quality_ref2_summary.csv`.

Line endings, quoting, column order, row order, number formatting, missing-value
representation, and final newline are therefore contracted. When bytes differ,
the checker also parses both files as UTF-8 CSV and reports row and column
locations to aid diagnosis; the byte mismatch remains the acceptance result.

### Volume and skin artifacts

The checker recursively inventories generated files that satisfy either
condition:

```text
path suffix is .dat
file name is skins.json
```

Only regular, non-symlink files are inventoried. For every inventoried artifact,
the contract records:

```text
relative POSIX path
byte size
SHA-256
```

The generated artifact set must exactly match
`17_quality_ref2_artifact_sha256.json`. A missing path, unexpected path, size
change, or SHA-256 change is a failure. Symlinks are not followed and do not
satisfy an expected artifact entry.

Files outside these three comparison layers are not examined. For example, PNG,
Markdown, and unrelated JSON or CSV files do not affect this checker unless
their contents are represented inside a contracted file. Filesystem metadata
such as modification time is not compared.

## Comparing an existing output

Use `--existing-output` to compare a completed report directory without running
the report command:

```bash
python scripts/check_synthetic_quality_refactor_contract.py \
  --existing-output /path/to/report
```

The path must resolve to a directory. Comparison is read-only unless
`--update-fixtures` is also supplied.

## Result semantics

The command reports all detected JSON, CSV, and artifact differences before
exiting.

```text
exit 0  all contracted comparisons pass, or a fixture update completes
exit 1  one or more contracted differences exist
exit 2  command usage or fixture-update authorization is invalid
```

A passing result means the canonical report matches the committed Python
non-regression contract. It does not mean that uncontracted files, other case
sizes, other variants, other workflows, or external reference outputs are
unchanged.

## Fixture replacement

Fixture replacement is disabled unless both controls are present:

```bash
PYOSV_UPDATE_REFACTOR_CONTRACT=1 \
python scripts/check_synthetic_quality_refactor_contract.py \
  --update-fixtures
```

The same guarded update can use a completed report:

```bash
PYOSV_UPDATE_REFACTOR_CONTRACT=1 \
python scripts/check_synthetic_quality_refactor_contract.py \
  --existing-output /path/to/report \
  --update-fixtures
```

An update replaces only the three canonical 17³ fixtures:

- parsed `metrics.json` is written as sorted, indented JSON with a final newline;
- `summary.csv` bytes are copied unchanged;
- the recursive `.dat` and `skins.json` path/size/SHA-256 manifest is written as
  sorted, indented JSON with a final newline.

The checker does not update 49³ evidence files.

Fixture replacement changes the contract; it is not a way to satisfy the
existing contract. Use it only when the intended Python numerical or artifact
contract has changed. A responsibility-preserving refactor must pass without
fixture replacement. After replacement, run the checker without update controls
to verify the new fixture set.

Do not remove compared fields, artifacts, or paths; change CSV serialization;
add comparator exceptions; or replace exact comparisons with tolerances merely
to make an unexplained difference pass. Any such change modifies the
non-regression contract itself.

## Related specifications

- [Architecture](architecture.md)
- [Reference-First Equivalence Policy](equivalence_policy.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [Quality Workflow and Variants](quality_mode.md)

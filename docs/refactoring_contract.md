# Refactoring non-regression contract

The synthetic quality refactoring contract freezes the current output of the 17³ extended
quality report before responsibilities are moved between modules. It covers both oracle and
scanner inputs for `current_default` and `boundary_aware_voter_v1`, with the quality scanner at
refinement factor 2 and downstream scanner diagnostics enabled.

Run the contract from the repository root:

```bash
python scripts/check_synthetic_quality_refactor_contract.py
```

By default, the checker creates a temporary report and compares it with the committed fixtures.
Use `--existing-output PATH` to compare a report that has already been generated. The comparison
requires semantic equality for `metrics.json`, byte equality for `summary.csv`, and exact relative
path, size, and SHA-256 equality for every `.dat` and `skins.json` artifact. PNG files, paths outside
the output directory, modification times, run times, and environment details are not contracted.

Differences are reported by JSON path, CSV row and column, and artifact path. The 49³ boundary
fixture records separately verified values and is not regenerated or evaluated by the default
check.

Fixture replacement is intentionally guarded. Both the command option and environment variable
are required:

```bash
PYOSV_UPDATE_REFACTOR_CONTRACT=1 \
python scripts/check_synthetic_quality_refactor_contract.py --update-fixtures
```

Fixture updates are appropriate only when the contract itself is deliberately re-baselined, not
as part of an ordinary refactoring.

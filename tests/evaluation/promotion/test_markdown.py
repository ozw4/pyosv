from __future__ import annotations

from pyosv.evaluation.promotion.markdown import comparison_markdown


def test_comparison_markdown_exact_empty_contract() -> None:
    report = {
        "config": {
            "baseline_summary": "base.csv",
            "candidate_summary": "candidate.csv",
            "baseline_variant": "base",
            "candidate_variant": "candidate",
        },
        "row_count": 0,
        "missing_baseline_rows": [],
        "missing_candidate_rows": [],
        "promotion_gate": {
            "name": "none",
            "passed": True,
            "boundary_plane": None,
            "non_boundary_regressions": [],
            "oracle_regressions": [],
            "topology_regressions": [],
            "false_fallback_replacements": [],
        },
    }

    assert (
        comparison_markdown(report)
        == """# Quality Delta

- baseline: `base.csv`
- candidate: `candidate.csv`
- baseline variant: `base`
- candidate variant: `candidate`
- row count: 0
- missing baseline rows: 0
- missing candidate rows: 0
- promotion gate: `none` pass

## Material Regressions

None.

## False Fallback Replacements

None.
"""
    )

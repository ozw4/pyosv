from __future__ import annotations

import math

import pytest

from pyosv.evaluation.mode_comparison_publication.semantic import (
    build_table_contract,
    canonical_digest,
    finite_json_normalize,
)


def test_canonical_digest_is_key_ordered_and_type_preserving() -> None:
    assert canonical_digest({"b": 2, "a": [None, True]}) == canonical_digest(
        {"a": [None, True], "b": 2}
    )
    assert canonical_digest(True) != canonical_digest(1)
    assert canonical_digest(None) != canonical_digest(0)


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_canonical_digest_rejects_nonfinite_json(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        finite_json_normalize({"value": value})


def test_table_contract_distinguishes_nullable_empty_and_numeric_zero() -> None:
    header = ("identity", "value")
    identity = ("identity",)
    empty = build_table_contract(header, ({"identity": "row", "value": None},), identity)
    zero = build_table_contract(header, ({"identity": "row", "value": 0.0},), identity)
    assert empty["ordered_identity_sha256"] == zero["ordered_identity_sha256"]
    assert empty["ordered_semantic_rows_sha256"] != zero["ordered_semantic_rows_sha256"]


def test_table_contract_retains_ordered_row_identity() -> None:
    header = ("identity", "value")
    first = ({"identity": "left", "value": 1.0}, {"identity": "right", "value": 2.0})
    second = tuple(reversed(first))
    assert (
        build_table_contract(header, first, ("identity",))["ordered_identity_sha256"]
        != build_table_contract(header, second, ("identity",))["ordered_identity_sha256"]
    )

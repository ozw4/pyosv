"""Typed access to synthetic-quality summary CSV rows."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .specifications import MATCH_KEY_FIELDS

MetricValue = float | bool | str | None
MatchKey = tuple[str, ...]


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class SummaryRow:
    """One CSV row; unknown columns are intentionally preserved in ``values``."""

    values: dict[str, str]

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> SummaryRow:
        return cls({field: text(value) for field, value in row.items()})

    @property
    def variant(self) -> str:
        return self.values.get("variant", "")

    @property
    def key(self) -> MatchKey:
        return tuple(self.values.get(field, "") for field in MATCH_KEY_FIELDS)

    def value(self, column: str) -> MetricValue:
        if column not in self.values:
            return None
        raw = text(self.values[column])
        if not raw:
            return None
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            value = float(raw)
        except ValueError:
            return raw
        return value if math.isfinite(value) else None


def read_summary_rows(path: Path, variant: str) -> dict[MatchKey, SummaryRow]:
    with path.open(encoding="utf-8", newline="") as file:
        rows: dict[MatchKey, SummaryRow] = {}
        for raw_row in csv.DictReader(file):
            row = SummaryRow.from_mapping(raw_row)
            if row.variant == variant:
                rows.setdefault(row.key, row)
        return rows


def key_dict(key: MatchKey) -> dict[str, str]:
    return dict(zip(MATCH_KEY_FIELDS, key, strict=True))


def numeric(value: MetricValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, int | float) else None

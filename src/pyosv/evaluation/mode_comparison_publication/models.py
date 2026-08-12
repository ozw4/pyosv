"""Small immutable data carriers used by the publication report builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SyntheticSourceBundle:
    """Validated, scalar-only synthetic source evidence."""

    path: Path
    manifest: Mapping[str, Any]
    completion_sha256: str
    metric_rows: tuple[Any, ...]
    contrast_rows: tuple[Any, ...]
    runtime_rows: tuple[Any, ...]
    skinning_enabled: bool
    case_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class F3SourceBundle:
    """Validated F3 scalar evidence plus the external data-root contract."""

    path: Path
    data_root: Path
    dataset_spec: Any
    run_manifest: Mapping[str, Any]
    completion_sha256: str
    result: Any
    metric_evidence: tuple[Any, ...]
    dataset_identity: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PublicationReport:
    """Derived tables and validated source handles ready for atomic writing."""

    synthetic: SyntheticSourceBundle
    f3: F3SourceBundle
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]

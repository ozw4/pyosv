"""Reusable configuration for synthetic quality evaluation."""

from .config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from .scanner import ScannerAttributes, scanner_attributes_from_case

__all__ = [
    "SyntheticScannerConfig",
    "SyntheticSkinningConfig",
    "SyntheticTruthMetricConfig",
    "SyntheticVotingConfig",
    "ScannerAttributes",
    "scanner_attributes_from_case",
]

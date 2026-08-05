"""Protocols shared by future fault-warping estimator backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import FaultWarpingConfig, FaultWarpingInput, FaultWarpingResult


@runtime_checkable
class FaultWarpingEstimator(Protocol):
    """A backend that estimates row-aligned apparent sample-axis shifts."""

    def estimate(
        self,
        inputs: FaultWarpingInput,
        config: FaultWarpingConfig,
    ) -> FaultWarpingResult:
        """Estimate apparent sample-axis shifts under the supplied contract."""
        ...

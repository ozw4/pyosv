from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from pyosv.evaluation import f3d_scanner_policy as policy


POLICY_CONFIG_KWARGS = {
    "reference_thin_sigma": 1.0,
    "ru": 1,
    "rv": 2,
    "rw": 3,
    "strain_max1": 0.25,
    "strain_max2": 0.25,
    "surface_smoothing1": 2.0,
    "surface_smoothing2": 2.0,
    "surface_orientation_smoothing": None,
    "final_normalization_smoothing": None,
    "d": 1,
    "fm": 0.3,
}


class _SpyScanner:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.scan_calls: list[tuple[Any, ...]] = []
        self.thin_calls: list[dict[str, Any]] = []
        self.ft = np.full(shape, 0.5, dtype=np.float32)
        self.pt = np.full(shape, 10.0, dtype=np.float32)
        self.tt = np.full(shape, 70.0, dtype=np.float32)
        self.thinned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def scan(
        self,
        phi_min: float,
        phi_max: float,
        theta_min: float,
        theta_max: float,
        ep: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.scan_calls.append((phi_min, phi_max, theta_min, theta_max, ep))
        return self.ft, self.pt, self.tt

    def thin(
        self,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mode = str(kwargs["mode"])
        likelihood = np.full_like(ft, 1.0 if mode == "reference" else 2.0)
        strike = np.full_like(pt, 20.0 if mode == "reference" else 30.0)
        dip = np.full_like(tt, 60.0 if mode == "reference" else 65.0)
        result = (likelihood, strike, dip)
        self.thinned[mode] = result
        self.thin_calls.append(
            {
                "ft": ft,
                "pt": pt,
                "tt": tt,
                **kwargs,
            }
        )
        return result


class _SpyVoter:
    def __init__(self, *, branch_index: int, ru: int, rv: int, rw: int) -> None:
        self.branch_index = branch_index
        self.radii = (ru, rv, rw)
        self.strain_calls: list[tuple[float, float]] = []
        self.surface_smoothing_calls: list[tuple[float, float]] = []
        self.surface_support_calls: list[dict[str, float]] = []
        self.boundary_policy_calls: list[str] = []
        self.orientation_smoothing_calls: list[float] = []
        self.final_smoothing_calls: list[float] = []
        self.apply_calls: list[dict[str, Any]] = []
        self.thin_calls: list[dict[str, Any]] = []

    def set_strain_max(self, value1: float, value2: float) -> None:
        self.strain_calls.append((value1, value2))

    def set_surface_smoothing(self, value1: float, value2: float) -> None:
        self.surface_smoothing_calls.append((value1, value2))

    def set_surface_support_policy(self, **kwargs: float) -> None:
        self.surface_support_calls.append(kwargs)

    def set_surface_voting_boundary_policy(self, value: str) -> None:
        self.boundary_policy_calls.append(value)

    def set_surface_orientation_smoothing(self, value: float) -> None:
        self.orientation_smoothing_calls.append(value)

    def set_final_normalization_smoothing(self, value: float) -> None:
        self.final_smoothing_calls.append(value)

    def apply_voting(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.apply_calls.append(kwargs)
        ft = np.asarray(kwargs["ft"])
        fv = np.full_like(ft, float(self.branch_index + 3))
        vp = np.full_like(ft, float(self.branch_index + 30))
        vt = np.full_like(ft, float(self.branch_index + 60))
        return fv, vp, vt

    def thin(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        self.thin_calls.append(
            {
                "fv": fv,
                "vp": vp,
                "vt": vt,
                **kwargs,
            }
        )
        return np.asarray(fv).copy()


def _policy_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        policy.build_policy_config(policy.BASELINE_POLICY, **POLICY_CONFIG_KWARGS),
        policy.build_policy_config(policy.CANDIDATE_POLICY, **POLICY_CONFIG_KWARGS),
    )


def _outputs_with_i2_planes(
    *plane_indices: int,
    shape: tuple[int, int, int] = (9, 9, 9),
) -> dict[str, np.ndarray]:
    ridge = np.zeros(shape, dtype=np.float32)
    for index in plane_indices:
        ridge[:, index, :] = np.float32(1.0)
    return {
        "ft_py.dat": np.ones(shape, dtype=np.float32),
        "pt_py.dat": np.zeros(shape, dtype=np.float32),
        "tt_py.dat": np.full(shape, 90.0, dtype=np.float32),
        "fet_py.dat": ridge.copy(),
        "fpt_py.dat": np.zeros(shape, dtype=np.float32),
        "ftt_py.dat": np.full(shape, 90.0, dtype=np.float32),
        "fv_py.dat": ridge.copy(),
        "vp_py.dat": np.zeros(shape, dtype=np.float32),
        "vt_py.dat": np.full(shape, 90.0, dtype=np.float32),
        "fvt_py.dat": ridge.copy(),
    }


def _finite_report(values: np.ndarray) -> dict[str, int]:
    array = np.asarray(values)
    return {
        "size": int(array.size),
        "finite_count": int(np.count_nonzero(np.isfinite(array))),
    }


def _crop_report(
    outputs: Mapping[str, np.ndarray],
    *,
    index: int = 1,
    public_distance_p95: float = 2.0,
) -> dict[str, Any]:
    return {
        "index": index,
        "stage_density": policy.build_stage_density_report(outputs, interior_margin=1),
        "finite_checks": {
            "pyosv": {
                name.removesuffix(".dat"): _finite_report(values)
                for name, values in outputs.items()
            }
        },
        "sparse_ridge_distance_metrics": {
            "interior": {
                "fvt": {
                    "candidate_to_reference_p95": float(public_distance_p95),
                }
            }
        },
    }


def _comparison_evidence(
    baseline_outputs: Mapping[str, np.ndarray],
    candidate_outputs: Mapping[str, np.ndarray],
    *,
    baseline_distance_p95: float = 2.0,
    candidate_distance_p95: float = 3.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        _crop_report(baseline_outputs, public_distance_p95=baseline_distance_p95),
        _crop_report(candidate_outputs, public_distance_p95=candidate_distance_p95),
        policy.build_direct_policy_comparison(
            baseline_outputs,
            candidate_outputs,
            interior_margin=1,
        ),
    )


def _validate(
    baseline_crop: Mapping[str, Any],
    candidate_crop: Mapping[str, Any],
    direct_comparison: Mapping[str, Any],
    *,
    scanner_execution_count: int = 1,
    baseline_config: Mapping[str, Any] | None = None,
    candidate_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    default_baseline_config, default_candidate_config = _policy_configs()
    return policy.validate_policy_comparison(
        baseline_crops=[baseline_crop],
        candidate_crops=[candidate_crop],
        direct_comparisons=[direct_comparison],
        baseline_config=(default_baseline_config if baseline_config is None else baseline_config),
        candidate_config=(
            default_candidate_config if candidate_config is None else candidate_config
        ),
        scanner_execution_count=scanner_execution_count,
        expected_crop_count=1,
    )


def test_fixed_policy_ids_profile_and_quality_configuration() -> None:
    assert policy.COMPARISON_PROFILE == "quality-workflow-scanner-thinning-v1"
    assert policy.BASELINE_POLICY_ID == "quality_reference_like_scanner_thin_reference_v1"
    assert policy.CANDIDATE_POLICY_ID == "quality_reference_like_scanner_thin_normal_v1"
    assert policy.policy_definition("baseline") is policy.BASELINE_POLICY
    assert policy.policy_definition("candidate") is policy.CANDIDATE_POLICY
    with pytest.raises(ValueError, match="baseline, candidate"):
        policy.policy_definition("unknown")

    baseline_config, candidate_config = _policy_configs()
    assert baseline_config["requested"]["scanner_thin_mode"] == "reference"
    assert candidate_config["requested"]["scanner_thin_mode"] == "normal"
    assert baseline_config["effective"]["effective_remove_edge_effects"] is True
    assert candidate_config["effective"]["effective_remove_edge_effects"] is None
    assert baseline_config["effective"]["voter_thin_mode"] == "hybrid_v2"
    assert baseline_config["effective"]["surface_support_min_fraction"] == 0.0
    assert baseline_config["effective"]["surface_support_exponent"] == 0.0
    assert baseline_config["effective"]["surface_voting_boundary_policy"] == "reference"
    assert baseline_config["effective"]["final_normalization_smoothing"] == 0.0

    check = policy.validate_configuration_contract(
        baseline_config=baseline_config,
        candidate_config=candidate_config,
    )
    assert check["passed"] is True
    assert check["requested_difference_paths"] == ["scanner_thin_mode"]
    assert check["effective_difference_paths"] == [
        "effective_remove_edge_effects",
        "scanner_thin_mode",
    ]
    assert policy.recursive_difference_paths(
        {"outer": {"value": 1}},
        {"outer": {"value": 2}},
    ) == ["outer.value"]
    assert policy.recursive_difference_paths(
        {"outer": {"value": True}},
        {"outer": {"value": 1}},
    ) == ["outer.value"]


def test_shared_scan_pipeline_uses_one_scan_and_independent_quality_voters() -> None:
    shape = (4, 5, 6)
    ep = np.zeros(shape, dtype=np.float32)
    scanner = _SpyScanner(shape)
    scanner_factory_calls: list[dict[str, float]] = []
    voters: list[_SpyVoter] = []

    def scanner_factory(**kwargs: float) -> _SpyScanner:
        scanner_factory_calls.append(kwargs)
        return scanner

    def voter_factory(*, ru: int, rv: int, rw: int) -> _SpyVoter:
        voter = _SpyVoter(branch_index=len(voters), ru=ru, rv=rv, rw=rw)
        voters.append(voter)
        return voter

    result = policy.run_shared_scan_policy_pipeline(
        ep,
        sigma1=2.0,
        sigma2=3.0,
        phi_min=10.0,
        phi_max=20.0,
        theta_min=60.0,
        theta_max=80.0,
        ru=1,
        rv=2,
        rw=3,
        strain_max1=0.2,
        strain_max2=0.3,
        surface_smoothing1=1.5,
        surface_smoothing2=2.5,
        surface_orientation_smoothing=4.0,
        final_normalization_smoothing=0.75,
        d=2,
        fm=0.4,
        reference_thin_sigma=1.25,
        scanner_factory=scanner_factory,
        voter_factory=voter_factory,
    )

    assert result["scanner_execution_count"] == 1
    assert scanner_factory_calls == [{"sigma1": 2.0, "sigma2": 3.0}]
    assert len(scanner.scan_calls) == 1
    assert scanner.scan_calls[0][:4] == (10.0, 20.0, 60.0, 80.0)
    assert scanner.scan_calls[0][4] is ep
    assert [call["mode"] for call in scanner.thin_calls] == ["reference", "normal"]
    assert all(call["remove_edge_effects"] is True for call in scanner.thin_calls)
    assert all(call["reference_sigma"] == 1.25 for call in scanner.thin_calls)

    assert len(voters) == 2
    assert voters[0] is not voters[1]
    for voter in voters:
        assert voter.radii == (1, 2, 3)
        assert voter.strain_calls == [(0.2, 0.3)]
        assert voter.surface_smoothing_calls == [(1.5, 2.5)]
        assert voter.surface_support_calls == [{"min_fraction": 0.0, "exponent": 0.0}]
        assert voter.boundary_policy_calls == ["reference"]
        assert voter.orientation_smoothing_calls == [4.0]
        assert voter.final_smoothing_calls == [0.75]
        assert voter.apply_calls[0]["d"] == 2
        assert voter.apply_calls[0]["fm"] == 0.4
        assert voter.thin_calls[0]["mode"] == "hybrid_v2"
        assert voter.thin_calls[0]["reference_sigma"] == 1.25

    for branch_index, role in enumerate(policy.POLICY_ROLES):
        mode = "reference" if role == "baseline" else "normal"
        branch = result["policies"][role]
        outputs = branch["outputs"]
        fet, fpt, ftt = scanner.thinned[mode]
        voter = voters[branch_index]
        assert set(outputs) == set(policy.OUTPUT_NAMES)
        assert branch["policy_id"] == policy.policy_definition(role).policy_id
        assert outputs["ft_py.dat"] is scanner.ft
        assert outputs["pt_py.dat"] is scanner.pt
        assert outputs["tt_py.dat"] is scanner.tt
        assert outputs["fet_py.dat"] is fet
        assert outputs["fpt_py.dat"] is fpt
        assert outputs["ftt_py.dat"] is ftt
        assert voter.apply_calls[0]["ft"] is fet
        assert voter.apply_calls[0]["pt"] is fpt
        assert voter.apply_calls[0]["tt"] is ftt
        assert voter.thin_calls[0]["plateau_tie_breaker"] is fet


def test_stage_density_direct_metrics_consensus_and_validation_pass() -> None:
    baseline_outputs = _outputs_with_i2_planes(4)
    candidate_outputs = _outputs_with_i2_planes(5)
    baseline_crop, candidate_crop, direct = _comparison_evidence(
        baseline_outputs,
        candidate_outputs,
    )

    assert baseline_crop["stage_density"]["fet"]["nonzero_count"] == 81
    assert baseline_crop["stage_density"]["fv"]["nonzero_fraction"] == pytest.approx(1.0 / 9.0)
    assert baseline_crop["stage_density"]["fvt"]["edge_density_proxy"] == 0.0
    assert direct["fvt_density"]["candidate_over_baseline_ratio"] == 1.0
    assert direct["buffered_ridge_overlap"]["interior"]["buffered_precision"] == 1.0
    assert direct["buffered_ridge_overlap"]["interior"]["buffered_recall"] == 1.0
    assert direct["sparse_ridge_distance_metrics"]["interior"]["candidate_to_reference_p95"] == 1.0
    assert direct["ridge_mask_difference"]["candidate_only_fraction"] == 1.0
    assert direct["ridge_mask_difference"]["baseline_only_fraction"] == 1.0
    assert direct["ridge_mask_difference"]["edge_shell_candidate_only_fraction"] > 0.0

    consensus = policy.build_consensus(
        baseline_crops=[baseline_crop],
        candidate_crops=[candidate_crop],
        direct_comparisons=[direct],
    )
    assert consensus["policies"]["baseline"]["stage_density"]["fvt"]["mean"] == (
        pytest.approx(1.0 / 9.0)
    )
    assert consensus["candidate_minus_baseline"]["fvt_density_ratio"] == 1.0
    assert consensus["candidate_minus_baseline"]["public_fvt_sparse_distance_p95_delta_mean"] == 1.0
    assert (
        consensus["candidate_minus_baseline"]["direct_comparison"]["candidate_only_fraction"][
            "mean"
        ]
        == 1.0
    )

    validation = _validate(baseline_crop, candidate_crop, direct)
    assert validation["role"] == "truthless_external_smoke"
    assert validation["passed"] is True
    assert validation["reasons"] == []
    assert all(check["passed"] for check in validation["checks"].values())

    serialized = policy.report_to_json(
        {
            "format_version": np.int64(1),
            "consensus": consensus,
            "policy_validation": validation,
            "diagnostic": np.asarray([1.0, np.nan, np.inf], dtype=np.float32),
        },
        pretty=True,
    )
    loaded = json.loads(serialized)
    assert serialized.endswith("\n")
    assert loaded["format_version"] == 1
    assert loaded["diagnostic"] == [1.0, None, None]
    assert loaded["policy_validation"]["passed"] is True


def test_validation_rejects_wrong_scanner_execution_count() -> None:
    baseline_crop, candidate_crop, direct = _comparison_evidence(
        _outputs_with_i2_planes(4),
        _outputs_with_i2_planes(5),
    )

    validation = _validate(
        baseline_crop,
        candidate_crop,
        direct,
        scanner_execution_count=2,
    )

    assert validation["passed"] is False
    assert validation["checks"]["shared_scan_contract"]["passed"] is False
    assert validation["checks"]["shared_scan_contract"]["scanner_execution_count"] == 2


def test_validation_rejects_a_nonfinite_output_stage() -> None:
    baseline_crop, candidate_crop, direct = _comparison_evidence(
        _outputs_with_i2_planes(4),
        _outputs_with_i2_planes(5),
    )
    candidate_crop["finite_checks"]["pyosv"]["fpt_py"]["finite_count"] -= 1

    validation = _validate(baseline_crop, candidate_crop, direct)

    assert validation["passed"] is False
    check = validation["checks"]["finite_outputs"]
    assert check["passed"] is False
    assert check["failure_count"] == 1
    assert check["policy_failure_count"] == {"baseline": 0, "candidate": 1}


def test_validation_rejects_an_empty_stage_in_either_policy() -> None:
    baseline_outputs = _outputs_with_i2_planes(4)
    candidate_outputs = _outputs_with_i2_planes(5)
    candidate_outputs["fet_py.dat"].fill(0.0)
    baseline_crop, candidate_crop, direct = _comparison_evidence(
        baseline_outputs,
        candidate_outputs,
    )

    validation = _validate(baseline_crop, candidate_crop, direct)

    assert validation["passed"] is False
    check = validation["checks"]["nonempty_stages"]
    assert check["passed"] is False
    assert check["failures"]["baseline"] == []
    assert check["failures"]["candidate"] == [{"crop_index": 1, "stage": "fet_py.dat"}]


def test_validation_rejects_worst_crop_density_ratio_when_aggregate_passes() -> None:
    baseline_crop1, candidate_crop1, direct1 = _comparison_evidence(
        _outputs_with_i2_planes(4),
        _outputs_with_i2_planes(3, 4, 5),
    )
    baseline_crop2, candidate_crop2, direct2 = _comparison_evidence(
        _outputs_with_i2_planes(4),
        _outputs_with_i2_planes(5),
    )
    baseline_crop2["index"] = 2
    candidate_crop2["index"] = 2
    baseline_config, candidate_config = _policy_configs()

    validation = policy.validate_policy_comparison(
        baseline_crops=[baseline_crop1, baseline_crop2],
        candidate_crops=[candidate_crop1, candidate_crop2],
        direct_comparisons=[direct1, direct2],
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        scanner_execution_count=2,
        expected_crop_count=2,
    )

    assert validation["passed"] is False
    check = validation["checks"]["fvt_density_ratio"]
    assert check["passed"] is False
    assert check["per_crop"] == [3.0, 1.0]
    assert check["aggregate"] == 2.0
    assert check["minimum"] == 0.5
    assert check["maximum"] == 2.0


def test_validation_rejects_unexpected_configuration_difference() -> None:
    baseline_crop, candidate_crop, direct = _comparison_evidence(
        _outputs_with_i2_planes(4),
        _outputs_with_i2_planes(5),
    )
    baseline_config, candidate_config = _policy_configs()
    invalid_candidate_config = copy.deepcopy(candidate_config)
    invalid_candidate_config["requested"]["ru"] = 99
    invalid_candidate_config["effective"]["ru"] = 99

    validation = _validate(
        baseline_crop,
        candidate_crop,
        direct,
        baseline_config=baseline_config,
        candidate_config=invalid_candidate_config,
    )

    assert validation["passed"] is False
    check = validation["checks"]["configuration_contract"]
    assert check["passed"] is False
    assert check["requested_difference_paths"] == ["ru", "scanner_thin_mode"]
    assert check["effective_difference_paths"] == [
        "effective_remove_edge_effects",
        "ru",
        "scanner_thin_mode",
    ]

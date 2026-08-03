from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.reskin_policy_comparison as comparison_module
import pyosv.evaluation.f3d_mode_comparison.result as result_module
import pyosv.evaluation.f3d_mode_comparison.runner as runner_module
from pyosv.cli import f3d_mode_comparison
from pyosv.evaluation.dense_reskin_quality import (
    build_dense_reskin_promotion_gate,
    controlled_dense_reskin_cases,
)
from pyosv.evaluation.f3d_mode_comparison import (
    F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE,
    F3_RESKIN_POLICY_COMPARISON_COMPLETION_SCHEMA_VERSION,
    F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION,
    canonical_fingerprint,
    compare_reskin_policies_from_bundle,
    validate_completed_f3d_bundle,
    validate_f3_reskin_policy_comparison,
)
from pyosv.evaluation.workflow3d import execute_skinning_phase3d
from pyosv.synthetic_metrics import metrics_from_surface_evidence

from .test_final_acceptance_integration import (
    _fixed_runtime_identity,
    _generate_official_bundle,
    _official_fixture,
)
from .test_integration import _run_fixture, _write_fixture
from .test_bundle_validation import (
    _controlled_reskinned_primary_workflow,
    _reskinned_primary_config,
)
from .test_publication_contract_v3_integration import (
    _historical_contract4_reskinned_workflow,
    _historical_reskinned_workflow,
    _install_historical_skin_contract_writer,
    _tree_bytes,
)


def _file_state(paths: tuple[Path, ...]) -> dict[Path, tuple[int, int, bytes]]:
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes()) for path in paths
    }


def _write_rehashed_comparison(
    report: dict[str, Any],
    paths: tuple[Path, ...],
    completion_path: Path,
    completion: dict[str, Any],
) -> None:
    paths[0].write_text(comparison_module._canonical_json(report) + "\n", encoding="utf-8")
    paths[1].write_text(comparison_module._comparison_csv(report), encoding="utf-8")
    paths[2].write_text(comparison_module._comparison_markdown(report), encoding="utf-8")
    for policy, path in zip(
        ("existing_cells_v1", "reference_dense_v1"),
        paths[3:],
        strict=True,
    ):
        path.write_text(
            comparison_module._canonical_json(report["policies"][policy]["canonical_skin_artifact"])
            + "\n",
            encoding="utf-8",
        )
    completion["report_semantic_evidence_sha256"] = (
        comparison_module._report_semantic_evidence_digest(report)
    )
    completion["artifact_files"] = {
        path.name: comparison_module.artifact_file_metadata(path) for path in paths
    }
    completion_path.write_text(
        comparison_module._canonical_json(completion) + "\n",
        encoding="utf-8",
    )


def _comparison_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workflow_runner: Any = _controlled_reskinned_primary_workflow,
    plan_config: Any | None = None,
) -> tuple[
    Path,
    Any,
    dict[str, Any],
    tuple[Path, ...],
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    data_root = tmp_path / "data"
    output_root = tmp_path / "bundle"
    spec = _official_fixture(data_root, shape=(15, 15, 15))
    runtime_identity = _fixed_runtime_identity()
    source = _generate_official_bundle(
        data_root,
        output_root,
        spec,
        runtime_identity,
        Counter(),
        monkeypatch,
        workflow_runner=workflow_runner,
        plan_config=plan_config,
    )
    original_load = result_module.load_f3d_mode_comparison_result

    def load_fixture(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("_dataset_spec", spec)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(result_module, "load_f3d_mode_comparison_result", load_fixture)
    monkeypatch.setattr(
        comparison_module,
        "numerical_runtime_identity",
        lambda: deepcopy(runtime_identity),
    )
    paths = compare_reskin_policies_from_bundle(output_root)
    completion_path = paths[0].parent / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE
    return (
        output_root,
        source,
        runtime_identity,
        paths,
        completion_path,
        json.loads(paths[0].read_text(encoding="utf-8")),
        json.loads(completion_path.read_text(encoding="utf-8")),
    )


def _controlled_discarding_workflow(**kwargs: Any) -> Any:
    base = _controlled_reskinned_primary_workflow(**kwargs)
    fv = base.fv.copy()
    fv[12, 12, 12] = np.float32(0.9)
    fvt = fv.copy()
    skin = execute_skinning_phase3d(
        fv=fv,
        fvt=fvt,
        vp=base.vp,
        vt=base.vt,
        skinning_settings=kwargs["skinning_settings"],
        variant_spec=kwargs["variant_spec"],
        scanner_target_positive_mask=None,
        boundary_fallback_runner=kwargs["boundary_fallback_runner"],
    )
    return replace(
        base,
        fv=fv,
        fvt=fvt,
        skin=skin,
        diagnostics=replace(base.diagnostics, skinning=skin.diagnostics),
    )


def test_comparison_implementation_identity_tracks_direct_artifact_modules() -> None:
    modules = comparison_module._comparison_implementation_identity()["algorithm_modules"]

    assert {
        "evaluation/f3d_mode_comparison/artifacts.py",
        "evaluation/f3d_mode_comparison/runtime_identity.py",
        "evaluation/f3d_mode_comparison/skin_artifacts.py",
    } <= modules.keys()


def test_skin_artifact_source_change_changes_comparison_implementation_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_identity = comparison_module._comparison_implementation_identity()
    source_path = Path(comparison_module.__file__).with_name("skin_artifacts.py")
    changed_source = tmp_path / "skin_artifacts.py"
    changed_source.write_bytes(source_path.read_bytes() + b"\n# changed comparison source\n")

    original_implementation_identity = comparison_module.implementation_identity

    def changed_identity(*, source_files: Any = None) -> dict[str, Any]:
        assert source_files is not None
        files = dict(source_files)
        files["evaluation/f3d_mode_comparison/skin_artifacts.py"] = changed_source
        return original_implementation_identity(source_files=files)

    monkeypatch.setattr(comparison_module, "implementation_identity", changed_identity)
    changed_identity_payload = comparison_module._comparison_implementation_identity()

    skin_module = "evaluation/f3d_mode_comparison/skin_artifacts.py"
    assert (
        source_identity["algorithm_modules"][skin_module]
        != changed_identity_payload["algorithm_modules"][skin_module]
    )
    assert canonical_fingerprint(source_identity) != canonical_fingerprint(changed_identity_payload)


def test_comparison_resume_rejects_stale_implementation_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        output_root,
        source,
        _runtime_identity,
        paths,
        completion_path,
        report,
        completion,
    ) = _comparison_fixture(tmp_path, monkeypatch)
    stale_report = deepcopy(report)
    stale_completion = deepcopy(completion)
    skin_module = "evaluation/f3d_mode_comparison/skin_artifacts.py"
    stale_module = stale_report["comparison_implementation_identity"]["algorithm_modules"][
        skin_module
    ]
    stale_module["sha256"] = "0" * 64
    stale_identity_digest = canonical_fingerprint(
        stale_report["comparison_implementation_identity"]
    )
    stale_completion["comparison_implementation_identity_sha256"] = stale_identity_digest
    _write_rehashed_comparison(stale_report, paths, completion_path, stale_completion)

    with pytest.raises(ValueError, match="implementation identity mismatch"):
        compare_reskin_policies_from_bundle(output_root, resume=True)

    assert (
        source.run_fingerprint
        == json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))[
            "run_fingerprint"
        ]
    )


def test_dense_reskin_comparison_strict_deep_and_complete_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        output_root,
        source,
        runtime_identity,
        paths,
        completion_path,
        report,
        completion,
    ) = _comparison_fixture(tmp_path, monkeypatch)
    assert [cell.label for cell in source.cells] == [
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    q_qual = next(cell for cell in source.cells if cell.label == "Q-QUAL")
    runtime_sha256 = canonical_fingerprint(runtime_identity)

    assert report["schema_version"] == F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION
    assert (
        completion["completion_schema_version"]
        == F3_RESKIN_POLICY_COMPARISON_COMPLETION_SCHEMA_VERSION
    )
    assert report["validation_level"] == completion["validation_level"] == "shallow"
    assert report["source_run_fingerprint"] == source.run_fingerprint
    assert report["source_runtime_identity_sha256"] == runtime_sha256
    assert report["comparison_runtime_identity_sha256"] == runtime_sha256
    assert report["upstream_parent_stage_fingerprints"] == {
        "scanner": q_qual.stages.scanner,
        "voting": q_qual.stages.voting,
        "thinning": q_qual.stages.thinning,
    }
    effective = report["resolved_config"]["effective_skinning_configs"]
    baseline = dict(effective["existing_cells_v1"])
    candidate = dict(effective["reference_dense_v1"])
    assert baseline.pop("reskin_policy") == "existing_cells_v1"
    assert candidate.pop("reskin_policy") == "reference_dense_v1"
    assert baseline == candidate
    assert report["comparison_config_fingerprint"] == canonical_fingerprint(
        report["resolved_config"]
    )
    assert all(
        policy["canonical_skin_artifact"]["format_version"] == 2
        for policy in report["policies"].values()
    )
    for policy in report["policies"].values():
        assert policy["canonical_skin_artifact"]["skins"]
        assert policy["parent_ridge_surface"]["evidence"]["surface_distance"][
            "candidate_to_reference"
        ]["quantiles"]
        final = policy["diagnostics"]["reskin"]
        attempted = final["attempted"]
        assert final["processed_skin_count"] <= attempted["processed_skin_count"]
        assert final["output_cell_count"] <= attempted["output_cell_count"]

    parent_reads = Counter()
    original_open_parent = comparison_module._open_parent_dat

    def tracked_open_parent(*args: Any, **kwargs: Any) -> Any:
        parent_reads["DAT"] += 1
        return original_open_parent(*args, **kwargs)

    monkeypatch.setattr(comparison_module, "_open_parent_dat", tracked_open_parent)
    monkeypatch.setattr(
        comparison_module,
        "numerical_runtime_identity",
        lambda: pytest.fail("shallow validation queried the current runtime"),
    )
    assert validate_f3_reskin_policy_comparison(output_root) == paths
    assert compare_reskin_policies_from_bundle(output_root, resume=True) == paths
    assert not parent_reads

    monkeypatch.setattr(
        comparison_module,
        "numerical_runtime_identity",
        lambda: deepcopy(runtime_identity),
    )
    assert compare_reskin_policies_from_bundle(output_root, resume=True, deep=True) == paths
    assert parent_reads == Counter({"DAT": 5})
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["validation_level"] == "deep"
    assert validate_f3_reskin_policy_comparison(output_root, require_deep=True) == paths

    artifact_paths = (*paths, completion_path)
    before_resume = _file_state(artifact_paths)
    parent_reads.clear()
    monkeypatch.setattr(
        comparison_module,
        "numerical_runtime_identity",
        lambda: pytest.fail("deep-complete resume queried the current runtime"),
    )
    assert compare_reskin_policies_from_bundle(output_root, resume=True, deep=True) == paths
    assert not parent_reads
    assert _file_state(artifact_paths) == before_resume


def test_link_topology_algebra_tamper_is_shallow_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root, _, _, paths, completion_path, report, completion = _comparison_fixture(
        tmp_path,
        monkeypatch,
    )
    policy = report["policies"]["existing_cells_v1"]
    topology = policy["skin_topology"]
    parsed_skin_count = len(policy["canonical_skin_artifact"]["skins"])
    cell_count = topology["cell_count"]
    single_cell_skin_count = sum(
        skin["cell_count"] == 1 for skin in policy["canonical_skin_artifact"]["skins"]
    )
    original_links = policy["link_topology"]
    alternate = None
    for component_count in range(parsed_skin_count, cell_count + 1):
        for isolated_cell_count in range(single_cell_skin_count, component_count + 1):
            if component_count == cell_count and isolated_cell_count != cell_count:
                continue
            if isolated_cell_count == cell_count and component_count != cell_count:
                continue
            if 2 * (component_count - isolated_cell_count) > cell_count - isolated_cell_count:
                continue
            candidate = (component_count, isolated_cell_count)
            if candidate != (
                original_links["linked_component_count"],
                original_links["isolated_cell_count"],
            ):
                alternate = candidate
                break
        if alternate is not None:
            break
    assert alternate is not None

    tampered_report = deepcopy(report)
    tampered_links = tampered_report["policies"]["existing_cells_v1"]["link_topology"]
    tampered_links["linked_component_count"], tampered_links["isolated_cell_count"] = alternate
    tampered_completion = deepcopy(completion)
    _write_rehashed_comparison(
        tampered_report,
        paths,
        completion_path,
        tampered_completion,
    )

    assert validate_f3_reskin_policy_comparison(output_root) == paths
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_f3_reskin_policy_comparison(output_root, deep=True)


def test_source_bindings_and_rehashed_tamper_rejection_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        output_root,
        source,
        runtime_identity,
        paths,
        completion_path,
        original_report,
        original_completion,
    ) = _comparison_fixture(tmp_path, monkeypatch)
    snapshots = _file_state((*paths, completion_path))
    parent_reads: Counter[str] = Counter()
    original_open_parent = comparison_module._open_parent_dat

    def tracked_open_parent(*args: Any, **kwargs: Any) -> Any:
        parent_reads["DAT"] += 1
        return original_open_parent(*args, **kwargs)

    monkeypatch.setattr(comparison_module, "_open_parent_dat", tracked_open_parent)

    def restore() -> None:
        for path, (_, _, payload) in snapshots.items():
            path.write_bytes(payload)
        parent_reads.clear()

    manifest_path = output_root / "run_manifest.json"
    manifest_snapshot = manifest_path.read_bytes()
    for name, mutate, expected_error in (
        (
            "run-fingerprint",
            lambda manifest: manifest["implementation_identity"].__setitem__(
                "fixture_source_binding", "changed"
            ),
            "source run fingerprint",
        ),
        (
            "runtime-identity",
            lambda manifest: manifest["runtime_identity"].__setitem__(
                "platform_machine", "changed-valid-machine"
            ),
            "source runtime digest",
        ),
    ):
        manifest = json.loads(manifest_snapshot)
        mutate(manifest)
        manifest["run_fingerprint"] = canonical_fingerprint(
            {field: manifest[field] for field in comparison_module._RUN_COMPUTATION_FIELDS}
        )
        manifest_path.write_text(
            comparison_module._canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        if name == "runtime-identity":
            report = deepcopy(original_report)
            completion = deepcopy(original_completion)
            report["source_run_fingerprint"] = manifest["run_fingerprint"]
            completion["source_run_fingerprint"] = manifest["run_fingerprint"]
            _write_rehashed_comparison(report, paths, completion_path, completion)
        with pytest.raises(ValueError, match=expected_error):
            validate_f3_reskin_policy_comparison(output_root, result=source)
        assert not parent_reads
        manifest_path.write_bytes(manifest_snapshot)
        restore()

    source_loads: Counter[str] = Counter()
    production_loader = result_module.load_f3d_mode_comparison_result

    def tracked_source_loader(*args: Any, **kwargs: Any) -> Any:
        source_loads["bundle"] += 1
        return production_loader(*args, **kwargs)

    monkeypatch.setattr(
        result_module,
        "load_f3d_mode_comparison_result",
        tracked_source_loader,
    )
    cells_report_path = output_root / "reports" / "cells.json"
    q_qual_path = output_root / "cells" / "Q-QUAL.json"
    bundle_completion_path = output_root / "completion.json"
    source_snapshots = _file_state((cells_report_path, q_qual_path, bundle_completion_path))

    def write_source_cell_tamper(field: str) -> None:
        cells_report = json.loads(cells_report_path.read_text(encoding="utf-8"))
        report_cell = next(cell for cell in cells_report["cells"] if cell["label"] == "Q-QUAL")
        cell_payload = json.loads(q_qual_path.read_text(encoding="utf-8"))
        if field == "resolved_config":
            report_cell["resolved_config"]["skinning"]["min_skin_size"] += 1
            cell_payload["resolved_config"]["skinning"]["min_skin_size"] += 1
        else:
            report_cell["stages"]["thinning"] = "e" * 64
            cell_payload["stages"]["thinning"] = "e" * 64
        cells_report_path.write_text(
            comparison_module._canonical_json(cells_report) + "\n",
            encoding="utf-8",
        )
        q_qual_path.write_text(
            comparison_module._canonical_json(cell_payload) + "\n",
            encoding="utf-8",
        )
        bundle_completion = json.loads(bundle_completion_path.read_text(encoding="utf-8"))
        bundle_completion["report_files"]["cells.json"] = comparison_module.artifact_file_metadata(
            cells_report_path
        )
        bundle_completion_path.write_text(
            comparison_module._canonical_json(bundle_completion) + "\n",
            encoding="utf-8",
        )

    for field, expected_error in (
        ("resolved_config", "resolved_config"),
        ("parent_stage", "(?:stage|artifact)"),
    ):
        write_source_cell_tamper(field)
        with pytest.raises(ValueError, match=expected_error):
            validate_f3_reskin_policy_comparison(output_root)
        assert source_loads == Counter({"bundle": 1})
        assert not parent_reads
        for path, (_, _, payload) in source_snapshots.items():
            path.write_bytes(payload)
        source_loads.clear()

    def policy(report: dict[str, Any]) -> dict[str, Any]:
        return report["policies"]["existing_cells_v1"]

    def first_cell(report: dict[str, Any]) -> dict[str, Any]:
        return policy(report)["canonical_skin_artifact"]["skins"][0]["cells"][0]

    def mutate_resolved_config(report: dict[str, Any], completion: dict[str, Any]) -> None:
        report["resolved_config"]["shared_skinning_config"]["min_skin_size"] += 1
        report["comparison_config_fingerprint"] = canonical_fingerprint(report["resolved_config"])
        completion["comparison_config_fingerprint"] = report["comparison_config_fingerprint"]

    def mutate_overlap(report: dict[str, Any], _: dict[str, Any]) -> None:
        policy(report)["parent_ridge_surface"]["evidence"]["overlap"][
            "exact_intersection_count"
        ] += 1

    def mutate_distance(report: dict[str, Any], _: dict[str, Any]) -> None:
        policy(report)["parent_ridge_surface"]["evidence"]["surface_distance"][
            "candidate_to_reference"
        ]["mean_numerator"] += 1.0

    def mutate_quantile(report: dict[str, Any], _: dict[str, Any]) -> None:
        quantiles = policy(report)["parent_ridge_surface"]["evidence"]["surface_distance"][
            "candidate_to_reference"
        ]["quantiles"]
        quantiles["median"]["rank"] += 0.25

    def mutate_metric(report: dict[str, Any], _: dict[str, Any]) -> None:
        policy(report)["parent_ridge_surface"]["metrics"]["overlap"]["buffered_precision"] += 0.125

    def mutate_contrast(report: dict[str, Any], _: dict[str, Any]) -> None:
        report["contrast"]["generated_cell_count_delta"] += 1

    def mutate_final_diagnostics(report: dict[str, Any], _: dict[str, Any]) -> None:
        policy(report)["diagnostics"]["reskin"]["output_cell_count"] += 1

    def mutate_attempted_diagnostics(report: dict[str, Any], _: dict[str, Any]) -> None:
        policy(report)["diagnostics"]["reskin"]["attempted"]["output_cell_count"] += 1

    def mutate_generation(report: dict[str, Any], _: dict[str, Any]) -> None:
        first_cell(report)["generation"] = "connected_component"

    def mutate_support(report: dict[str, Any], _: dict[str, Any]) -> None:
        cell = first_cell(report)
        cell["reskin_support"] = int(cell["reskin_support"] or 0) + 1

    tamper_cases = (
        (
            "comparison runtime digest",
            lambda report, completion: (
                report.__setitem__("comparison_runtime_identity_sha256", "e" * 64),
                completion.__setitem__("comparison_runtime_identity_sha256", "e" * 64),
            ),
            "runtime digest mismatch",
        ),
        (
            "comparison implementation identity",
            lambda report, completion: report["comparison_implementation_identity"].__setitem__(
                "name", "tampered"
            ),
            "implementation identity mismatch",
        ),
        ("resolved config", mutate_resolved_config, "resolved config mismatch"),
        (
            "config fingerprint",
            lambda report, completion: (
                report.__setitem__("comparison_config_fingerprint", "e" * 64),
                completion.__setitem__("comparison_config_fingerprint", "e" * 64),
            ),
            "config fingerprint mismatch",
        ),
        ("overlap count evidence", mutate_overlap, "inconsistent"),
        ("distance accumulator evidence", mutate_distance, "inconsistent"),
        ("quantile evidence", mutate_quantile, "rank mismatch"),
        ("derived metric", mutate_metric, "metrics/evidence mismatch"),
        ("derived contrast", mutate_contrast, "contrast metrics mismatch"),
        ("final diagnostics", mutate_final_diagnostics, "reskin diagnostics are invalid"),
        (
            "attempted diagnostics",
            mutate_attempted_diagnostics,
            "reskin diagnostics are invalid",
        ),
        (
            "canonical generation",
            mutate_generation,
            "(?:generation provenance|output_cell_count does not match skins)",
        ),
        (
            "canonical support",
            mutate_support,
            "(?:support|generation provenance|metrics mismatch)",
        ),
        (
            "completion validation level",
            lambda report, completion: completion.__setitem__("validation_level", "deep"),
            "validation level mismatch",
        ),
    )
    assert len(tamper_cases) >= 12
    for _, mutate, expected_error in tamper_cases:
        report = deepcopy(original_report)
        completion = deepcopy(original_completion)
        mutate(report, completion)
        _write_rehashed_comparison(report, paths, completion_path, completion)
        with pytest.raises(ValueError, match=expected_error):
            validate_f3_reskin_policy_comparison(output_root, result=source)
        assert not parent_reads
        restore()

    report = deepcopy(original_report)
    completion = deepcopy(original_completion)
    surface = policy(report)["parent_ridge_surface"]
    surface["evidence"]["overlap"]["buffer_radius"] += 1.0
    surface["metrics"] = metrics_from_surface_evidence(surface["evidence"])
    report["contrast"] = comparison_module._comparison_contrast(report["policies"])
    _write_rehashed_comparison(report, paths, completion_path, completion)
    assert validate_f3_reskin_policy_comparison(output_root, result=source) == paths
    assert not parent_reads
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_f3_reskin_policy_comparison(
            output_root,
            deep=True,
            result=source,
        )
    assert parent_reads == Counter({"DAT": 5})
    restore()

    report["validation_level"] = "deep"
    completion["validation_level"] = "deep"
    _write_rehashed_comparison(report, paths, completion_path, completion)
    assert compare_reskin_policies_from_bundle(output_root, resume=True, deep=True) == paths
    assert not parent_reads
    restore()
    monkeypatch.setattr(
        comparison_module,
        "numerical_runtime_identity",
        lambda: deepcopy(runtime_identity),
    )


@pytest.mark.parametrize("min_skin_size", (2, 50), ids=("accepted-and-discarded", "all-discarded"))
def test_final_artifacts_exclude_discarded_attempts_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    min_skin_size: int,
) -> None:
    plan_config = _reskinned_primary_config()
    plan_config = replace(
        plan_config,
        skinning_template=replace(
            plan_config.skinning_template,
            method="quality",
            growth_source="pre_thin",
            min_skin_size=min_skin_size,
        ),
    )
    (
        output_root,
        _,
        runtime_identity,
        paths,
        _,
        _,
        _,
    ) = _comparison_fixture(
        tmp_path,
        monkeypatch,
        workflow_runner=_controlled_discarding_workflow,
        plan_config=plan_config,
    )
    assert validate_f3_reskin_policy_comparison(output_root) == paths
    monkeypatch.setattr(
        comparison_module,
        "numerical_runtime_identity",
        lambda: deepcopy(runtime_identity),
    )
    assert validate_f3_reskin_policy_comparison(output_root, deep=True) == paths

    report = json.loads(paths[0].read_text(encoding="utf-8"))
    for item, artifact_path in zip(
        report["policies"].values(),
        paths[3:],
        strict=True,
    ):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact == item["canonical_skin_artifact"]
        final = item["diagnostics"]["reskin"]
        attempted = final["attempted"]
        cells = [cell for skin in artifact["skins"] for cell in skin["cells"]]
        assert attempted["processed_skin_count"] > final["processed_skin_count"]
        assert attempted["output_cell_count"] > final["output_cell_count"]
        assert all((cell["i1"], cell["i2"], cell["i3"]) != (12, 12, 12) for cell in cells)
        if min_skin_size == 50:
            assert not cells
            assert final["processed_skin_count"] == final["output_cell_count"] == 0
        else:
            assert cells


@pytest.mark.parametrize("contract", (3, 4))
def test_historical_skin_artifacts_remain_read_only_when_comparison_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: int,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "bundle"
    spec = _write_fixture(data_root, shape=(13, 13, 13))
    runtime_identity = _fixed_runtime_identity()
    workflow_runner = _historical_contract4_reskinned_workflow
    if contract == 3:
        _install_historical_skin_contract_writer(monkeypatch)
        workflow_runner = _historical_reskinned_workflow
    else:
        monkeypatch.setattr(
            runner_module,
            "F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION",
            4,
        )
    plan_config = _reskinned_primary_config()
    plan_config = replace(
        plan_config,
        skinning_template=replace(
            plan_config.skinning_template,
            method="quality",
        ),
    )
    source = _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
        plan_config=plan_config,
        workspace_runtime_identity=runtime_identity,
        workflow_runner=workflow_runner,
    )
    before = _tree_bytes(output_root)
    original_load = result_module.load_f3d_mode_comparison_result

    def load_fixture(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("_dataset_spec", spec)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(result_module, "load_f3d_mode_comparison_result", load_fixture)
    monkeypatch.setattr(
        comparison_module,
        "numerical_runtime_identity",
        lambda: deepcopy(runtime_identity),
    )
    paths = compare_reskin_policies_from_bundle(output_root)
    assert validate_completed_f3d_bundle(output_root, _dataset_spec=spec)
    for relative, payload in before.items():
        path = output_root / relative
        assert path.is_file()
        assert path.read_bytes() == payload
    for cell in source.cells:
        stage = output_root / "stages" / "skinning" / cell.stages.skinning
        artifact = json.loads((stage / "skins.json").read_text(encoding="utf-8"))
        stage_manifest = json.loads((stage / "stage_manifest.json").read_text(encoding="utf-8"))
        assert artifact["format_version"] == (1 if contract == 3 else 2)
        assert (
            stage_manifest["resolved_settings"]["skin_artifact_semantic_contract_version"]
            == contract
        )
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["format_version"] == 2 for path in paths[3:]
    )


@pytest.mark.skipif(
    int(np.__version__.partition(".")[0]) >= 2,
    reason="canonical dense reskin evidence requires the supported numpy<2 runtime",
)
def test_controlled_promotion_gate_contract_is_deterministic() -> None:
    first = build_dense_reskin_promotion_gate()
    second = build_dense_reskin_promotion_gate()
    expected_case_ids = (
        "plane_3x3_center_hole",
        "plane_4x4_internal_2x2_hole",
        "low_support_gap",
        "dipping_surface_internal_hole",
        "parallel_surfaces",
        "corner_touch_orientation_boundary",
        "rounded_subvoxel_surface",
        "volume_boundary_surface",
        "valid_mask_barrier",
        "prior_occupancy_barrier",
    )
    case_ids = tuple(case.case_id for case in controlled_dense_reskin_cases())
    assert first == second
    assert case_ids == expected_case_ids
    canonical_payload = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected_payload_sha256 = "98554f6a3bbf073217ed0aba21998db1bb19ac4b1b7e03f4f913ea7c6d1b5a32"
    assert hashlib.sha256(canonical_payload).hexdigest() == expected_payload_sha256
    assert first["schema_version"] == 2
    assert first["passed"] is True
    assert first["reasons"] == []
    assert first["aggregate"]["deterministic_reexecution"] is True
    assert tuple(first["aggregate"]["case_ids"]) == expected_case_ids
    assert first["aggregate"]["case_count"] == len(case_ids) == 10
    assert first["aggregate"]["buffered_precision_delta"] >= -0.02
    assert first["aggregate"]["symmetric_chamfer_mean_delta"] <= 0.25
    assert first["aggregate"]["small_skin_cell_fraction_delta"] <= 0.05
    for result in first["case_results"]:
        candidate = result["candidate"]
        assert candidate["finite_failure_count"] == 0
        assert candidate["duplicate_rounded_cell_index_count"] == 0
        assert candidate["generated_on_invalid_mask_count"] == 0
        assert candidate["generated_on_prior_occupancy_count"] == 0
        assert candidate["out_of_volume_cell_count"] == 0
        assert candidate["clamped_artifact_count"] == 0


def test_cli_acceptance_paths_forward_comparison_resume_and_deep_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    data_root.mkdir()
    calls: list[tuple[str, bool, bool]] = []

    def run_experiment(**kwargs: Any) -> Path:
        calls.append(("run", bool(kwargs["resume"]), bool(kwargs["deep"])))
        output_root.mkdir(exist_ok=True)
        (output_root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "plan": {
                        "reference_workflow_settings": {
                            "skinning_config": {
                                "reskin_policy": "existing_cells_v1",
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return output_root

    def compare(_: Path, *, resume: bool = False, deep: bool = False) -> None:
        calls.append(("compare", resume, deep))

    monkeypatch.setattr(f3d_mode_comparison, "run_experiment", run_experiment)
    monkeypatch.setattr(
        f3d_mode_comparison,
        "compare_reskin_policies_from_bundle",
        compare,
    )
    pair = [
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_root),
        "--compare-reskin-policies",
        "existing_cells_v1,reference_dense_v1",
    ]
    assert f3d_mode_comparison.main(pair) == 0
    assert f3d_mode_comparison.main([*pair, "--resume", "--deep-validate"]) == 0
    assert calls == [
        ("run", False, False),
        ("compare", False, False),
        ("run", True, True),
        ("compare", True, True),
    ]

"""Backward-compatible entry point for the synthetic-quality report CLI."""

from pyosv.cli import synthetic_quality as _impl


SyntheticVotingConfig = _impl.SyntheticVotingConfig
SyntheticScannerConfig = _impl.SyntheticScannerConfig
SyntheticTruthMetricConfig = _impl.SyntheticTruthMetricConfig
SyntheticSkinningConfig = _impl.SyntheticSkinningConfig
build_parser = _impl.build_parser
get_variant_spec = _impl.get_variant_spec
run_example = _impl.run_example
main = _impl.main

_run_voter_thinning_diagnostic = _impl._run_voter_thinning_diagnostic
_scanner_downstream_diagnostics = _impl._scanner_downstream_diagnostics
_scanner_stage_loss_diagnostics = _impl._scanner_stage_loss_diagnostics
_candidate_count = _impl._candidate_count
_edge_candidate_fraction = _impl._edge_candidate_fraction
_edge_mask = _impl._edge_mask
_delta_or_none = _impl._delta_or_none
_boundary_seed_retention_v1_seeds = _impl._boundary_seed_retention_v1_seeds
_recenter_edge_fvt_to_target = _impl._recenter_edge_fvt_to_target
_apply_boundary_edge_thin_v1 = _impl._apply_boundary_edge_thin_v1


def _scan_backend_attributes(scanner, scanner_config, scanner_input, backend):
    return _impl.scan_backend_attributes(scanner, scanner_config, scanner_input, backend)


def _scan_ensemble_attributes(scanner, scanner_config, scanner_input):
    return _impl.scan_ensemble_attributes(
        scanner,
        scanner_config,
        scanner_input,
        backend_scan=_scan_backend_attributes,
    )


def _scanner_attributes_from_case(case, scanner_config):
    attributes = _impl.scanner_attributes_from_case(
        case,
        scanner_config,
        backend_scan=_scan_backend_attributes,
        ensemble_scan=_scan_ensemble_attributes,
    )
    return dict(attributes.report), dict(attributes.volumes)


def _run_voting_from_attributes(case, **kwargs):
    kwargs.setdefault("thinning_diagnostic_runner", _run_voter_thinning_diagnostic)
    evaluation = _impl._package_run_voting_from_attributes(case, **kwargs)
    return (
        dict(evaluation.report_payload),
        dict(evaluation.artifacts.volumes),
        dict(evaluation.artifacts.skins_payload),
    )


def _run_scanner_pipeline(case, **kwargs):
    kwargs.setdefault("thinning_diagnostic_runner", _run_voter_thinning_diagnostic)
    kwargs.setdefault("scanner_downstream_diagnostic_runner", _scanner_downstream_diagnostics)
    kwargs.setdefault("scanner_stage_loss_diagnostic_runner", _scanner_stage_loss_diagnostics)
    evaluation = _impl._package_run_scanner_pipeline(case, **kwargs)
    return (
        dict(evaluation.report_payload),
        dict(evaluation.artifacts.volumes),
        dict(evaluation.artifacts.skins_payload),
    )


if __name__ == "__main__":
    raise SystemExit(main())

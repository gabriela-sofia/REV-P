import os
from tests.test_revp_v1uv_curitiba_source_target_builder import install_candidate_discovery, set_env
import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def test_completion_report_writes_manifest_and_blocks_truth(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    common.run_source_target_builder(common.parse_args([]))
    install_candidate_discovery(data, official=True, date="2022-01-15", hazard="alagamento")
    common.run_geocuritiba_resolver(common.parse_args(["--dry-run"]))
    common.run_open_data_resolver(common.parse_args(["--dry-run"]))
    common.run_defesa_civil_pr_resolver(common.parse_args([]))
    common.run_hydromet_source_resolver(common.parse_args([]))
    common.run_candidate_event_builder(common.parse_args([]))
    common.run_event_evidence_audit(common.parse_args([]))
    common.run_event_registry_updater(common.parse_args([]))
    result = common.run_completion_report(common.parse_args([]))
    blockers = common.load_csv(os.path.join(data, "v1uv_curitiba_ground_reference_blocker_matrix.csv"))
    manifest = common.load_csv(os.path.join(data, "v1uv_versionable_artifacts_manifest.csv"))
    assert result["status"] == "CURITIBA_PUBLIC_EVENT_CANDIDATE_FOR_REVIEW"
    assert manifest
    assert all(r["ground_truth_operational"] == "false" for r in blockers)

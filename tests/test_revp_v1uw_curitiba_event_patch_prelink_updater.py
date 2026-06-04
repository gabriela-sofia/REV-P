from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_event_patch_prelink_is_candidate_only(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_event_source_snapshotter(common.parse_args(["--dry-run"]))
    common.run_document_text_extractor(common.parse_args([]))
    common.run_event_date_hazard_audit(common.parse_args([]))
    common.run_event_candidate_status_builder(common.parse_args([]))
    rows = common.run_event_patch_prelink_updater(common.parse_args([]))
    assert rows[0]["event_patch_candidate_only"] == "true"
    assert rows[0]["patch_bound_truth"] == "false"

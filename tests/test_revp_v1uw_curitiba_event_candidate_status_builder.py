from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_event_candidate_status_does_not_create_ground_reference(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_event_source_snapshotter(common.parse_args(["--dry-run"]))
    common.run_document_text_extractor(common.parse_args([]))
    common.run_event_date_hazard_audit(common.parse_args([]))
    common.run_hydromet_anchor_resolver(common.parse_args([]))
    common.run_geocuritiba_layer_deepener(common.parse_args([]))
    common.run_open_data_resource_deepener(common.parse_args([]))
    rows = common.run_event_candidate_status_builder(common.parse_args([]))
    assert rows[0]["can_advance_to_overlay_preflight"] == "false"
    assert rows[0]["can_create_ground_reference"] == "false"

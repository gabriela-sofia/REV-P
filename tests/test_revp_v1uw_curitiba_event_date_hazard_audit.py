from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_date_hazard_audit_separates_alert_and_observed_occurrence(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_event_source_snapshotter(common.parse_args(["--dry-run"]))
    common.run_document_text_extractor(common.parse_args([]))
    rows = common.run_event_date_hazard_audit(common.parse_args([]))
    assert rows[0]["date_gate"] == "CUR_2022_01_15_EXACT"
    assert rows[0]["can_create_ground_reference"] == "false"

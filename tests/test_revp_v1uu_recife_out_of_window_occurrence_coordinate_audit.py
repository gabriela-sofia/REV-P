from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_out_of_window_occurrence_coordinates_do_not_become_candidate(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    common.run_contextual_coordinate_asset_classifier()
    rows = common.run_out_of_window_occurrence_coordinate_audit()
    assert len(rows) == 1
    assert rows[0]["can_support_event_candidate"] == "false"
    assert rows[0]["out_of_window_class"] == "SERVICE_CALL_COORDINATE_OUTSIDE_EVENT_WINDOW"

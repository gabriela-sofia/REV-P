from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_event_patch_context_attachment_does_not_create_truth(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    common.run_contextual_coordinate_asset_classifier()
    rows = common.run_event_patch_context_attacher()
    assert len(rows) == 1
    assert rows[0]["contextual_coordinate_support"] == "AVAILABLE"
    assert rows[0]["ground_reference_status"] == "BLOCKED_CONTEXT_ONLY"
    assert rows[0]["can_create_ground_reference"] == "false"

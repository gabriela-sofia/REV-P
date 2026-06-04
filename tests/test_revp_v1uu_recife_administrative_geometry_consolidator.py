from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_administrative_geometry_never_becomes_occurrence(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    common.run_contextual_coordinate_asset_classifier()
    rows = common.run_administrative_geometry_consolidator()
    assert rows[0]["admin_layer_type"] == "bairro"
    assert rows[0]["can_support_occurrence_review"] == "false"
    assert rows[0]["can_create_ground_reference"] == "false"

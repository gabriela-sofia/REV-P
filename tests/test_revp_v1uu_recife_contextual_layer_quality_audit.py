from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_quality_audit_does_not_allow_overlay(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    common.run_contextual_coordinate_asset_classifier()
    rows = common.run_contextual_layer_quality_audit()
    assert rows
    assert all(r["can_support_overlay"] == "false" for r in rows)
    assert any(r["crs_status"] == "CRS_PRESENT" for r in rows)

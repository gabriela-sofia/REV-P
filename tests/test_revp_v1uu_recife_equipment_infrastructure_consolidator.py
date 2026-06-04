from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_equipment_infrastructure_context_not_occurrence(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    common.run_contextual_coordinate_asset_classifier()
    rows = common.run_equipment_infrastructure_consolidator()
    assert rows
    assert all(r["can_support_occurrence_review"] == "false" for r in rows)
    assert all(r["can_support_overlay"] == "false" for r in rows)

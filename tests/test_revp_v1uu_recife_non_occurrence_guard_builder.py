from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_non_occurrence_guard_records_safe_and_prohibited_use(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    common.run_contextual_coordinate_asset_classifier()
    rows = common.run_non_occurrence_guard_builder()
    assert rows
    assert all("ground_reference" in r["prohibited_use"] for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)

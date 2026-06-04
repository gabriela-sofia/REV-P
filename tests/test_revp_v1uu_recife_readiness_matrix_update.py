from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_readiness_update_improves_only_contextual_support(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    common.run_contextual_coordinate_asset_classifier()
    common.run_contextual_layer_quality_audit()
    common.run_event_patch_context_attacher()
    rows = common.run_readiness_matrix_update()
    dims = {r["dimension"]: r["classification"] for r in rows}
    assert dims["contextual_coordinate_support"] in {"STRONG", "MODERATE"}
    assert dims["occurrence_coordinate_support"] == "BLOCKED_OUT_OF_WINDOW"
    assert dims["ground_reference_blocker_status"] == "BLOCKED"
    assert all(r["ground_truth_operational"] == "false" for r in rows)

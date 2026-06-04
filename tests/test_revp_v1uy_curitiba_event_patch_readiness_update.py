from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import run_pipeline, set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_readiness_update_never_opens_overlay_or_training(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    run_pipeline(data, v1ux_raw)
    rows = common.load_csv(common.dataset_path("v1uy_curitiba_event_patch_readiness_update.csv"))
    assert any(r["dimension"] == "event_patch_linkage_status" for r in rows)
    assert all(r["no_overlay_executed"] == "true" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)

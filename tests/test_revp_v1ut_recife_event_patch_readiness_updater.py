from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, install_v1us_rec_candidate, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_readiness_updater_is_additive_and_keeps_training_blocked(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_contextual_equipment.csv", "context_equipment",
                  classification="INFRASTRUCTURE_CONTEXT")
    install_v1us_rec_candidate(data)
    common.run_coordinate_asset_locator()
    common.run_coordinate_schema_reparser()
    common.run_event_window_coordinate_filter()
    common.run_hazard_coordinate_crossfilter()
    common.run_coordinate_candidate_audit()
    rows = common.run_event_patch_readiness_updater()
    assert len(rows) == 4
    assert {r["dimension"] for r in rows} == {"coordinate_support", "overlay_readiness", "ground_reference_readiness", "training_readiness"}
    assert all(r["can_create_training_label"] == "false" for r in rows)

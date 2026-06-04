import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env, write_csv


def test_event_status_updater_does_not_release_overlay_without_geometry(tmp_path, monkeypatch):
    data, _, _, _ = set_env(tmp_path, monkeypatch)
    write_csv(data / "v1up_petropolis_observed_geometry_candidate_audit.csv", common.AUDIT_COLUMNS, [])
    write_csv(data / "v1up_petropolis_phenomenon_separation_registry.csv", common.PHENOMENON_COLUMNS, [])
    write_csv(data / "v1up_petropolis_artifact_inventory.csv", common.INVENTORY_COLUMNS, [])
    for name, cols in [
        ("v1up_petropolis_sgb_rigeo_registry.csv", common.RIGEO_COLUMNS),
        ("v1up_petropolis_rj_public_portal_registry.csv", common.PORTAL_COLUMNS),
        ("v1up_petropolis_copernicus_charter_registry.csv", common.COPERNICUS_COLUMNS),
        ("v1up_petropolis_cemaden_registry.csv", common.CEMADEN_COLUMNS),
    ]:
        write_csv(data / name, cols, [])
    rows = common.run_event_status_updater()
    assert all(r["can_advance_to_overlay_preflight"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)

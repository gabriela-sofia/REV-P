import os

import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env, write_csv


def test_completion_report_writes_docs_next_actions_and_manifest(tmp_path, monkeypatch):
    data, docs, configs, _ = set_env(tmp_path, monkeypatch)
    monkeypatch.setattr(common, "V1UP_ARTIFACTS", [])
    write_csv(data / "v1up_petropolis_event_status_registry.csv", common.EVENT_STATUS_COLUMNS, [{
        "event_id": "PET_2022_02_15", "v1up_status": "DOCUMENT_ONLY_NO_GEOMETRY", "has_public_artifact": "true", "has_downloaded_artifact": "false", "has_geodata": "false", "has_observed_geometry_candidate": "false", "phenomenon_separation_status": "NOT_PROVEN", "ground_truth_operational": "false", "can_create_ground_reference": "false", "can_create_training_label": "false", "can_advance_to_overlay_preflight": "false", "main_blocker": "BLOCKED_GEOMETRY_MISSING", "recommended_next_action": "", "notes": "",
    }])
    for name, cols in [
        ("v1up_petropolis_download_manifest.csv", common.DOWNLOAD_COLUMNS),
        ("v1up_petropolis_artifact_inventory.csv", common.INVENTORY_COLUMNS),
        ("v1up_petropolis_observed_geometry_candidate_audit.csv", common.AUDIT_COLUMNS),
        ("v1up_petropolis_phenomenon_separation_registry.csv", common.PHENOMENON_COLUMNS),
        ("v1up_petropolis_sgb_rigeo_registry.csv", common.RIGEO_COLUMNS),
        ("v1up_petropolis_geosgb_service_registry.csv", common.GEOSGB_COLUMNS),
        ("v1up_petropolis_rj_public_portal_registry.csv", common.PORTAL_COLUMNS),
        ("v1up_petropolis_cemaden_registry.csv", common.CEMADEN_COLUMNS),
        ("v1up_petropolis_copernicus_charter_registry.csv", common.COPERNICUS_COLUMNS),
    ]:
        write_csv(data / name, cols, [])
    result = common.run_completion_report()
    assert result["next_action"] == "v1uq - Petropolis Phenomenon Separation Deep Audit"
    assert os.path.exists(docs / "protocolo_c_status_atual_v1up.md")
    assert os.path.exists(data / "v1up_next_actions_registry.csv")
    assert os.path.exists(configs / "v1up_petropolis_download_policy.yaml")

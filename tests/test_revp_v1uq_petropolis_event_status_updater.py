import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, write_csv


def test_event_status_updater_does_not_release_overlay(tmp_path, monkeypatch):
    data, _, _, _, _ = set_env(tmp_path, monkeypatch)
    write_csv(data / "v1uq_petropolis_phenomenon_separation_decision_matrix.csv", common.DECISION_COLUMNS, [{
        "decision_id": "d", "event_id": "PET_2022_02_15", "flood_signal": "true", "landslide_signal": "true", "mixed_signal": "false", "separation_status": "PHENOMENON_SEPARATION_STRONG_TEXTUAL", "locality_linkage_status": "LOCALITY_TEXT_LINK_PRESENT", "temporal_linkage_status": "TEMPORAL_TEXT_LINK_PRESENT", "geodata_status": "MISSING_GEODATA_SIGNAL_PRESENT", "evidence_strength": "STRONG", "can_advance_to_geometry_search": "true", "can_advance_to_overlay_preflight": "false", "can_create_ground_reference": "false", "can_create_training_label": "false", "main_blocker": "GEOMETRY_STILL_MISSING", "required_next_action": "v1ur - Petropolis Public Geodata Path Recovery", "notes": "",
    }])
    rows = common.run_event_status_updater()
    assert rows[0]["can_advance_to_overlay_preflight"] == "false"
    assert rows[0]["can_create_ground_reference"] == "false"

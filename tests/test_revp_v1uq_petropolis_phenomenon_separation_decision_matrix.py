import scripts.protocolo_c.revp_v1uq_petropolis_common as common
from tests.test_revp_v1uq_petropolis_pdf_text_extractor import set_env, write_csv


def test_decision_matrix_never_creates_ground_reference(tmp_path, monkeypatch):
    data, _, _, _, _ = set_env(tmp_path, monkeypatch)
    write_csv(data / "v1uq_petropolis_page_level_evidence_registry.csv", common.PAGE_COLUMNS, [
        {"page_evidence_id": "p1", "event_id": "PET_2022_02_15", "asset_id": "a", "page_number": "1", "dominant_phenomenon_class": "URBAN_FLOODING", "flood_signal_strength": "STRONG", "landslide_signal_strength": "NONE", "mixed_signal_strength": "NONE", "locality_signal_strength": "STRONG", "date_signal_strength": "STRONG", "geodata_signal_strength": "NONE", "is_map_page": "false", "is_context_only": "false", "evidence_role": "PHENOMENON_REVIEW", "can_support_phenomenon_gate": "true", "can_create_ground_reference": "false", "notes": ""},
        {"page_evidence_id": "p2", "event_id": "PET_2022_02_15", "asset_id": "a", "page_number": "2", "dominant_phenomenon_class": "LANDSLIDE_OR_MASS_MOVEMENT", "flood_signal_strength": "NONE", "landslide_signal_strength": "STRONG", "mixed_signal_strength": "NONE", "locality_signal_strength": "STRONG", "date_signal_strength": "STRONG", "geodata_signal_strength": "NONE", "is_map_page": "false", "is_context_only": "false", "evidence_role": "PHENOMENON_REVIEW", "can_support_phenomenon_gate": "true", "can_create_ground_reference": "false", "notes": ""},
    ])
    write_csv(data / "v1uq_petropolis_locality_term_audit.csv", common.LOCALITY_COLUMNS, [{"locality_audit_id": "l", "event_id": "PET_2022_02_15", "asset_id": "a", "page_number": "1", "locality_token_hash": "h", "locality_class": "PETROPOLIS_LOCALITY_TEXT_SIGNAL", "locality_signal_strength": "STRONG", "linked_phenomenon_class": "URBAN_FLOODING", "can_support_contextual_review": "true", "can_support_overlay": "false", "notes": ""}])
    write_csv(data / "v1uq_petropolis_event_date_linkage_audit.csv", common.DATE_COLUMNS, [{"date_linkage_id": "d", "event_id": "PET_2022_02_15", "asset_id": "a", "page_number": "1", "date_signal_class": "EXACT_EVENT_DATE", "date_signal_strength": "STRONG", "event_specificity": "EVENT_SPECIFIC", "temporal_link_status": "EXACT_EVENT_DATE", "can_support_temporal_gate": "true", "notes": ""}])
    write_csv(data / "v1uq_petropolis_missing_geodata_signal_audit.csv", common.MISSING_GEODATA_COLUMNS, [])
    rows = common.run_phenomenon_separation_decision_matrix()
    assert rows[0]["can_create_ground_reference"] == "false"
    assert rows[0]["can_advance_to_overlay_preflight"] == "false"

import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env, write_csv


def test_observed_geometry_candidate_audit_never_creates_ground_reference(tmp_path, monkeypatch):
    data, _, _, _ = set_env(tmp_path, monkeypatch)
    write_csv(data / "v1up_petropolis_artifact_inventory.csv", common.INVENTORY_COLUMNS, [{
        "asset_id": "a1", "event_id": "PET_2022_02_15", "source_id": "SGB_RIGEO", "record_id": "", "safe_filename": "deslizamento.geojson", "format_hint": "geojson", "sha256": "", "size_bytes": "", "artifact_class": "geojson", "contained_files": "", "has_pdf_text": "", "has_internal_links": "", "has_geodata": "true", "geometry_type": "Polygon", "crs": "EPSG:4326", "feature_count": "1", "bounds": "", "fields": "", "has_date_field": "true", "has_phenomenon_field": "true", "has_locality_field": "false", "has_coordinate_fields": "false", "inventory_status": "", "notes": "",
    }])
    write_csv(data / "v1up_petropolis_phenomenon_separation_registry.csv", common.PHENOMENON_COLUMNS, [{
        "phenomenon_id": "p1", "event_id": "PET_2022_02_15", "asset_id": "a1", "source_id": "SGB_RIGEO", "phenomenon_class": "LANDSLIDE_OR_MASS_MOVEMENT", "flood_signal": "false", "landslide_signal": "true", "mixed_signal": "false", "separation_status": "PHENOMENON_CLASS_SEPARATED", "evidence_strength": "STRONG", "can_support_phenomenon_gate": "true", "can_create_ground_reference": "false", "can_create_training_label": "false", "notes": "",
    }])
    rows = common.run_observed_geometry_candidate_audit()
    assert rows[0]["audit_status"] == "PETROPOLIS_OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW"
    assert rows[0]["can_create_ground_reference"] == "false"
    assert rows[0]["can_create_training_label"] == "false"
    assert rows[0]["ground_truth_operational"] == "false"

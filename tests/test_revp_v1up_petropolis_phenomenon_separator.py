import scripts.protocolo_c.revp_v1up_petropolis_common as common
from tests.test_revp_v1up_petropolis_source_target_builder import set_env, write_csv


def test_phenomenon_separator_distinguishes_flood_landslide_mixed_and_context(tmp_path, monkeypatch):
    data, _, _, _ = set_env(tmp_path, monkeypatch)
    write_csv(data / "v1up_petropolis_artifact_inventory.csv", common.INVENTORY_COLUMNS, [
        {"asset_id": "a1", "event_id": "PET_2022_02_15", "source_id": "SGB_RIGEO", "record_id": "", "safe_filename": "deslizamento.geojson", "format_hint": "geojson", "sha256": "", "size_bytes": "", "artifact_class": "geojson", "contained_files": "", "has_pdf_text": "", "has_internal_links": "", "has_geodata": "true", "geometry_type": "Polygon", "crs": "EPSG:4326", "feature_count": "1", "bounds": "", "fields": "fenomeno", "has_date_field": "true", "has_phenomenon_field": "true", "has_locality_field": "false", "has_coordinate_fields": "false", "inventory_status": "", "notes": "cicatriz deslizamento"},
        {"asset_id": "a2", "event_id": "PET_2022_02_15", "source_id": "SGB_RIGEO", "record_id": "", "safe_filename": "alagamento.geojson", "format_hint": "geojson", "sha256": "", "size_bytes": "", "artifact_class": "geojson", "contained_files": "", "has_pdf_text": "", "has_internal_links": "", "has_geodata": "true", "geometry_type": "Point", "crs": "EPSG:4326", "feature_count": "1", "bounds": "", "fields": "fenomeno", "has_date_field": "true", "has_phenomenon_field": "true", "has_locality_field": "false", "has_coordinate_fields": "false", "inventory_status": "", "notes": "alagamento"},
        {"asset_id": "a3", "event_id": "PET_2022_02_15", "source_id": "SGB_RIGEO", "record_id": "", "safe_filename": "risco.pdf", "format_hint": "pdf", "sha256": "", "size_bytes": "", "artifact_class": "technical_pdf", "contained_files": "", "has_pdf_text": "", "has_internal_links": "", "has_geodata": "false", "geometry_type": "", "crs": "", "feature_count": "0", "bounds": "", "fields": "", "has_date_field": "false", "has_phenomenon_field": "false", "has_locality_field": "false", "has_coordinate_fields": "false", "inventory_status": "", "notes": "risco suscetibilidade"},
    ])
    for name in ["v1up_petropolis_sgb_rigeo_registry.csv", "v1up_petropolis_rj_public_portal_registry.csv", "v1up_petropolis_copernicus_charter_registry.csv", "v1up_petropolis_cemaden_registry.csv"]:
        write_csv(data / name, ["event_id"], [])
    rows = common.run_phenomenon_separator()
    classes = {r["phenomenon_class"] for r in rows}
    assert "LANDSLIDE_OR_MASS_MOVEMENT" in classes
    assert "URBAN_FLOODING" in classes
    assert "RISK_OR_SUSCEPTIBILITY_CONTEXT" in classes

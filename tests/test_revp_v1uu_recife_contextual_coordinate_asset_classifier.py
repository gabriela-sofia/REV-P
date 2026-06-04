import csv
import os

import scripts.protocolo_c.revp_v1uu_recife_common as common


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    staging = tmp_path / "local_only" / "staging"
    reports = tmp_path / "local_only" / "reports"
    for p in (data, docs, cfg, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data


def install_minimal_inputs(data):
    write_csv(os.path.join(data, "v1ut_recife_coordinate_asset_locator.csv"),
              common.CLASS_COLUMNS[:0] + ["coordinate_asset_id", "event_id", "asset_id", "artifact_id", "source_id", "asset_type", "row_count", "rows_with_coordinates_reported", "has_geometry", "previous_classification"],
              [
                  {"coordinate_asset_id": "CA1", "event_id": common.EVENT_ID, "asset_id": "admin_asset", "artifact_id": "FINV1", "source_id": "ckan", "asset_type": "geospatial_vector", "row_count": "10", "rows_with_coordinates_reported": "10", "has_geometry": "true", "previous_classification": "INFRASTRUCTURE_CONTEXT"},
                  {"coordinate_asset_id": "CA2", "event_id": common.EVENT_ID, "asset_id": "equip_asset", "artifact_id": "FINV2", "source_id": "ckan", "asset_type": "tabular", "row_count": "5", "rows_with_coordinates_reported": "5", "has_geometry": "false", "previous_classification": "INFRASTRUCTURE_CONTEXT"},
                  {"coordinate_asset_id": "CA3", "event_id": common.EVENT_ID, "asset_id": "occ_asset", "artifact_id": "FINV3", "source_id": "ckan", "asset_type": "tabular", "row_count": "2", "rows_with_coordinates_reported": "2", "has_geometry": "false", "previous_classification": "OCCURRENCE_COORDINATES_CANDIDATE"},
              ])
    write_csv(os.path.join(data, "v1ut_recife_coordinate_schema_reparse.csv"),
              ["coordinate_asset_id", "event_id", "asset_id", "artifact_id", "asset_type", "rows_with_parseable_coordinates", "rows_in_recife_plausible_range", "geometry_type", "coordinate_semantics"],
              [
                  {"coordinate_asset_id": "CA1", "event_id": common.EVENT_ID, "asset_id": "admin_asset", "artifact_id": "FINV1", "asset_type": "geospatial_vector", "rows_with_parseable_coordinates": "10", "rows_in_recife_plausible_range": "10", "geometry_type": "Polygon", "coordinate_semantics": "ADMIN_REGION_GEOMETRY"},
                  {"coordinate_asset_id": "CA2", "event_id": common.EVENT_ID, "asset_id": "equip_asset", "artifact_id": "FINV2", "asset_type": "tabular", "rows_with_parseable_coordinates": "5", "rows_in_recife_plausible_range": "5", "geometry_type": "", "coordinate_semantics": "CONTEXTUAL_EQUIPMENT_OR_INFRASTRUCTURE_COORDINATE"},
                  {"coordinate_asset_id": "CA3", "event_id": common.EVENT_ID, "asset_id": "occ_asset", "artifact_id": "FINV3", "asset_type": "tabular", "rows_with_parseable_coordinates": "2", "rows_in_recife_plausible_range": "2", "geometry_type": "", "coordinate_semantics": "OCCURRENCE_OR_SERVICE_CALL_COORDINATE"},
              ])
    write_csv(os.path.join(data, "v1ut_recife_hazard_coordinate_crossfilter.csv"),
              ["event_id", "asset_id", "row_hash", "event_window_match", "coordinate_status", "has_hazard_term", "hazard_coordinate_status", "can_promote_to_coordinate_candidate"],
              [{"event_id": common.EVENT_ID, "asset_id": "occ_asset", "row_hash": "h1", "event_window_match": "OUTSIDE_REC_2022_CORE_WINDOW", "coordinate_status": "PUBLIC_COORDINATE_IN_RECIFE_RANGE", "has_hazard_term": "true", "hazard_coordinate_status": "COORDINATE_NOT_IN_EVENT_WINDOW", "can_promote_to_coordinate_candidate": "false"}])
    write_csv(os.path.join(data, "v1uj_focused_artifact_inventory.csv"),
              ["inventory_id", "internal_path", "geometry_type", "crs", "feature_count", "columns_detected", "classification"],
              [
                  {"inventory_id": "FINV1", "internal_path": "bairros.geojson", "geometry_type": "Polygon", "crs": "EPSG:4326", "feature_count": "10", "columns_detected": "bairro", "classification": "CONTEXTUAL_OFFICIAL_LAYER"},
                  {"inventory_id": "FINV2", "internal_path": "equipamentos.csv", "geometry_type": "", "crs": "", "feature_count": "", "columns_detected": "equipamento latitude longitude", "classification": "CONTEXTUAL_OFFICIAL_LAYER"},
                  {"inventory_id": "FINV3", "internal_path": "service_calls.csv", "geometry_type": "", "crs": "", "feature_count": "", "columns_detected": "Data Ocorrencia latitude longitude", "classification": "OCCURRENCE_COORDINATES_CANDIDATE"},
              ])
    write_csv(os.path.join(data, "v1us_event_patch_candidate_registry.csv"),
              ["event_patch_candidate_id", "event_id", "region", "patch_id"],
              [{"event_patch_candidate_id": "EPC1", "event_id": common.EVENT_ID, "region": "REC", "patch_id": "REC_01"},
               {"event_patch_candidate_id": "EPC2", "event_id": "CUR_2022", "region": "CUR", "patch_id": "CUR_01"}])
    write_csv(os.path.join(data, "v1us_event_patch_readiness_matrix.csv"),
              ["event_patch_candidate_id", "event_id", "dimension", "classification"], [])


def test_context_classifier_separates_admin_equipment_and_out_window(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    rows = common.run_contextual_coordinate_asset_classifier()
    classes = {r["asset_id"]: r["context_layer_class"] for r in rows}
    assert classes["admin_asset"] == "ADMINISTRATIVE_OR_REGION_GEOMETRY"
    assert classes["equip_asset"] == "EQUIPMENT_OR_FACILITY_POINTS"
    assert classes["occ_asset"] == "OUT_OF_WINDOW_SERVICE_CALL_COORDINATES"
    assert all(r["can_support_overlay"] == "false" for r in rows)

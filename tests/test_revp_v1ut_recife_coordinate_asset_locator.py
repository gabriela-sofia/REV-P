import csv
import os
import shutil

import scripts.protocolo_c.revp_v1ut_recife_common as common

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "v1ut")


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
    raw = tmp_path / "local_only" / "raw"
    staging = tmp_path / "local_only" / "staging"
    reports = tmp_path / "local_only" / "reports"
    for p in (data, docs, cfg, raw, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "RAW_DIR", str(raw))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data, raw


def install_asset(data, raw, fixture_name, asset_id="asset_coord", asset_type="tabular",
                  classification="OCCURRENCE_COORDINATES_CANDIDATE",
                  coord_fields="latitude|longitude", rows_with_coordinates="1",
                  row_count="1"):
    safe = f"REC_2022_05_24_30__ckan__{asset_id}__abcdef123456__{fixture_name}"
    shutil.copy(os.path.join(FIX, fixture_name), raw / safe)
    inv_cols = ["inventory_id", "event_id", "internal_path", "asset_type", "extension", "classification", "columns_detected"]
    schema_cols = ["asset_schema_id", "event_id", "artifact_id", "asset_id", "source_id", "title", "asset_type", "row_count", "has_coordinate_fields", "coordinate_field_candidates"]
    audit_cols = ["coordinate_audit_id", "event_id", "asset_id", "asset_type", "coordinate_fields", "rows_checked", "rows_with_coordinates", "rows_in_recife_range", "geometry_status", "coordinate_classification"]
    inv = common.load_csv(os.path.join(data, "v1uj_focused_artifact_inventory.csv"))
    schema = common.load_csv(os.path.join(data, "v1uk_recife_asset_schema_registry.csv"))
    audit = common.load_csv(os.path.join(data, "v1uk_recife_coordinate_evidence_audit.csv"))
    extension = ".geojson" if fixture_name.endswith(".geojson") else ".csv"
    inv.append({"inventory_id": f"FINV_{asset_id}", "event_id": common.EVENT_ID, "internal_path": safe,
                "asset_type": asset_type, "extension": extension, "classification": classification,
                "columns_detected": coord_fields})
    schema.append({"asset_schema_id": f"SCHEMA_{asset_id}", "event_id": common.EVENT_ID,
                   "artifact_id": f"FINV_{asset_id}", "asset_id": asset_id, "source_id": "ckan",
                   "title": fixture_name, "asset_type": asset_type, "row_count": row_count,
                   "has_coordinate_fields": "true" if coord_fields else "false",
                   "coordinate_field_candidates": coord_fields})
    audit.append({"coordinate_audit_id": f"COORD_{asset_id}", "event_id": common.EVENT_ID,
                  "asset_id": asset_id, "asset_type": asset_type,
                  "coordinate_fields": coord_fields, "rows_checked": row_count,
                  "rows_with_coordinates": rows_with_coordinates,
                  "rows_in_recife_range": rows_with_coordinates,
                  "geometry_status": "GEOMETRY_PRESENT" if asset_type == "geospatial_vector" else "",
                  "coordinate_classification": classification})
    write_csv(os.path.join(data, "v1uj_focused_artifact_inventory.csv"), inv_cols, inv)
    write_csv(os.path.join(data, "v1uk_recife_asset_schema_registry.csv"), schema_cols, schema)
    write_csv(os.path.join(data, "v1uk_recife_coordinate_evidence_audit.csv"), audit_cols, audit)
    return asset_id


def install_v1us_rec_candidate(data):
    write_csv(os.path.join(data, "v1us_event_patch_candidate_registry.csv"),
              ["event_patch_candidate_id", "event_id", "region", "patch_id"],
              [{"event_patch_candidate_id": "EPC1", "event_id": common.EVENT_ID, "region": "REC", "patch_id": "REC_01"}])
    write_csv(os.path.join(data, "v1us_event_patch_readiness_matrix.csv"),
              ["event_patch_candidate_id", "event_id", "dimension", "classification"],
              [{"event_patch_candidate_id": "EPC1", "event_id": common.EVENT_ID, "dimension": "coordinate_support", "classification": "BLOCKED"}])


def test_asset_locator_marks_public_coordinate_assets_for_reparse(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_coordinate_hazard_window.csv")
    rows = common.run_coordinate_asset_locator()
    assert rows[0]["should_reparse"] == "true"
    assert rows[0]["has_coordinate_fields"] == "true"
    assert rows[0]["suspected_blocker"] == "REPARSE_REQUIRED_BEFORE_PROMOTION"

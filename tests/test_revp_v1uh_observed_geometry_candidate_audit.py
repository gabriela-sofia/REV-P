"""Tests for v1uh — Observed Geometry Candidate Audit."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c",
                       "revp_v1uh_observed_geometry_candidate_audit.py")

CANDIDATE_COLUMNS = [
    "candidate_id", "event_id", "response_id", "asset_id", "institution",
    "candidate_class", "has_geometry", "geometry_type", "crs",
    "can_be_ground_reference_candidate",
    "can_create_ground_reference", "can_create_training_label",
]


def _make_assets(tmp_path, assets):
    path = os.path.join(tmp_path, "assets.csv")
    cols = ["asset_id", "response_id", "event_id", "institution",
            "container_type", "internal_path", "asset_type", "extension",
            "file_size_bytes", "sha256", "has_geometry", "geometry_type",
            "crs", "feature_count", "has_prj", "has_attribute_table",
            "columns_detected", "pdf_pages", "text_extract_status",
            "inventory_status", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for a in assets:
            row = {c: "" for c in cols}
            row.update(a)
            writer.writerow(row)
    return path


def _run(tmp_path, assets):
    assets_path = _make_assets(tmp_path, assets)
    out = os.path.join(tmp_path, "candidates.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--assets", assets_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestObservedGeometryCandidateAudit:
    def test_empty_assets(self, tmp_path):
        rows = _run(str(tmp_path), [])
        assert len(rows) == 0

    def test_geospatial_becomes_candidate(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "asset_id": "A1", "asset_type": "geospatial_vector",
            "has_geometry": "true", "extension": ".geojson",
            "internal_path": "occurrences.geojson",
        }])
        assert rows[0]["candidate_class"] == "OBSERVED_EVENT_GEOMETRY_CANDIDATE"
        assert rows[0]["can_be_ground_reference_candidate"] == "true"
        assert rows[0]["can_create_ground_reference"] == "false"
        assert rows[0]["can_create_training_label"] == "false"

    def test_susceptibility_not_observed(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "asset_id": "A2", "asset_type": "tabular",
            "has_geometry": "false", "extension": ".csv",
            "internal_path": "suscetibilidade_cprm.csv",
            "columns_detected": "id|classe_suscetibilidade|area",
        }])
        assert rows[0]["candidate_class"] == "MODELED_SUSCEPTIBILITY_CONTEXT"
        assert rows[0]["can_be_ground_reference_candidate"] == "false"
        assert rows[0]["is_modeled_product"] == "true"

    def test_pdf_is_document_only(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "asset_id": "A3", "asset_type": "document",
            "has_geometry": "false", "extension": ".pdf",
        }])
        assert rows[0]["candidate_class"] == "DOCUMENT_ONLY"
        assert rows[0]["can_be_ground_reference_candidate"] == "false"

    def test_static_map_not_geometry(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "asset_id": "A4", "asset_type": "static_map",
            "has_geometry": "false", "extension": ".png",
        }])
        assert rows[0]["candidate_class"] == "STATIC_MAP_ONLY"
        assert rows[0]["is_static_map"] == "true"

    def test_csv_with_coords_is_table_candidate(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "asset_id": "A5", "asset_type": "tabular",
            "has_geometry": "false", "extension": ".csv",
            "columns_detected": "data|tipo|bairro|latitude|longitude",
        }])
        assert rows[0]["candidate_class"] == "TABLE_WITH_COORDINATES_CANDIDATE"
        assert rows[0]["can_be_ground_reference_candidate"] == "true"
        assert rows[0]["can_create_ground_reference"] == "false"

    def test_no_ground_reference_ever(self, tmp_path):
        rows = _run(str(tmp_path), [
            {"asset_id": "A6", "asset_type": "geospatial_vector",
             "has_geometry": "true", "extension": ".shp"},
            {"asset_id": "A7", "asset_type": "tabular",
             "has_geometry": "false", "extension": ".csv",
             "columns_detected": "lat|lon|data"},
            {"asset_id": "A8", "asset_type": "document",
             "has_geometry": "false", "extension": ".pdf"},
        ])
        for r in rows:
            assert r["can_create_ground_reference"] == "false"
            assert r["can_create_training_label"] == "false"

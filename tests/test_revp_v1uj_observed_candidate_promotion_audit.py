"""Tests for v1uj — Observed Candidate Promotion Audit."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_observed_candidate_promotion_audit.py")

INV_COLS = ["inventory_id", "source_tag", "event_id", "container_type",
            "internal_path", "asset_type", "extension", "has_geometry",
            "crs", "columns_detected", "date_term_detected", "hazard_term_detected",
            "locality_term_detected", "susceptibility_term_detected",
            "classification", "inventory_status"]

GEO_COLS = ["geosgb_record_id", "event_id", "service_url", "geometry_type",
            "spatial_reference", "fields", "is_observed_occurrence_candidate",
            "is_contextual_layer"]


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _run(tmp_path, inv_rows, geo_rows):
    inv = os.path.join(tmp_path, "inv.csv")
    geo = os.path.join(tmp_path, "geo.csv")
    out = os.path.join(tmp_path, "audit.csv")
    _write(inv, INV_COLS, inv_rows)
    _write(geo, GEO_COLS, geo_rows)
    cmd = [sys.executable, SCRIPT, "--inventory", inv, "--geosgb", geo, "--out", out]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


GOOD_INV = {
    "inventory_id": "FINV_0001", "source_tag": "ckan", "event_id": "PET_2022_02_15",
    "container_type": "standalone", "internal_path": "ocorrencias_2022.geojson",
    "asset_type": "geospatial_vector", "extension": ".geojson", "has_geometry": "true",
    "crs": "urn:ogc:def:crs:EPSG::4326", "columns_detected": "",
    "date_term_detected": "true", "hazard_term_detected": "true",
    "locality_term_detected": "true", "susceptibility_term_detected": "false",
    "classification": "observed_geometry", "inventory_status": "INVENTORIED",
}


class TestObservedCandidatePromotionAudit:
    def test_empty(self, tmp_path):
        rows = _run(str(tmp_path), [], [])
        assert rows == []

    def test_good_candidate_for_review(self, tmp_path):
        rows = _run(str(tmp_path), [GOOD_INV], [])
        assert len(rows) == 1
        assert rows[0]["max_status"] == "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW"

    def test_guardrail_gates_always_fail(self, tmp_path):
        rows = _run(str(tmp_path), [GOOD_INV], [])
        r = rows[0]
        assert r["G11_supervisor_review_required"] == "FAIL"
        assert r["G12_overlay_not_executed"] == "FAIL"
        assert r["G13_label_forbidden"] == "FAIL"
        assert r["can_create_ground_reference"] == "false"
        assert r["can_create_training_label"] == "false"

    def test_never_ground_reference_status(self, tmp_path):
        rows = _run(str(tmp_path), [GOOD_INV], [])
        for r in rows:
            assert r["max_status"] not in ("GROUND_REFERENCE", "GROUND_TRUTH", "LABEL")

    def test_susceptibility_not_observed(self, tmp_path):
        geo = {
            "geosgb_record_id": "GSGB_0001", "event_id": "PET_2022_02_15",
            "service_url": "https://geosgb.sgb.gov.br/x/MapServer",
            "geometry_type": "esriGeometryPolygon", "spatial_reference": "4326",
            "fields": "grau_susc", "is_observed_occurrence_candidate": "false",
            "is_contextual_layer": "true",
        }
        rows = _run(str(tmp_path), [], [geo])
        assert rows[0]["max_status"] != "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW"
        assert rows[0]["G09_phenomenon_not_only_susceptibility"] == "FAIL"

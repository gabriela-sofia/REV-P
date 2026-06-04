"""Tests for v1ui — Public Evidence Gate Delta."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_public_evidence_gate_delta.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")


def _run(tmp_path, extractions=None, candidates=None):
    ext_path = os.path.join(tmp_path, "ext.csv")
    ext_cols = ["extraction_id", "event_id", "has_geometry", "has_coordinate_fields",
                "crs", "has_date_field", "has_hazard_field", "has_locality_field",
                "is_event_specific", "can_be_observed_geometry_candidate"]
    with open(ext_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ext_cols)
        writer.writeheader()
        if extractions:
            writer.writerows(extractions)

    cand_path = os.path.join(tmp_path, "cand.csv")
    cand_cols = ["candidate_audit_id", "event_id", "max_status"]
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cand_cols)
        writer.writeheader()
        if candidates:
            writer.writerows(candidates)

    out = os.path.join(tmp_path, "delta.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--events", EVENTS,
         "--extractions", ext_path, "--candidates", cand_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPublicEvidenceGateDelta:
    def test_empty_produces_no_change(self, tmp_path):
        rows = _run(str(tmp_path))
        assert len(rows) >= 3 * 10
        for r in rows:
            assert r["delta_type"] == "NO_CHANGE"

    def test_ground_reference_always_no(self, tmp_path):
        rows = _run(str(tmp_path))
        gr_rows = [r for r in rows if r["dimension"] == "ground_reference_possible"]
        for r in gr_rows:
            assert r["v1ui_status"] == "NO"

    def test_label_always_no(self, tmp_path):
        rows = _run(str(tmp_path))
        label_rows = [r for r in rows if r["dimension"] == "label_possible"]
        for r in label_rows:
            assert r["v1ui_status"] == "NO"

    def test_gain_when_artifact_found(self, tmp_path):
        rows = _run(str(tmp_path), extractions=[{
            "extraction_id": "E1", "event_id": "REC_2022_05_24_30",
            "has_geometry": "true", "has_coordinate_fields": "true",
            "crs": "EPSG:4326", "has_date_field": "true",
            "has_hazard_field": "true", "has_locality_field": "true",
            "is_event_specific": "true",
            "can_be_observed_geometry_candidate": "true",
        }])
        rec_rows = [r for r in rows if r["event_id"] == "REC_2022_05_24_30"]
        gains = [r for r in rec_rows if r["delta_type"] == "GAIN"]
        assert len(gains) >= 1

    def test_delta_ids_unique(self, tmp_path):
        rows = _run(str(tmp_path))
        ids = [r["delta_id"] for r in rows]
        assert len(ids) == len(set(ids))

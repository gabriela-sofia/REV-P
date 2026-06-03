"""Tests for v1uh — Phenomenon and Temporal Gate Audit."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c",
                       "revp_v1uh_phenomenon_temporal_gate_audit.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")


def _make_inputs(tmp_path, candidates, mappings=None):
    cand_path = os.path.join(tmp_path, "candidates.csv")
    cand_cols = ["candidate_id", "event_id", "asset_id",
                 "has_event_date_field", "has_hazard_field",
                 "can_be_ground_reference_candidate"]
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cand_cols)
        writer.writeheader()
        writer.writerows(candidates)

    map_path = os.path.join(tmp_path, "mappings.csv")
    map_cols = ["mapping_id", "candidate_id", "canonical_field", "mapping_status"]
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=map_cols)
        writer.writeheader()
        if mappings:
            writer.writerows(mappings)

    return cand_path, map_path


def _run(tmp_path, candidates, mappings=None):
    cand_path, map_path = _make_inputs(tmp_path, candidates, mappings)
    out = os.path.join(tmp_path, "gates.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--candidates", cand_path, "--events", EVENTS,
         "--mappings", map_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPhenomenonTemporalGateAudit:
    def test_empty_input(self, tmp_path):
        rows = _run(str(tmp_path), [])
        assert len(rows) == 0

    def test_no_date_no_hazard_blocked(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "candidate_id": "C1", "event_id": "PET_2022_02_15",
            "asset_id": "A1", "has_event_date_field": "false",
            "has_hazard_field": "false",
            "can_be_ground_reference_candidate": "true",
        }])
        assert rows[0]["temporal_gate_status"] == "BLOCKED"
        assert rows[0]["phenomenon_gate_status"] == "BLOCKED"
        assert rows[0]["can_create_ground_reference"] == "false"

    def test_pet_mixed_phenomenon_blocked(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "candidate_id": "C1", "event_id": "PET_2022_02_15",
            "asset_id": "A1", "has_event_date_field": "true",
            "has_hazard_field": "true",
            "can_be_ground_reference_candidate": "true",
        }])
        assert rows[0]["phenomenon_gate_status"] == "BLOCKED"
        assert "separation" in rows[0]["phenomenon_status"].lower()

    def test_rec_single_hazard_can_pass(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "candidate_id": "C1", "event_id": "REC_2022_05_24_30",
            "asset_id": "A1", "has_event_date_field": "true",
            "has_hazard_field": "true",
            "can_be_ground_reference_candidate": "true",
        }])
        assert rows[0]["phenomenon_gate_status"] == "PASS"

    def test_mapped_date_counts(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "REC_2022_05_24_30",
                      "asset_id": "A1", "has_event_date_field": "false",
                      "has_hazard_field": "true",
                      "can_be_ground_reference_candidate": "true"}],
                    [{"mapping_id": "M1", "candidate_id": "C1",
                      "canonical_field": "event_date",
                      "mapping_status": "MAPPED"}])
        assert rows[0]["temporal_gate_status"] == "PASS"

    def test_can_create_ground_reference_always_false(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "candidate_id": "C1", "event_id": "REC_2022_05_24_30",
            "asset_id": "A1", "has_event_date_field": "true",
            "has_hazard_field": "true",
            "can_be_ground_reference_candidate": "true",
        }])
        assert rows[0]["can_create_ground_reference"] == "false"
        assert rows[0]["can_create_training_label"] == "false"

    def test_audit_gate_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uh_phenomenon_temporal_gate_audit import audit_gate
        result = audit_gate(
            {"has_event_date_field": "false", "has_hazard_field": "false"},
            {"event_id": "PET_2022_02_15", "start_date": "2022-02-15",
             "hazard_scope": "mixed"},
            [])
        assert result["can_create_ground_reference"] == "false"
        assert result["temporal_gate_status"] == "BLOCKED"
        assert result["phenomenon_gate_status"] == "BLOCKED"

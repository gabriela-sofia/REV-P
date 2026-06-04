import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.protocolo_c.revp_v1ul_recife_common import run_sensitive_review_packager


def test_sensitive_field_is_redacted_and_literal_address_is_absent(tmp_path):
    router = tmp_path / "router.csv"
    with open(router, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_id", "candidate_row_id", "row_hash", "review_route",
            "event_window_match", "hazard_signal", "coordinate_status",
            "locality_status", "sensitive_review_required",
        ])
        writer.writeheader()
        writer.writerow({
            "event_id": "REC_2022_05_24_30",
            "candidate_row_id": "c1",
            "row_hash": "hash_only",
            "review_route": "ROUTE_LOCALITY_ONLY_REVIEW",
            "event_window_match": "event_core_window",
            "hazard_signal": "HAS_HAZARD_SIGNAL",
            "coordinate_status": "NO_COORDINATES",
            "locality_status": "ADDRESS_TEXT_AVAILABLE",
            "sensitive_review_required": "true",
        })
    out = tmp_path / "sensitive.csv"
    rows = run_sensitive_review_packager(str(out), str(router))
    content = out.read_text(encoding="utf-8")
    assert rows[0]["redaction_status"] == "REDACTED_HASH_ONLY"
    assert rows[0]["public_registry_safe"] == "true"
    assert "Rua " not in content
    assert "Avenida " not in content
    assert "CPF" not in content

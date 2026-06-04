import csv
import os
import shutil

from scripts.protocolo_c.revp_v1uk_recife_common import run_schema_audit

FIX = os.path.join("tests", "fixtures", "v1uk")


def _context(tmp_path, fixture_name):
    raw = tmp_path / "raw"
    raw.mkdir()
    safe = f"REC_2022_05_24_30__ckan__asset1__abcdef123456__{fixture_name}"
    shutil.copy(os.path.join(FIX, fixture_name), raw / safe)
    inv = tmp_path / "inventory.csv"
    man = tmp_path / "manifest.csv"
    with open(inv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["inventory_id", "event_id", "internal_path", "asset_type", "extension", "classification"])
        w.writeheader()
        w.writerow({"inventory_id": "FINV", "event_id": "REC_2022_05_24_30", "internal_path": safe,
                    "asset_type": "tabular", "extension": ".csv",
                    "classification": "DOCUMENTED_OCCURRENCE_TABLE_NO_GEOMETRY"})
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["safe_filename", "sha256", "file_size_bytes"])
        w.writeheader()
        w.writerow({"safe_filename": safe, "sha256": "hash", "file_size_bytes": "1"})
    return raw, inv, man


def test_schema_audit_detects_date_locality_hazard(tmp_path):
    raw, inv, man = _context(tmp_path, "defesa_civil_atendimento_sem_coordenada.csv")
    out = tmp_path / "schema.csv"
    rows = run_schema_audit(str(out), str(inv), str(man), str(raw))
    assert rows[0]["has_date_field"] == "true"
    assert rows[0]["has_hazard_field"] == "true"
    assert rows[0]["has_locality_field"] == "true"
    assert rows[0]["has_sensitive_fields"] == "true"

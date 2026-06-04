import csv
import os
import shutil

from scripts.protocolo_c.revp_v1uk_recife_common import run_coordinate_audit

FIX = os.path.join("tests", "fixtures", "v1uk")


def _write_context(tmp_path, fixture_name, ext, classification):
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
                    "asset_type": "tabular" if ext == ".csv" else "geospatial_vector",
                    "extension": ext, "classification": classification})
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["safe_filename", "sha256", "file_size_bytes"])
        w.writeheader()
        w.writerow({"safe_filename": safe, "sha256": "hash", "file_size_bytes": "1"})
    return raw, inv, man


def test_csv_lat_lon_becomes_coordinate_candidate(tmp_path, monkeypatch):
    raw, inv, man = _write_context(tmp_path, "atendimento_com_lat_lon.csv", ".csv", "TABLE_WITH_COORDINATES_CANDIDATE_FOR_REVIEW")
    monkeypatch.setenv("V1UK_INVENTORY_PATH", str(inv))
    monkeypatch.setenv("V1UK_MANIFEST_PATH", str(man))
    rows = run_coordinate_audit(str(tmp_path / "coord.csv"), str(raw))
    assert rows[0]["coordinate_classification"] == "OCCURRENCE_COORDINATES_CANDIDATE"


def test_geojson_contextual_not_occurrence(tmp_path, monkeypatch):
    raw, inv, man = _write_context(tmp_path, "defesa_civil_contextual.geojson", ".geojson", "CONTEXTUAL_OFFICIAL_LAYER")
    monkeypatch.setenv("V1UK_INVENTORY_PATH", str(inv))
    monkeypatch.setenv("V1UK_MANIFEST_PATH", str(man))
    rows = run_coordinate_audit(str(tmp_path / "coord.csv"), str(raw))
    assert rows[0]["coordinate_classification"] == "REGIONAL_CONTEXT_POINTS"

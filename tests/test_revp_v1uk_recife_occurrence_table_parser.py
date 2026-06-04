import csv
import os
import shutil

from scripts.protocolo_c.revp_v1uk_recife_common import discover_assets, parse_table_profile_for_asset

FIX = os.path.join("tests", "fixtures", "v1uk")


def test_parser_counts_window_and_does_not_emit_sensitive_literals(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    safe = "REC_2022_05_24_30__ckan__asset1__abcdef123456__campos_sensiveis.csv"
    shutil.copy(os.path.join(FIX, "campos_sensiveis.csv"), raw / safe)
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
    monkeypatch.setenv("V1UK_INVENTORY_PATH", str(inv))
    monkeypatch.setenv("V1UK_MANIFEST_PATH", str(man))
    asset = discover_assets(raw_dir=str(raw))[0]
    profile = parse_table_profile_for_asset(asset)
    assert profile["rows_in_event_window"] == "1"
    assert "Rua Sensivel" not in str(profile)
    assert "PROC-123" not in str(profile)

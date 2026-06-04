import csv
import json
import os

import scripts.protocolo_c.revp_v1uq_petropolis_common as common


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    configs = tmp_path / "configs" / "protocolo_c"
    raw = tmp_path / "raw"
    staging = tmp_path / "local_only" / "staging"
    reports = tmp_path / "local_only" / "reports"
    for path in [data, docs, configs, raw, staging, reports]:
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(configs))
    monkeypatch.setattr(common, "V1UP_RAW_DIR", str(raw))
    monkeypatch.setattr(common, "LOCAL_STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "LOCAL_REPORTS_DIR", str(reports))
    return data, docs, configs, raw, staging


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_pdf_payload(data, raw, name="doc.pdf"):
    (raw / name).write_bytes(b"%PDF synthetic")
    write_csv(data / "v1up_petropolis_download_manifest.csv", [
        "download_id", "event_id", "source_id", "record_id", "url",
        "url_sha1_12", "safe_filename", "basename", "format_hint",
        "download_status", "downloaded", "sha256", "size_bytes",
        "duplicate_of_sha256", "collision_status", "storage_scope",
        "blocking_reason", "notes",
    ], [{
        "download_id": "DL_v1up_0000", "event_id": "PET_2022_02_15",
        "source_id": "SGB_RIGEO", "record_id": "r0", "url": "",
        "url_sha1_12": "", "safe_filename": name, "basename": name,
        "format_hint": "pdf", "download_status": "DOWNLOADED",
        "downloaded": "true", "sha256": "", "size_bytes": "",
        "duplicate_of_sha256": "", "collision_status": "",
        "storage_scope": "RAW_LOCAL_SCOPE_REDACTED",
        "blocking_reason": "", "notes": "",
    }])


def seed_page_text(data, staging, pages, event_id="PET_2022_02_15", asset_id="PDF_DL_v1up_0000"):
    write_csv(data / "v1uq_petropolis_pdf_text_extraction_registry.csv", common.TEXT_COLUMNS, [{
        "text_extract_id": "TEXT_v1uq_0000", "event_id": event_id,
        "asset_id": asset_id, "source_id": "SGB_RIGEO", "pdf_sha256": "abc",
        "page_count": str(len(pages)), "extraction_backend": "fixture",
        "extraction_status": "EXTRACTED", "extracted_text_local_hash": "hash",
        "total_chars": "10", "pages_with_text": str(len(pages)),
        "contains_flood_terms": "true", "contains_landslide_terms": "true",
        "contains_geodata_terms": "true", "contains_date_terms": "true",
        "can_create_ground_reference": "false", "can_create_training_label": "false",
        "notes": "",
    }])
    payload = [{"page_number": i + 1, "text": text} for i, text in enumerate(pages)]
    with open(staging / f"{asset_id}_page_text.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_pdf_text_extractor_keeps_full_text_local_and_public_csv_metadata_only(tmp_path, monkeypatch):
    data, _, _, raw, staging = set_env(tmp_path, monkeypatch)
    seed_pdf_payload(data, raw)
    monkeypatch.setattr(common, "extract_pages_from_bytes", lambda data: ("fixture", [(1, "alagamento 15/02/2022")], "EXTRACTED"))
    rows = common.run_pdf_text_extractor()
    assert rows[0]["contains_flood_terms"] == "true"
    assert "alagamento" not in ",".join(rows[0].values())
    assert os.path.exists(staging / "PDF_DL_v1up_0000_page_text.json")


def test_pdf_backend_missing_fail_closed(tmp_path, monkeypatch):
    data, _, _, raw, _ = set_env(tmp_path, monkeypatch)
    seed_pdf_payload(data, raw)
    monkeypatch.setattr(common, "extract_pages_from_bytes", lambda data: ("PDF_BACKEND_MISSING", [], "PDF_BACKEND_MISSING"))
    rows = common.run_pdf_text_extractor()
    assert rows[0]["extraction_status"] == "PDF_BACKEND_MISSING"
    assert rows[0]["page_count"] == "0"

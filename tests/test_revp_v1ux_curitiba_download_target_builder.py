import csv
import os
import shutil
import zipfile
from pathlib import Path

import scripts.protocolo_c.revp_v1ux_curitiba_common as common


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v1ux"


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    raw = tmp_path / "local_only" / "protocolo_c" / "curitiba_public_evidence_download" / "raw" / "v1ux"
    staging = tmp_path / "local_only" / "protocolo_c" / "curitiba_public_evidence_download" / "staging" / "v1ux"
    quarantine = tmp_path / "local_only" / "protocolo_c" / "curitiba_public_evidence_download" / "quarantine" / "v1ux"
    reports = tmp_path / "local_only" / "protocolo_c" / "curitiba_public_evidence_download" / "reports" / "v1ux"
    for path in (data, docs, cfg, raw, staging, quarantine, reports):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "RAW_DIR", str(raw))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "QUARANTINE_DIR", str(quarantine))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data, raw


def install_inputs(data):
    write_csv(data / "v1uw_curitiba_event_candidate_status.csv", [
        "candidate_event_id", "proposed_event_id", "candidate_status",
        "can_enter_event_patch_linkage", "can_create_ground_reference",
        "can_create_training_label",
    ], [{
        "candidate_event_id": "CE_v1uv_0000",
        "proposed_event_id": "CUR_2022_01_15",
        "candidate_status": "CURITIBA_EVENT_CANDIDATE_HYDROMET_SUPPORTED",
        "can_enter_event_patch_linkage": "true",
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
    }])
    write_csv(data / "v1uv_curitiba_public_event_discovery.csv", [
        "discovery_id", "source_id", "result_url", "title_hash",
    ], [{
        "discovery_id": "DISC_0000",
        "source_id": "curitiba_prefeitura_news",
        "result_url": "https://www.curitiba.pr.gov.br/noticias/defesa-civil-alerta-para-mais-chuva-durante-a-madrugada/62283",
        "title_hash": "titlehash",
    }, {
        "discovery_id": "DISC_0001",
        "source_id": "defesa_civil_pr",
        "result_url": "https://www.defesacivil.pr.gov.br/Noticia/Chuvas-fortes-causam-transtornos-em-Curitiba-e-no-Litoral-Defesa-Civil-se-mobiliza-para",
        "title_hash": "titlehash2",
    }])
    write_csv(data / "v1uv_curitiba_open_data_registry.csv", [
        "open_data_record_id", "source_id", "resource_format", "package_url", "dataset_title_hash",
    ], [{
        "open_data_record_id": "OD_0000",
        "source_id": "curitiba_open_data",
        "resource_format": "CSV",
        "package_url": "https://dadosabertos.curitiba.pr.gov.br/",
        "dataset_title_hash": "openhash",
    }])
    write_csv(data / "v1uw_curitiba_open_data_resource_deepening.csv", [
        "resource_deepening_id", "source_id", "resource_format", "resource_class",
        "resource_url_hash", "resource_name_hash",
    ], [{
        "resource_deepening_id": "ODD_0000", "source_id": "curitiba_open_data",
        "resource_format": "CSV", "resource_class": "occurrence table candidate",
        "resource_url_hash": "csvhash", "resource_name_hash": "namecsv",
    }, {
        "resource_deepening_id": "ODD_0001", "source_id": "curitiba_open_data",
        "resource_format": "GeoJSON", "resource_class": "context",
        "resource_url_hash": "geohash", "resource_name_hash": "namegeo",
    }, {
        "resource_deepening_id": "ODD_0002", "source_id": "curitiba_open_data",
        "resource_format": "ZIP", "resource_class": "context",
        "resource_url_hash": "ziphash", "resource_name_hash": "namezip",
    }, {
        "resource_deepening_id": "ODD_0003", "source_id": "curitiba_open_data",
        "resource_format": "PDF", "resource_class": "document",
        "resource_url_hash": "pdfhash", "resource_name_hash": "namepdf",
    }])
    write_csv(data / "v1uw_curitiba_geocuritiba_layer_deepening.csv", [
        "geocuritiba_deepening_id", "layer_name", "layer_class",
    ], [{
        "geocuritiba_deepening_id": "GEO_0000",
        "layer_name": "bacias_hidrograficas",
        "layer_class": "administrative_context",
    }])


def seed_fixture_downloads(data, raw):
    raw.mkdir(parents=True, exist_ok=True)
    fixture_files = [
        ("DL_v1ux_0000", "event_occurrence.csv", ".csv"),
        ("DL_v1ux_0001", "date_no_hazard.csv", ".csv"),
        ("DL_v1ux_0002", "coordinates_context.csv", ".csv"),
        ("DL_v1ux_0003", "administrative_context.geojson", ".geojson"),
        ("DL_v1ux_0004", "possible_occurrence.geojson", ".geojson"),
        ("DL_v1ux_0005", "zip_with_csv.zip", ".zip"),
        ("DL_v1ux_0006", "document_only.pdf", ".pdf"),
        ("DL_v1ux_0007", "open_data_metadata.json", ".json"),
        ("DL_v1ux_0008", "hydromet_only.csv", ".csv"),
    ]
    rows = []
    for idx, (download_id, filename, ext) in enumerate(fixture_files):
        dest = raw / filename
        if filename == "zip_with_csv.zip":
            with zipfile.ZipFile(dest, "w") as z:
                z.write(FIXTURE_DIR / "zip_member.csv", "zip_member.csv")
        else:
            shutil.copyfile(FIXTURE_DIR / filename, dest)
        rows.append({
            "download_id": download_id,
            "download_target_id": f"DT_v1ux_{idx:04d}",
            "candidate_event_id": "CE_v1uv_0000",
            "source_id": "fixture",
            "url_sha1_12": f"url{idx:02d}",
            "safe_filename": filename,
            "local_path_hash": common.hash_text(filename, 24),
            "sha256": common.sha256_file(dest),
            "file_size_bytes": str(dest.stat().st_size),
            "mime_type": "",
            "extension": ext,
            "download_status": "FIXTURE",
            "duplicate_status": "UNIQUE",
            "raw_data_versioned": "false",
            "notes": "fixture metadata only",
        })
    write_csv(data / "v1ux_curitiba_public_artifact_download_manifest.csv", common.DOWNLOAD_COLUMNS, rows)
    return rows


def run_fixture_pipeline(data, raw):
    seed_fixture_downloads(data, raw)
    common.run_artifact_inventory(common.parse_args([]))
    common.run_schema_audit(common.parse_args([]))
    common.run_geodata_metadata_audit(common.parse_args([]))
    common.run_event_table_detector(common.parse_args([]))
    common.run_hazard_date_locality_field_mapper(common.parse_args([]))
    common.run_candidate_evidence_classifier(common.parse_args([]))
    common.run_event_patch_readiness_update(common.parse_args([]))


def test_download_target_builder_keeps_targets_review_only(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_download_target_builder(common.parse_args([]))
    assert len(rows) == 7
    assert {r["download_allowed"] for r in rows if r["source_id"] == "geocuritiba"} == {"false"}
    assert any(r["priority_class"] == "HIGH_PRIORITY_EVENT_TABLE" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)

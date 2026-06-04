import csv
import os
from pathlib import Path

import scripts.protocolo_c.revp_v1uy_curitiba_common as common


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v1uy"


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
    v1ux_raw = tmp_path / "local_only" / "v1ux_raw"
    raw = tmp_path / "local_only" / "v1uy" / "raw"
    staging = tmp_path / "local_only" / "v1uy" / "staging"
    quarantine = tmp_path / "local_only" / "v1uy" / "quarantine"
    reports = tmp_path / "local_only" / "v1uy" / "reports"
    for p in (data, docs, cfg, v1ux_raw, raw, staging, quarantine, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "V1UX_RAW_DIR", str(v1ux_raw))
    monkeypatch.setattr(common, "RAW_DIR", str(raw))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "QUARANTINE_DIR", str(quarantine))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data, v1ux_raw


def install_base_inputs(data, v1ux_raw):
    write_csv(data / "v1uw_curitiba_event_candidate_status.csv", [
        "candidate_event_id", "proposed_event_id",
    ], [{"candidate_event_id": "CE_v1uv_0000", "proposed_event_id": "CUR_2022_01_15"}])
    write_csv(data / "v1ux_curitiba_download_target_registry.csv", [
        "download_target_id", "candidate_event_id", "source_registry", "source_record_id",
        "source_id", "resource_url_hash", "resource_url", "expected_format", "priority_class",
        "download_allowed",
    ], [{
        "download_target_id": "DT_v1ux_0003", "candidate_event_id": "CE_v1uv_0000",
        "source_registry": "v1uw", "source_record_id": "ODD_v1uw_0001",
        "source_id": "curitiba_open_data", "resource_url_hash": "hash_geo",
        "resource_url": "https://dadosabertos.curitiba.pr.gov.br/", "expected_format": "GeoJSON",
        "priority_class": "HIGH_PRIORITY_GEODATA", "download_allowed": "true",
    }, {
        "download_target_id": "DT_v1ux_0004", "candidate_event_id": "CE_v1uv_0000",
        "source_registry": "v1uw", "source_record_id": "ODD_v1uw_0002",
        "source_id": "curitiba_open_data", "resource_url_hash": "hash_zip",
        "resource_url": "https://dadosabertos.curitiba.pr.gov.br/", "expected_format": "ZIP",
        "priority_class": "MEDIUM_PRIORITY_CONTEXT_LAYER", "download_allowed": "true",
    }])
    geo_html = v1ux_raw / "bad_geo.geojson"
    zip_html = v1ux_raw / "bad_zip.zip"
    geo_html.write_bytes((FIXTURE_DIR / "geojson_target_returns_html.html").read_bytes())
    zip_html.write_bytes((FIXTURE_DIR / "zip_target_returns_html.html").read_bytes())
    write_csv(data / "v1ux_curitiba_public_artifact_download_manifest.csv", [
        "download_id", "download_target_id", "candidate_event_id", "source_id", "safe_filename",
        "extension", "download_status", "raw_data_versioned",
    ], [{
        "download_id": "DL_v1ux_0003", "download_target_id": "DT_v1ux_0003",
        "candidate_event_id": "CE_v1uv_0000", "source_id": "curitiba_open_data",
        "safe_filename": "bad_geo.geojson", "extension": ".geojson",
        "download_status": "DOWNLOADED", "raw_data_versioned": "false",
    }, {
        "download_id": "DL_v1ux_0004", "download_target_id": "DT_v1ux_0004",
        "candidate_event_id": "CE_v1uv_0000", "source_id": "curitiba_open_data",
        "safe_filename": "bad_zip.zip", "extension": ".zip",
        "download_status": "DOWNLOADED", "raw_data_versioned": "false",
    }])
    write_csv(data / "v1ux_curitiba_artifact_inventory.csv", [
        "inventory_id", "download_id", "artifact_type", "inventory_status",
    ], [{
        "inventory_id": "INV1", "download_id": "DL_v1ux_0003",
        "artifact_type": "GeoJSON", "inventory_status": "INVENTORY_ERROR_JSONDecodeError",
    }, {
        "inventory_id": "INV2", "download_id": "DL_v1ux_0004",
        "artifact_type": "ZIP", "inventory_status": "INVENTORY_ERROR_BadZipFile",
    }])
    write_csv(data / "v1uv_curitiba_geocuritiba_registry.csv", [
        "geocuritiba_record_id", "source_id", "service_url", "layer_name",
        "layer_id", "geometry_type", "spatial_reference", "fields", "layer_class",
    ], [{
        "geocuritiba_record_id": "GEO_v1uv_0000", "source_id": "geocuritiba",
        "service_url": "https://geocuritiba.ippuc.org.br/", "layer_name": "drenagem_canais",
        "layer_id": "drenagem", "geometry_type": "LineString",
        "spatial_reference": "EPSG:31982", "fields": "nome_canal|bacia",
        "layer_class": "infrastructure",
    }, {
        "geocuritiba_record_id": "GEO_v1uv_0001", "source_id": "geocuritiba",
        "service_url": "https://geocuritiba.ippuc.org.br/", "layer_name": "defesa_civil_ocorrencias_alagamento",
        "layer_id": "ocorrencias", "geometry_type": "Point",
        "spatial_reference": "EPSG:31982", "fields": "data_evento|tipo_ocorrencia|bairro|Shape",
        "layer_class": "occurrence",
    }])
    write_csv(data / "v1uv_curitiba_open_data_registry.csv", [
        "open_data_record_id", "source_id", "package_url", "resource_format",
    ], [{"open_data_record_id": "OD_v1uv_0001", "source_id": "curitiba_open_data",
         "package_url": "https://dadosabertos.curitiba.pr.gov.br/", "resource_format": "GeoJSON"}])
    write_csv(data / "v1uw_curitiba_open_data_resource_deepening.csv", [
        "resource_deepening_id", "source_id", "resource_format", "resource_class",
    ], [{"resource_deepening_id": "ODD_v1uw_0001", "source_id": "curitiba_open_data",
         "resource_format": "GeoJSON", "resource_class": "context layer"}])


def run_pipeline(data, v1ux_raw):
    install_base_inputs(data, v1ux_raw)
    common.run_content_mismatch_resolver(common.parse_args([]))
    common.run_geodata_endpoint_probe(common.parse_args([]))
    common.run_layer_metadata_extractor(common.parse_args([]))
    common.run_feature_schema_sampler(common.parse_args([]))
    common.run_context_layer_classifier(common.parse_args([]))
    common.run_possible_occurrence_layer_audit(common.parse_args([]))
    common.run_controlled_feature_download_planner(common.parse_args([]))
    common.run_event_patch_readiness_update(common.parse_args([]))
    common.run_ground_reference_blocker_builder(common.parse_args([]))


def test_content_mismatch_resolver_identifies_html_masked_as_geodata(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, v1ux_raw)
    rows = common.run_content_mismatch_resolver(common.parse_args([]))
    assert len(rows) == 2
    assert {r["detected_type"] for r in rows} == {"portal page"}
    assert {r["mismatch_status"] for r in rows} == {"RECOVERABLE_DIRECT_DOWNLOAD_LINK"}
    assert all(r["raw_data_versioned"] == "false" for r in rows)

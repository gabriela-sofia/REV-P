import csv
import os

import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    raw = tmp_path / "local_only" / "raw"
    staging = tmp_path / "local_only" / "staging"
    reports = tmp_path / "local_only" / "reports"
    for p in (data, docs, cfg, raw, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "RAW_DIR", str(raw))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data, raw


def install_inputs(data):
    write_csv(os.path.join(data, "v1uv_curitiba_candidate_event_registry.csv"), [
        "candidate_event_id", "event_id_candidate", "city", "uf", "start_date", "end_date",
        "hazard_scope", "official_source_status", "source_url_hash", "source_registry",
        "evidence_type", "event_candidate_class", "confidence_score",
        "can_enter_multiregion_registry", "can_create_ground_reference",
        "can_create_training_label", "blocker", "notes",
    ], [{"candidate_event_id": "CE_v1uv_0000", "event_id_candidate": "CUR_2022_01_15",
         "city": "Curitiba", "uf": "PR", "start_date": "2022-01-15",
         "end_date": "2022-01-15", "hazard_scope": "urban_flooding|intense_rain",
         "official_source_status": "OFFICIAL_PUBLIC_SOURCE", "source_url_hash": "urlhash",
         "source_registry": "v1uv", "evidence_type": "official", "event_candidate_class": "CURITIBA_EVENT_CANDIDATE_PUBLIC_OFFICIAL",
         "confidence_score": "90", "can_enter_multiregion_registry": "true",
         "can_create_ground_reference": "false", "can_create_training_label": "false",
         "blocker": "", "notes": ""}])
    write_csv(os.path.join(data, "v1uv_curitiba_public_event_discovery.csv"), [
        "discovery_id", "source_id", "result_url", "http_status", "content_type", "title_hash",
        "date_signal", "hazard_signal", "official_source_status", "event_specificity",
        "candidate_status", "blocking_reason", "notes",
    ], [{"discovery_id": "D1", "source_id": "curitiba_prefeitura_news",
         "result_url": "https://www.curitiba.pr.gov.br/noticias/x", "http_status": "200",
         "content_type": "text/html", "title_hash": "t", "date_signal": "2022-01-15",
         "hazard_signal": "alagamento", "official_source_status": "OFFICIAL_PUBLIC_SOURCE",
         "event_specificity": "DATED_HAZARD_CURITIBA_EVENT", "candidate_status": "PUBLIC_OFFICIAL_EVENT_CANDIDATE_SIGNAL",
         "blocking_reason": "", "notes": ""}])
    write_csv(os.path.join(data, "v1uv_curitiba_hydromet_source_registry.csv"), [
        "hydromet_record_id", "candidate_event_id", "source_id", "source_name", "station_or_source",
        "municipality", "date_signal", "hydromet_signal", "official_source_status",
        "temporal_support_status", "can_be_observed_occurrence", "notes",
    ], [{"hydromet_record_id": "H1", "candidate_event_id": "CE_v1uv_0000",
         "source_id": "curitiba_prefeitura_news", "source_name": "official",
         "station_or_source": "Simepar|Cemaden", "municipality": "Curitiba",
         "date_signal": "2022-01-15", "hydromet_signal": "rain",
         "official_source_status": "OFFICIAL_PUBLIC_SOURCE",
         "temporal_support_status": "TEMPORAL_HYDROMET_SUPPORT",
         "can_be_observed_occurrence": "false", "notes": ""}])
    write_csv(os.path.join(data, "v1uv_curitiba_geocuritiba_registry.csv"), [
        "geocuritiba_record_id", "layer_name", "geometry_type", "spatial_reference", "fields",
    ], [{"geocuritiba_record_id": "G1", "layer_name": "bacias_hidrograficas",
         "geometry_type": "Polygon", "spatial_reference": "EPSG:31982", "fields": "nome"}])
    write_csv(os.path.join(data, "v1uv_curitiba_open_data_registry.csv"), [
        "open_data_record_id", "source_id", "resource_format", "package_url", "dataset_title_hash",
    ], [{"open_data_record_id": "OD1", "source_id": "curitiba_open_data",
         "resource_format": "CSV", "package_url": "https://dadosabertos.curitiba.pr.gov.br/", "dataset_title_hash": "abc"}])
    write_csv(os.path.join(data, "v1us_patch_registry_resolution.csv"), [
        "patch_resolution_id", "patch_id", "region", "has_sentinel_date",
    ], [{"patch_resolution_id": "P1", "patch_id": "CUR_001", "region": "CUR", "has_sentinel_date": "false"}])


def test_snapshotter_saves_raw_only_in_local_only(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_event_source_snapshotter(common.parse_args(["--dry-run"]))
    assert rows[0]["snapshot_status"] == "SNAPSHOT_FALLBACK_OFFICIAL_METADATA"
    assert rows[0]["public_official_source"] == "true"
    assert list(raw.glob("*.html"))

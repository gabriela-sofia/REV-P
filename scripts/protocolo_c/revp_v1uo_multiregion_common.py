#!/usr/bin/env python3
"""v1uo Multi-Region Public Evidence Replication Engine.

Registry-driven, non-operational engine. It does not download raw data, execute
overlay, infer coordinates, or create training targets.
"""

import argparse
import csv
import hashlib
import os
from collections import Counter

PROTOCOL_VERSION = "v1uo"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
MAX_STATUS = "MULTI_REGION_EVIDENCE_CANDIDATE_DISCOVERY_NON_OPERATIONAL"

EVENT_COLUMNS = [
    "event_id", "region", "city", "uf", "start_date", "end_date",
    "hazard_scope", "current_best_evidence_status", "has_temporal_anchor",
    "has_official_source", "has_locality_only_evidence",
    "has_coordinate_evidence", "has_observed_geometry", "has_overlay",
    "has_ground_reference", "main_blocker", "next_programming_action",
    "notes",
]

SOURCE_COLUMNS = [
    "source_id", "region", "source_name", "source_type", "base_url",
    "access_mode", "expected_artifact_types", "can_contain_observed_geometry",
    "can_contain_locality_only_records", "can_contain_temporal_anchor",
    "can_contain_context_only", "priority", "crawler_strategy", "notes",
]

ADAPTER_COLUMNS = [
    "adapter_id", "region", "adapter_name", "supported_source_types",
    "hazard_terms", "locality_terms", "coordinate_terms", "date_terms",
    "geometry_terms", "known_blockers", "output_schema_version", "notes",
]

DISCOVERY_COLUMNS = [
    "discovery_id", "event_id", "region", "source_id", "adapter_id",
    "candidate_url", "http_status", "content_type", "discovery_status",
    "candidate_artifact_type", "event_specificity", "priority",
    "blocking_reason", "notes",
]

SCHEMA_COLUMNS = [
    "schema_audit_id", "event_id", "region", "source_id", "asset_count",
    "row_count", "has_date_field", "has_hazard_field", "has_locality_field",
    "has_coordinate_fields", "has_geometry", "schema_status",
    "normalized_status", "main_blocker", "notes",
]

ROUTER_COLUMNS = [
    "route_id", "event_id", "region", "candidate_class",
    "evidence_strength", "has_temporal_support", "has_official_source",
    "has_locality_support", "has_coordinate_support", "has_geometry_support",
    "can_advance_to_overlay_preflight", "can_advance_to_ground_reference",
    "can_create_training_label", "blocker", "required_next_action", "notes",
]

MATRIX_COLUMNS = [
    "matrix_id", "event_id", "region", "dimension", "classification",
    "basis", "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "notes",
]

RANK_COLUMNS = [
    "rank", "event_id", "region", "opportunity_class", "current_strength",
    "main_blocker", "expected_programming_value",
    "expected_ground_truth_value", "overclaim_risk",
    "recommended_next_version", "recommended_next_action", "notes",
]

PACKAGE_COLUMNS = [
    "package_id", "event_id", "region", "patch_id", "sentinel_scene_date",
    "evidence_sources_attached", "temporal_anchor_status", "hydromet_status",
    "locality_status", "coordinate_status", "geometry_status",
    "dino_review_support_status", "overlay_status", "ground_reference_status",
    "package_status", "can_create_ground_reference",
    "can_create_training_label", "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UO_ARTIFACTS = [
    "configs/protocolo_c/v1uo_multiregion_events.yaml",
    "configs/protocolo_c/v1uo_multiregion_public_sources.yaml",
    "configs/protocolo_c/v1uo_region_adapter_policy.yaml",
    "configs/protocolo_c/v1uo_schema_audit_policy.yaml",
    "configs/protocolo_c/v1uo_candidate_routing_policy.yaml",
    "configs/protocolo_c/v1uo_ground_truth_opportunity_policy.yaml",
    "configs/protocolo_c/v1uo_event_patch_package_policy.yaml",
    "datasets/protocolo_c/v1uo_multiregion_event_registry.csv",
    "datasets/protocolo_c/v1uo_public_source_registry.csv",
    "datasets/protocolo_c/v1uo_region_adapter_registry.csv",
    "datasets/protocolo_c/v1uo_multiregion_public_discovery_registry.csv",
    "datasets/protocolo_c/v1uo_multiregion_schema_audit_registry.csv",
    "datasets/protocolo_c/v1uo_multiregion_candidate_router.csv",
    "datasets/protocolo_c/v1uo_multiregion_evidence_matrix.csv",
    "datasets/protocolo_c/v1uo_ground_truth_opportunity_ranker.csv",
    "datasets/protocolo_c/v1uo_event_patch_package_prebuild_registry.csv",
    "datasets/protocolo_c/v1uo_next_actions_registry.csv",
    "datasets/protocolo_c/v1uo_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uo_multiregion_replication_engine.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uo_multiregion_replication_engine.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1uo.md",
]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_text(value):
    return "true" if bool(value) else "false"


def int_value(value):
    try:
        return int(value or 0)
    except ValueError:
        return 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(path):
    return path.replace("\\", "/")


def by_key(rows, key):
    return {r.get(key, ""): r for r in rows}


def event_candidates():
    rows = load_csv(os.path.join(DATASET_DIR, "event_candidate_registry.csv"))
    has_curitiba = any((r.get("region") or "").upper() == "CUR" or "curitiba" in (r.get("city", "").lower()) for r in rows)
    if not has_curitiba:
        rows.append({
            "event_id": "CUR_EVENT_REGISTRY_MISSING",
            "region": "CUR",
            "city": "Curitiba",
            "uf": "PR",
            "start_date": "",
            "end_date": "",
            "hazard_scope": "unknown",
            "current_level": "",
            "current_status": "EVENT_REGISTRY_MISSING_FOR_CURITIBA",
            "blocking_reason": "event_registry_missing_for_curitiba",
            "priority": "3",
            "notes": "No clear Curitiba event in event_candidate_registry.csv.",
        })
    return rows


def recife_status():
    rows = load_csv(os.path.join(DATASET_DIR, "v1un_recife_protocol_c_status_registry.csv"))
    return rows[0] if rows else {}


def hydromet_scorecards():
    return by_key(load_csv(os.path.join(DATASET_DIR, "v1uf_event_hydromet_scorecard.csv")), "event_id")


def evidence_scorecards():
    return by_key(load_csv(os.path.join(DATASET_DIR, "v1ue_event_evidence_scorecard.csv")), "event_id")


def source_rows():
    return [
        ("SRC_REC_CKAN", "REC", "Recife CKAN Dados Abertos", "CKAN", "https://dados.recife.pe.gov.br", "public_catalog", "csv|geojson|json", "false", "true", "false", "true", "1", "ckan_package_search", "Recife public portal and Defesa Civil records."),
        ("SRC_REC_DEFESA_CIVIL", "REC", "Defesa Civil Recife", "OPEN_DATA_PORTAL", "https://dados.recife.pe.gov.br", "public_catalog", "csv", "false", "true", "false", "true", "1", "ckan_resource_scan", "Locality-only records observed in prior chain."),
        ("SRC_PET_SGB", "PET", "SGB CPRM", "DOCUMENT_REPOSITORY", "https://www.sgb.gov.br", "public_catalog", "pdf|zip|shp|geojson", "true", "true", "false", "true", "1", "document_and_geodata_scan", "Priority for observed geohazard geometry."),
        ("SRC_PET_RIGEO", "PET", "RIGeo", "DOCUMENT_REPOSITORY", "https://rigeo.sgb.gov.br", "public_catalog", "pdf|zip|shp", "true", "true", "false", "true", "1", "repository_search", "Priority for Petropolis event artifacts."),
        ("SRC_PET_GEOSGB", "PET", "GeoSGB", "GEOSERVER", "https://geosgb.sgb.gov.br", "public_service", "wms|wfs|geojson", "true", "false", "false", "true", "2", "service_capabilities", "Geospatial service candidate."),
        ("SRC_PET_DRM_RJ", "PET", "DRM RJ", "DOCUMENT_REPOSITORY", "https://www.drm.rj.gov.br", "public_catalog", "pdf|map|shp", "true", "true", "false", "true", "2", "document_scan", "Regional geohazard source."),
        ("SRC_PET_DEFESA_CIVIL", "PET", "Defesa Civil Petropolis", "OPEN_DATA_PORTAL", "https://www.petropolis.rj.gov.br", "public_catalog", "csv|pdf|html", "false", "true", "true", "true", "2", "municipal_portal_scan", "Administrative records possible."),
        ("SRC_CUR_GEOCURITIBA", "CUR", "GeoCuritiba", "ARCGIS_REST", "https://geocuritiba.curitiba.pr.gov.br", "public_service", "feature_service|geojson", "true", "true", "false", "true", "1", "arcgis_rest_inventory", "Potential municipal geometry source."),
        ("SRC_CUR_DADOS_ABERTOS", "CUR", "Dados Abertos Curitiba", "OPEN_DATA_PORTAL", "https://www.curitiba.pr.gov.br/dadosabertos", "public_catalog", "csv|json|geojson", "true", "true", "false", "true", "1", "portal_catalog_scan", "Public data source candidate."),
        ("SRC_CUR_DEFESA_CIVIL", "CUR", "Defesa Civil Curitiba", "OPEN_DATA_PORTAL", "https://www.curitiba.pr.gov.br", "public_catalog", "html|pdf|csv", "false", "true", "true", "true", "2", "municipal_portal_scan", "Administrative records possible."),
        ("SRC_SIMEPAR", "CUR", "Simepar", "HYDROMET_SERIES", "https://www.simepar.br", "public_catalog", "series|csv|api", "false", "false", "true", "true", "2", "hydromet_source_check", "Hydromet anchor candidate if public access exists."),
        ("SRC_COPERNICUS_EMS", "ALL", "Copernicus EMS", "OPERATIONAL_MAPPING", "https://emergency.copernicus.eu", "public_catalog", "map|vector|pdf", "true", "false", "true", "true", "2", "activation_search", "Only useful with correct activation."),
        ("SRC_CHARTER", "ALL", "International Charter", "VHR_CONTEXT", "https://disasterscharter.org", "public_catalog", "activation|map|vhr_context", "true", "false", "true", "true", "3", "activation_search", "Context only unless event-specific product exists."),
        ("SRC_INMET", "ALL", "INMET", "HYDROMET_SERIES", "https://bdmep.inmet.gov.br", "public_catalog", "csv|zip|series", "false", "false", "true", "true", "1", "hydromet_series_registry", "Temporal anchor source."),
        ("SRC_ANA", "ALL", "ANA Hidro", "HYDROMET_SERIES", "https://www.snirh.gov.br", "public_catalog", "series|csv", "false", "false", "true", "true", "2", "hydromet_series_registry", "Hydrological context source."),
        ("SRC_CEMADEN", "ALL", "Cemaden", "HYDROMET_SERIES", "https://www.gov.br/cemaden", "public_catalog", "series|alert|csv", "false", "true", "true", "true", "2", "alert_and_series_scan", "Alerts and hydromet context."),
    ]


def run_multiregion_event_registry_builder(out_path=None):
    rec = recife_status()
    hyd = hydromet_scorecards()
    ev_score = evidence_scorecards()
    rows = []
    for e in event_candidates():
        event_id = e.get("event_id", "")
        region = e.get("region", "")
        hyd_row = hyd.get(event_id, {})
        score = ev_score.get(event_id, {})
        if event_id == "REC_2022_05_24_30":
            status = rec.get("new_status") or "LOCALITY_ONLY_HUMAN_REVIEW_EVIDENCE_CONSOLIDATED"
            has_locality = "true"
            has_coord = "false"
            blocker = "coordinate_support_blocked"
            next_action = "RECIFE_COORDINATE_RECOVERY_FROM_PUBLIC_CKAN"
        elif event_id == "PET_2024_03_21_28":
            status = "TEMPORAL_HYDROMET_ANCHOR_CONFIRMED" if hyd_row.get("has_temporal_anchor") == "true" else score.get("classification", "CANDIDATE_UNDER_REVIEW")
            has_locality = "false"
            has_coord = bool_text(hyd_row.get("has_station_coordinates") == "true")
            blocker = "PHENOMENON_SEPARATION_REQUIRED"
            next_action = "PETROPOLIS_PUBLIC_GEOMETRY_DEEPENING"
        elif event_id == "PET_2022_02_15":
            status = "BLOCKED_PHENOMENON_SEPARATION_REQUIRED"
            has_locality = "false"
            has_coord = bool_text(hyd_row.get("has_station_coordinates") == "true")
            blocker = "PHENOMENON_SEPARATION_REQUIRED"
            next_action = "PETROPOLIS_PHENOMENON_SEPARATION_AND_GEOMETRY_DEEPENING"
        elif region == "CUR":
            status = "NEEDS_EVENT_REGISTRY_OR_PUBLIC_SOURCE_DEEPENING"
            has_locality = "false"
            has_coord = "false"
            blocker = "EVENT_REGISTRY_MISSING_FOR_CURITIBA"
            next_action = "CURITIBA_EVENT_REGISTRY_AND_PUBLIC_SOURCE_DISCOVERY"
        else:
            status = e.get("current_status", "CANDIDATE_UNDER_REVIEW")
            has_locality = "false"
            has_coord = "false"
            blocker = e.get("blocking_reason", "")
            next_action = "REGION_SPECIFIC_PUBLIC_SOURCE_DEEPENING"
        rows.append({
            "event_id": event_id,
            "region": region,
            "city": e.get("city", ""),
            "uf": e.get("uf", ""),
            "start_date": e.get("start_date", ""),
            "end_date": e.get("end_date", ""),
            "hazard_scope": e.get("hazard_scope", ""),
            "current_best_evidence_status": status,
            "has_temporal_anchor": bool_text(hyd_row.get("has_temporal_anchor") == "true" or bool(e.get("start_date"))),
            "has_official_source": "true" if region in {"REC", "PET"} else "false",
            "has_locality_only_evidence": has_locality,
            "has_coordinate_evidence": has_coord,
            "has_observed_geometry": "false",
            "has_overlay": "false",
            "has_ground_reference": "false",
            "main_blocker": blocker,
            "next_programming_action": next_action,
            "notes": "v1uo non-operational multiregion registry; no overlay and no label creation.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_multiregion_event_registry.csv")
    write_csv(out_path, EVENT_COLUMNS, rows)
    print(f"[v1uo event registry] rows={len(rows)} -> {out_path}")
    return rows


def run_public_source_registry_builder(out_path=None):
    rows = []
    for row in source_rows():
        rows.append(dict(zip(SOURCE_COLUMNS, row)))
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_public_source_registry.csv")
    write_csv(out_path, SOURCE_COLUMNS, rows)
    print(f"[v1uo source registry] rows={len(rows)} -> {out_path}")
    return rows


def run_region_adapter_factory(out_path=None):
    adapters = [
        ("ADAPT_REC_CKAN", "REC", "recife_ckan_adapter", "CKAN|OPEN_DATA_PORTAL", "alagamento|inundacao|chuva|deslizamento|defesa civil", "bairro|localidade|endereco|regional", "latitude|longitude|geometry", "data|date|dt", "geometry|geom|geojson", "locality_only_no_geometry|coordinate_support_blocked", "v1uo_common_candidate_schema", "Recife adapter derived from v1uk-v1un outputs."),
        ("ADAPT_PET_GEOHAZARD", "PET", "petropolis_geohazard_adapter", "DOCUMENT_REPOSITORY|GEOSERVER|HYDROMET_SERIES", "deslizamento|inundacao|alagamento|chuva", "bairro|localidade|setor|bacia", "latitude|longitude|geometry|shp", "data|date|periodo", "geometry|shp|wfs|geojson", "phenomenon_separation_required|geometry_missing", "v1uo_common_candidate_schema", "Petropolis adapter prioritizes geometry and phenomenon separation."),
        ("ADAPT_CUR_PUBLIC_GEO", "CUR", "curitiba_public_geo_adapter", "ARCGIS_REST|OPEN_DATA_PORTAL|HYDROMET_SERIES", "alagamento|inundacao|drenagem|risco|chuva", "bairro|regional|logradouro", "latitude|longitude|geometry|shape", "data|date|dt", "geometry|feature|geojson", "event_registry_missing_for_curitiba|source_discovery_pending", "v1uo_common_candidate_schema", "Curitiba adapter requires event registry before promotion."),
        ("ADAPT_GENERIC_HYDROMET", "ALL", "generic_hydromet_adapter", "HYDROMET_SERIES", "chuva|precipitacao|nivel|vazao", "station|municipio", "station_latitude|station_longitude", "data|timestamp", "station_point", "station_is_context_not_observed_event_geometry", "v1uo_common_candidate_schema", "Temporal anchor adapter."),
        ("ADAPT_GENERIC_MAPPING", "ALL", "generic_operational_mapping_adapter", "OPERATIONAL_MAPPING|VHR_CONTEXT", "flood|landslide|disaster|emergency", "aoi|place|municipality", "geometry|footprint", "activation_date|event_date", "vector|raster|footprint", "activation_mismatch|context_only", "v1uo_common_candidate_schema", "Operational mapping adapter requires event-specific activation."),
    ]
    rows = [dict(zip(ADAPTER_COLUMNS, row)) for row in adapters]
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_region_adapter_registry.csv")
    write_csv(out_path, ADAPTER_COLUMNS, rows)
    print(f"[v1uo adapters] rows={len(rows)} -> {out_path}")
    return rows


def adapter_for_region(region):
    if region == "REC":
        return "ADAPT_REC_CKAN"
    if region == "PET":
        return "ADAPT_PET_GEOHAZARD"
    if region == "CUR":
        return "ADAPT_CUR_PUBLIC_GEO"
    return "ADAPT_GENERIC_HYDROMET"


def run_multiregion_public_discovery_runner(out_path=None):
    events = load_csv(os.path.join(DATASET_DIR, "v1uo_multiregion_event_registry.csv"))
    sources = load_csv(os.path.join(DATASET_DIR, "v1uo_public_source_registry.csv"))
    rows = []
    seq = 0
    for event in events:
        region = event.get("region", "")
        relevant = [s for s in sources if s.get("region") in {region, "ALL"}]
        for src in relevant:
            rows.append({
                "discovery_id": f"DISC_v1uo_{seq:05d}",
                "event_id": event.get("event_id", ""),
                "region": region,
                "source_id": src.get("source_id", ""),
                "adapter_id": adapter_for_region(region),
                "candidate_url": src.get("base_url", ""),
                "http_status": "NOT_REQUESTED",
                "content_type": "",
                "discovery_status": "DRY_RUN_NOT_DOWNLOADED",
                "candidate_artifact_type": src.get("expected_artifact_types", ""),
                "event_specificity": "EVENT_SPECIFIC_POSSIBLE" if region != "CUR" else "EVENT_REGISTRY_REQUIRED",
                "priority": src.get("priority", ""),
                "blocking_reason": "allow_web_not_enabled_no_raw_download",
                "notes": "Dry-run public discovery registry only.",
            })
            seq += 1
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_multiregion_public_discovery_registry.csv")
    write_csv(out_path, DISCOVERY_COLUMNS, rows)
    print(f"[v1uo discovery] rows={len(rows)} -> {out_path}")
    return rows


def run_multiregion_schema_audit_runner(out_path=None):
    events = load_csv(os.path.join(DATASET_DIR, "v1uo_multiregion_event_registry.csv"))
    schema = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_asset_schema_registry.csv"))
    profile = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_occurrence_table_profile.csv"))
    hyd = hydromet_scorecards()
    rows = []
    for idx, event in enumerate(events):
        event_id = event.get("event_id", "")
        region = event.get("region", "")
        if event_id == "REC_2022_05_24_30":
            row_count = sum(int_value(r.get("total_rows")) for r in profile)
            rows.append({
                "schema_audit_id": f"SCHEMA_v1uo_{idx:04d}",
                "event_id": event_id,
                "region": region,
                "source_id": "SRC_REC_CKAN",
                "asset_count": str(len(schema)),
                "row_count": str(row_count),
                "has_date_field": "true",
                "has_hazard_field": "true",
                "has_locality_field": "true",
                "has_coordinate_fields": "false",
                "has_geometry": "false",
                "schema_status": "REUSED_V1UK_SCHEMA_AUDIT",
                "normalized_status": "LOCALITY_ONLY_SCHEMA_NORMALIZED",
                "main_blocker": "coordinate_support_blocked",
                "notes": "Recife v1uk/v1un reused without modifying upstream outputs.",
            })
        elif region == "PET":
            h = hyd.get(event_id, {})
            rows.append({
                "schema_audit_id": f"SCHEMA_v1uo_{idx:04d}",
                "event_id": event_id,
                "region": region,
                "source_id": "SRC_INMET",
                "asset_count": "1" if h else "0",
                "row_count": "0",
                "has_date_field": bool_text(bool(h)),
                "has_hazard_field": "false",
                "has_locality_field": "false",
                "has_coordinate_fields": bool_text(h.get("has_station_coordinates") == "true"),
                "has_geometry": "false",
                "schema_status": "REUSED_V1UF_HYDROMET_REGISTRY" if h else "NO_DOWNLOADED_ASSETS_YET",
                "normalized_status": "TEMPORAL_ANCHOR_ONLY",
                "main_blocker": "PHENOMENON_SEPARATION_REQUIRED",
                "notes": "Station coordinates are hydromet context, not observed event geometry.",
            })
        else:
            rows.append({
                "schema_audit_id": f"SCHEMA_v1uo_{idx:04d}",
                "event_id": event_id,
                "region": region,
                "source_id": "SRC_CUR_GEOCURITIBA",
                "asset_count": "0",
                "row_count": "0",
                "has_date_field": "false",
                "has_hazard_field": "false",
                "has_locality_field": "false",
                "has_coordinate_fields": "false",
                "has_geometry": "false",
                "schema_status": "NO_DOWNLOADED_ASSETS_YET",
                "normalized_status": "EVENT_REGISTRY_MISSING_FOR_CURITIBA",
                "main_blocker": "EVENT_REGISTRY_MISSING_FOR_CURITIBA",
                "notes": "Curitiba blocked until event registry and source discovery are deepened.",
            })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_multiregion_schema_audit_registry.csv")
    write_csv(out_path, SCHEMA_COLUMNS, rows)
    print(f"[v1uo schema] rows={len(rows)} -> {out_path}")
    return rows


def run_multiregion_candidate_router(out_path=None):
    events = load_csv(os.path.join(DATASET_DIR, "v1uo_multiregion_event_registry.csv"))
    rows = []
    for idx, e in enumerate(events):
        region = e.get("region", "")
        if e.get("event_id") == "REC_2022_05_24_30":
            cls = "locality-only candidate"
            strength = "STRONG_CONTEXTUAL_LOCALITY_ONLY"
            blocker = "coordinate_support_blocked"
            action = "RECIFE_COORDINATE_RECOVERY_FROM_PUBLIC_CKAN"
            loc = "true"; coord = geom = "false"
        elif e.get("event_id") == "PET_2024_03_21_28":
            cls = "temporal-anchor-only"
            strength = "STRONG_TEMPORAL_ANCHOR_GEOMETRY_MISSING"
            blocker = "PHENOMENON_SEPARATION_REQUIRED"
            action = "PETROPOLIS_PUBLIC_GEOMETRY_DEEPENING"
            loc = "false"; coord = "true"; geom = "false"
        elif e.get("event_id") == "PET_2022_02_15":
            cls = "temporal-anchor-only"
            strength = "MODERATE_DOCUMENTARY_HIGH_MIXED_PHENOMENON_RISK"
            blocker = "PHENOMENON_SEPARATION_REQUIRED"
            action = "PETROPOLIS_PHENOMENON_SEPARATION_AND_GEOMETRY_DEEPENING"
            loc = "false"; coord = "true"; geom = "false"
        elif region == "CUR":
            cls = "rejected"
            strength = "ABSENT_EVENT_REGISTRY"
            blocker = "EVENT_REGISTRY_MISSING_FOR_CURITIBA"
            action = "CURITIBA_EVENT_REGISTRY_AND_PUBLIC_SOURCE_DISCOVERY"
            loc = coord = geom = "false"
        else:
            cls = "context-only"
            strength = "WEAK"
            blocker = "region_adapter_missing"
            action = "REGION_SPECIFIC_PUBLIC_SOURCE_DEEPENING"
            loc = coord = geom = "false"
        rows.append({
            "route_id": f"ROUTE_v1uo_{idx:04d}",
            "event_id": e.get("event_id", ""),
            "region": region,
            "candidate_class": cls,
            "evidence_strength": strength,
            "has_temporal_support": e.get("has_temporal_anchor", "false"),
            "has_official_source": e.get("has_official_source", "false"),
            "has_locality_support": loc,
            "has_coordinate_support": coord,
            "has_geometry_support": geom,
            "can_advance_to_overlay_preflight": "false",
            "can_advance_to_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": blocker,
            "required_next_action": action,
            "notes": "Generic v1uo router; no ground reference, no overlay, no label.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_multiregion_candidate_router.csv")
    write_csv(out_path, ROUTER_COLUMNS, rows)
    print(f"[v1uo router] rows={len(rows)} -> {out_path}")
    return rows


def dim_classifications(route):
    event_id = route.get("event_id", "")
    region = route.get("region", "")
    if event_id == "REC_2022_05_24_30":
        return {
            "temporal_support": "STRONG",
            "official_source_support": "STRONG",
            "hazard_typing_support": "MODERATE",
            "phenomenon_separation": "MODERATE",
            "locality_support": "STRONG",
            "coordinate_support": "BLOCKED",
            "geometry_support": "BLOCKED",
            "overlay_readiness": "BLOCKED",
            "ground_reference_readiness": "BLOCKED",
            "replication_readiness": "STRONG",
        }
    if event_id == "PET_2024_03_21_28":
        return {
            "temporal_support": "STRONG",
            "official_source_support": "STRONG",
            "hazard_typing_support": "MODERATE",
            "phenomenon_separation": "BLOCKED",
            "locality_support": "WEAK",
            "coordinate_support": "MODERATE",
            "geometry_support": "BLOCKED",
            "overlay_readiness": "BLOCKED",
            "ground_reference_readiness": "BLOCKED",
            "replication_readiness": "STRONG",
        }
    if event_id == "PET_2022_02_15":
        return {
            "temporal_support": "STRONG",
            "official_source_support": "STRONG",
            "hazard_typing_support": "WEAK",
            "phenomenon_separation": "BLOCKED",
            "locality_support": "WEAK",
            "coordinate_support": "MODERATE",
            "geometry_support": "BLOCKED",
            "overlay_readiness": "BLOCKED",
            "ground_reference_readiness": "BLOCKED",
            "replication_readiness": "MODERATE",
        }
    if region == "CUR":
        return {d: ("BLOCKED" if d in {"overlay_readiness", "ground_reference_readiness"} else "ABSENT") for d in [
            "temporal_support", "official_source_support", "hazard_typing_support",
            "phenomenon_separation", "locality_support", "coordinate_support",
            "geometry_support", "overlay_readiness", "ground_reference_readiness",
            "replication_readiness",
        ]}
    return {}


def run_multiregion_evidence_matrix_builder(out_path=None):
    routes = load_csv(os.path.join(DATASET_DIR, "v1uo_multiregion_candidate_router.csv"))
    rows = []
    seq = 0
    for route in routes:
        for dim, cls in dim_classifications(route).items():
            rows.append({
                "matrix_id": f"MATRIX_v1uo_{seq:05d}",
                "event_id": route.get("event_id", ""),
                "region": route.get("region", ""),
                "dimension": dim,
                "classification": cls,
                "basis": route.get("evidence_strength", ""),
                "ground_truth_operational": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "notes": "Multiregion comparative evidence dimension.",
            })
            seq += 1
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_multiregion_evidence_matrix.csv")
    write_csv(out_path, MATRIX_COLUMNS, rows)
    print(f"[v1uo matrix] rows={len(rows)} -> {out_path}")
    return rows


def score_route(route):
    event_id = route.get("event_id", "")
    if event_id == "PET_2024_03_21_28":
        return 95, "HIGH_PUBLIC_GEOMETRY_DEEPENING_OPPORTUNITY", "HIGH", "MEDIUM"
    if event_id == "PET_2022_02_15":
        return 80, "HIGH_DOCUMENTARY_VALUE_HIGH_PHENOMENON_RISK", "MODERATE", "HIGH"
    if event_id == "REC_2022_05_24_30":
        return 70, "HIGH_REPLICATION_VALUE_LIMITED_GEOMETRY", "LIMITED", "MEDIUM"
    if route.get("region") == "CUR":
        return 60, "EVENT_REGISTRY_DISCOVERY_REQUIRED", "UNKNOWN", "MEDIUM"
    return 10, "LOW_INFORMATION", "UNKNOWN", "HIGH"


def run_ground_truth_opportunity_ranker(out_path=None):
    routes = load_csv(os.path.join(DATASET_DIR, "v1uo_multiregion_candidate_router.csv"))
    scored = []
    for route in routes:
        score, opp, gt_value, risk = score_route(route)
        scored.append((score, route, opp, gt_value, risk))
    scored.sort(key=lambda item: (-item[0], item[1].get("event_id", "")))
    rows = []
    for rank, (score, route, opp, gt_value, risk) in enumerate(scored, start=1):
        if route.get("event_id") == "PET_2024_03_21_28":
            next_action = "Petrópolis Public Geometry Deepening"
            next_version = "v1up"
        elif route.get("region") == "CUR":
            next_action = "Curitiba Event Registry and Public Source Discovery"
            next_version = "v1up"
        elif route.get("event_id") == "REC_2022_05_24_30":
            next_action = "Recife Coordinate Recovery from Public CKAN"
            next_version = "v1up"
        elif route.get("event_id") == "PET_2022_02_15":
            next_action = "Petrópolis Public Geometry Deepening"
            next_version = "v1up"
        else:
            next_action = "Event-Patch Package Linkage Engine"
            next_version = "v1up"
        rows.append({
            "rank": str(rank),
            "event_id": route.get("event_id", ""),
            "region": route.get("region", ""),
            "opportunity_class": opp,
            "current_strength": route.get("evidence_strength", ""),
            "main_blocker": route.get("blocker", ""),
            "expected_programming_value": str(score),
            "expected_ground_truth_value": gt_value,
            "overclaim_risk": risk,
            "recommended_next_version": next_version,
            "recommended_next_action": next_action,
            "notes": "Ranked for programming value; no operational promotion in v1uo.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_ground_truth_opportunity_ranker.csv")
    write_csv(out_path, RANK_COLUMNS, rows)
    print(f"[v1uo ranker] rows={len(rows)} -> {out_path}")
    return rows


def run_event_patch_package_prebuilder(out_path=None):
    routes = load_csv(os.path.join(DATASET_DIR, "v1uo_multiregion_candidate_router.csv"))
    rows = []
    for idx, route in enumerate(routes):
        has_temporal = route.get("has_temporal_support") == "true"
        rows.append({
            "package_id": f"PKG_v1uo_{idx:04d}",
            "event_id": route.get("event_id", ""),
            "region": route.get("region", ""),
            "patch_id": "PATCH_LINKAGE_MISSING",
            "sentinel_scene_date": "PATCH_LINKAGE_MISSING",
            "evidence_sources_attached": "v1uo_registry_only",
            "temporal_anchor_status": "PRESENT" if has_temporal else "MISSING",
            "hydromet_status": "PRESENT" if route.get("region") in {"PET", "REC"} else "UNKNOWN",
            "locality_status": "PRESENT" if route.get("has_locality_support") == "true" else "MISSING_OR_NOT_ASSESSED",
            "coordinate_status": "CONTEXT_COORDINATE_ONLY" if route.get("has_coordinate_support") == "true" else "MISSING",
            "geometry_status": "MISSING",
            "dino_review_support_status": "SUPPORT_ONLY_NOT_RUN",
            "overlay_status": "NOT_EXECUTED",
            "ground_reference_status": "NOT_CREATED",
            "package_status": "PREBUILD_ONLY_PATCH_LINKAGE_MISSING",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Event-patch package prebuild only; no patch truth inferred.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uo_event_patch_package_prebuild_registry.csv")
    write_csv(out_path, PACKAGE_COLUMNS, rows)
    print(f"[v1uo package prebuild] rows={len(rows)} -> {out_path}")
    return rows


def write_policy_configs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    policies = {
        "v1uo_multiregion_events.yaml": [
            "protocol_version: v1uo",
            "regions: [REC, PET, CUR]",
            "status_max: MULTI_REGION_EVIDENCE_CANDIDATE_DISCOVERY_NON_OPERATIONAL",
        ],
        "v1uo_multiregion_public_sources.yaml": [
            "protocol_version: v1uo",
            "download_raw_data_by_default: false",
            "discovery_mode: dry_run_registry",
        ],
        "v1uo_region_adapter_policy.yaml": [
            "protocol_version: v1uo",
            "adapter_schema: v1uo_common_candidate_schema",
            "city_specific_code_duplication_allowed: false",
        ],
        "v1uo_schema_audit_policy.yaml": [
            "protocol_version: v1uo",
            "reuse_existing_registries: true",
            "heavy_download_allowed: false",
        ],
        "v1uo_candidate_routing_policy.yaml": [
            "protocol_version: v1uo",
            "can_create_ground_reference: false",
            "can_create_training_label: false",
            "overlay_execution_allowed: false",
        ],
        "v1uo_ground_truth_opportunity_policy.yaml": [
            "protocol_version: v1uo",
            "ranking_basis: programming_value_without_operational_promotion",
            "overclaim_risk_required: true",
        ],
        "v1uo_event_patch_package_policy.yaml": [
            "protocol_version: v1uo",
            "prebuild_only: true",
            "invent_patch_id_allowed: false",
            "invent_scene_date_allowed: false",
        ],
    }
    for name, lines in policies.items():
        with open(os.path.join(CONFIG_DIR, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def run_completion_report():
    write_policy_configs()
    events = load_csv(os.path.join(DATASET_DIR, "v1uo_multiregion_event_registry.csv"))
    matrix = load_csv(os.path.join(DATASET_DIR, "v1uo_multiregion_evidence_matrix.csv"))
    ranks = load_csv(os.path.join(DATASET_DIR, "v1uo_ground_truth_opportunity_ranker.csv"))
    packages = load_csv(os.path.join(DATASET_DIR, "v1uo_event_patch_package_prebuild_registry.csv"))
    top = ranks[0] if ranks else {}
    next_action = top.get("recommended_next_action", "Event-Patch Package Linkage Engine")
    action_rows = [{
        "action_id": "ACT_v1uo_0000",
        "event_id": top.get("event_id", ""),
        "action_type": "PROGRAMMING_DEEPENING",
        "priority": "1",
        "description": f"v1up - {next_action}",
        "target": top.get("region", ""),
        "status": "PENDING",
        "notes": "Selected by v1uo ranker; no v1up implementation in this stage.",
    }]
    write_csv(os.path.join(DATASET_DIR, "v1uo_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, action_rows)
    manifest = []
    for idx, path in enumerate(V1UO_ARTIFACTS):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_v1uo_{idx:04d}",
            "artifact_path": artifact_path(path),
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": bool_text(exists),
            "reason": "Safe v1uo engineering artifact" if exists else "File not found",
        })
    write_csv(os.path.join(DATASET_DIR, "v1uo_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    os.makedirs(DOCS_DIR, exist_ok=True)
    region_status = {r.get("region"): r.get("current_best_evidence_status") for r in events}
    blockers = {r.get("region"): r.get("main_blocker") for r in events}
    method = [
        "# Protocolo C v1uo - Multi-Region Replication Engine",
        "",
        "## Engineering Scope",
        "- Generalizes the Recife registry-driven chain into reusable multiregion components.",
        "- Runs as dry-run public discovery and registry normalization by default.",
        "- Does not download raw artifacts, execute overlay, infer coordinates, or create labels.",
        "",
        "## Components",
        "- multiregion event registry",
        "- public source registry",
        "- region adapter registry",
        "- dry-run public discovery registry",
        "- generic schema audit registry",
        "- generic candidate router",
        "- multiregion evidence matrix",
        "- opportunity ranker",
        "- event-patch package prebuilder",
    ]
    report = [
        "# Relatorio tecnico v1uo - Multi-Region Replication Engine",
        "",
        f"- events_registered: {len(events)}",
        f"- evidence_matrix_rows: {len(matrix)}",
        f"- event_patch_packages_prebuilt: {len(packages)}",
        f"- Recife status: {region_status.get('REC', '')}",
        f"- Petropolis status: {region_status.get('PET', '')}",
        f"- Curitiba status: {region_status.get('CUR', '')}",
        f"- top_opportunity: {top.get('event_id', '')} / {top.get('recommended_next_action', '')}",
        "",
        "## Region Blockers",
        f"- REC: {blockers.get('REC', '')}",
        f"- PET: {blockers.get('PET', '')}",
        f"- CUR: {blockers.get('CUR', '')}",
        "",
        "## Guardrails",
        "- ground_truth_operational=false",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
        "- no_overlay_executed=true",
        "- no_coordinates_invented=true",
        "- patch_bound_truth=false",
        "- operational_validation=false",
    ]
    status = [
        "# Status Atual - Protocolo C v1uo",
        "",
        f"status_max={MAX_STATUS}",
        f"events_registered={len(events)}",
        f"top_ranked_event={top.get('event_id', '')}",
        f"top_ranked_region={top.get('region', '')}",
        f"recommended_next_action=v1up - {next_action}",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "can_reopen_protocol_b=false",
        "dino_usage=SUPPORT_ONLY",
        "no_overlay_executed=true",
        "no_coordinates_invented=true",
        "patch_bound_truth=false",
        "operational_validation=false",
    ]
    files = {
        "protocolo_c_v1uo_multiregion_replication_engine.md": method,
        "protocolo_c_relatorio_v1uo_multiregion_replication_engine.md": report,
        "protocolo_c_status_atual_v1uo.md": status,
    }
    for name, lines in files.items():
        with open(os.path.join(DOCS_DIR, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(f"[v1uo completion] next_action=v1up - {next_action}")
    return {
        "events_registered": len(events),
        "top_ranked_event": top.get("event_id", ""),
        "recommended_next_action": f"v1up - {next_action}",
        "packages": len(packages),
    }


def simple_main(fn):
    parser = argparse.ArgumentParser()
    parser.parse_args()
    fn()

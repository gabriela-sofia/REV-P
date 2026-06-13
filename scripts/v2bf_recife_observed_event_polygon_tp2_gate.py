#!/usr/bin/env python3
"""Retrieve and classify public Recife May 2022 event-polygon evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "v2bf_recife_observed_event_polygon_tp2_gate_config.json"
STAGE = "v2bf_recife_observed_event_polygon_tp2_gate"
EVENT = "REC_2022_05_24_30"
MODES = ("search_plan", "download_public", "scan_sources", "classify_sources", "extract_polygon",
         "build_feeds", "tp2_gate", "full")
RETRIEVED_AT = "2026-06-13"

SEARCH_COLUMNS = "search_id target_event_id target_package_id source_name source_category search_query expected_artifact expected_geometry_type expected_file_format source_public download_allowed must_attempt_download result_url local_raw_path current_status blocking_reason notes".split()
DOWNLOAD_COLUMNS = "download_id search_id target_event_id source_name url attempted success http_status_or_error content_type file_size_bytes local_raw_path hash_sha256 retrieved_at blocking_reason notes".split()
INVENTORY_COLUMNS = "source_inventory_id target_event_id source_name source_category source_url_or_reference local_path file_name file_extension file_size_bytes hash_sha256 detected_format source_public access_status can_parse contains_geometry contains_polygon contains_point contains_context_only blocking_reason notes".split()
CLASS_COLUMNS = "classification_id target_event_id source_inventory_id source_name geometry_id geometry_type geometry_role crs crs_status is_point is_polygon_or_bbox is_observed_event_specific is_contextual is_risk_or_susceptibility is_visual_only can_be_tp2_candidate requires_human_review blocking_reason notes".split()
CANDIDATE_COLUMNS = "candidate_event_polygon_id event_id package_id patch_id source_name source_url_or_reference source_file local_derived_path geometry_format geometry_type crs crs_status geometry_valid area_m2_approx bbox_minx bbox_miny bbox_maxx bbox_maxy vertex_count geometry_hash is_observed_event_specific requires_human_review can_feed_v2ba can_feed_v2aw can_feed_v2au can_feed_v2az can_support_tp2 blocking_reason notes".split()
FEED_COLUMNS = "feed_id event_id patch_id package_id geometry_path geometry_format crs geometry_hash source_stage source_method source_document source_public access_status review_status requires_human_review ready blocking_reason notes".split()
GATE_COLUMNS = "gate_id turning_point_level gate_name required_condition observed_value gate_passed severity blocking_reason recommended_action notes".split()
PRECHECK_COLUMNS = "precheck_id package_id patch_id event_id tp1_patch_boundary_available tp2_event_polygon_available same_package ready_for_pair_overlay_test can_attempt_v2az_dry_run can_attempt_v2az_replay blocking_reason notes".split()
REJECTION_COLUMNS = "rejection_id source_name source_file geometry_or_evidence_type reason_not_tp2 allowed_use not_allowed_use can_support_manual_digitization can_feed_pipeline blocking_reason notes".split()
MANUAL_COLUMNS = "task_id target_event_id target_package_id input_support_layers output_required output_path_expected required_crs required_format step_by_step validation_command blocking_reason notes".split()

TABLES = {
    "v2bf_tp2_public_event_polygon_search_plan.csv": SEARCH_COLUMNS,
    "v2bf_tp2_public_download_attempts.csv": DOWNLOAD_COLUMNS,
    "v2bf_tp2_event_source_inventory.csv": INVENTORY_COLUMNS,
    "v2bf_tp2_event_geometry_classification.csv": CLASS_COLUMNS,
    "v2bf_REC_2022_05_24_30_observed_event_polygon_candidate_registry.csv": CANDIDATE_COLUMNS,
    "v2bf_ready_event_polygon_feed_for_v2ba.csv": FEED_COLUMNS,
    "v2bf_ready_event_polygon_feed_for_v2aw.csv": FEED_COLUMNS,
    "v2bf_ready_event_polygon_feed_for_v2au.csv": FEED_COLUMNS,
    "v2bf_ready_event_polygon_feed_for_v2az.csv": FEED_COLUMNS,
    "v2bf_tp2_readiness_gate.csv": GATE_COLUMNS,
    "v2bf_tp3_pair_precheck.csv": PRECHECK_COLUMNS,
    "v2bf_context_rejection_audit.csv": REJECTION_COLUMNS,
    "v2bf_tp2_manual_digitization_update_plan.csv": MANUAL_COLUMNS,
}

DIRECT = [
    ("Charter Activation 758 page", "charter_product_catalog", "Charter Activation 758 products and vector download",
     "https://disasterscharter.org/activations/landslide-in-brazil-activation-758-", "product catalog/page", "html", "charter_activation_758_page.html"),
    ("Charter 758 Recife quickview", "visual_support", "Charter 758 Recife georeferenced or vector product",
     "https://disasterscharter.org/cos-api/api/file/public/article-image/28495218", "quickview image", "png", "charter_758_recife_quickview.png"),
    ("Recife Defesa Civil risk locations GeoJSON", "risk_context", "Recife May 2022 affected area GeoJSON",
     "https://dados.recife.pe.gov.br/dataset/c1d733d9-5867-481e-9c18-5fe572300ab2/resource/ec18759d-fac2-445e-ae72-af9d9210b831/download/coordenadas-geograficas-da-regiao-sul.geojson", "risk location points", "geojson", "recife_defesa_civil_risk_locations.geojson"),
    ("Recife Defesa Civil risk locations CSV", "risk_context", "Recife May 2022 affected area CSV WKT",
     "https://dados.recife.pe.gov.br/dataset/c1d733d9-5867-481e-9c18-5fe572300ab2/resource/75344435-aca8-4aef-ab2c-9521d6e5ff18/download/areas-de-risco-da-regional-sul.csv", "risk location table", "csv", "recife_defesa_civil_risk_locations.csv"),
    ("SGB Recife May 2022 precipitation study", "precipitation_context", "SGB CPRM Recife May 2022 event polygon",
     "https://rigeo.sgb.gov.br/server/api/core/bitstreams/5163f1c7-0905-40ba-9fba-f14a3173eac1/content", "temporal/document support", "txt", "sgb_recife_may_2022_precipitation_study.txt"),
]
MANUAL_SEARCHES = [
    ("Charter Activation 758 vector product", "event_polygon_search", "Charter Activation 758 Recife vector shapefile"),
    ("Charter Activation 758 georeferenced product", "manual_digitization_support", "Charter Activation 758 Recife georeferenced map"),
    ("Copernicus EMS Recife May 2022", "event_polygon_search", "Copernicus EMS Recife May 2022 mapping product"),
    ("Recife May 2022 flood landslide vector", "event_polygon_search", "Recife May 2022 flood landslide vector"),
    ("Recife May 2022 flood landslide GeoJSON", "event_polygon_search", "Recife May 2022 flood landslide GeoJSON"),
    ("Recife May 2022 disaster shapefile", "event_polygon_search", "Recife May 2022 disaster shapefile"),
    ("SGB CPRM Recife May 2022 event polygon", "event_polygon_search", "SGB CPRM Recife May 2022 event polygon"),
    ("Defesa Civil Recife affected area", "event_polygon_search", "Defesa Civil Recife maio 2022 area atingida"),
    ("Defesa Civil Pernambuco affected area", "event_polygon_search", "Defesa Civil Pernambuco maio 2022 area atingida"),
    ("Prefeitura Recife geodata event", "event_polygon_search", "Prefeitura Recife geodados evento maio 2022"),
    ("CEMADEN Recife May 2022", "temporal_context", "CEMADEN evento Recife maio 2022"),
    ("ANA INMET Cemaden temporal context", "temporal_context", "ANA INMET Cemaden Recife maio 2022"),
]


def clean(value):
    return str(value or "").strip()


def b(value):
    return "true" if value else "false"


def stable_id(prefix, *parts):
    return f"{prefix}_{hashlib.sha256('|'.join(clean(x) for x in parts).encode()).hexdigest()[:12]}"


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def load_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_dirs(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dataset = Path(dataset_dir or os.getenv("DATASET_DIR") or ROOT / "datasets").resolve()
    external = Path(external_dir or os.getenv("EXTERNAL_DIR") or dataset / "external_sources" / "recife_minimal_tp").resolve()
    return {"dataset": dataset, "output": Path(output_dir or os.getenv("OUTPUT_DIR") or ROOT / "outputs_public").resolve(),
            "config": Path(config_dir or os.getenv("CONFIG_DIR") or ROOT / "configs").resolve(),
            "external": external, "docs": Path(docs_dir or ROOT / "docs").resolve()}


def build_search_plan(config):
    rows = []
    for name, category, query, url, artifact, fmt, filename in DIRECT:
        rows.append({"search_id": stable_id("V2BF_SEARCH", name), "target_event_id": config["priority_event_id"],
            "target_package_id": config["priority_package_id"], "source_name": name, "source_category": category,
            "search_query": query, "expected_artifact": artifact, "expected_geometry_type": "observed polygon" if category == "event_polygon_search" else "support/context",
            "expected_file_format": fmt, "source_public": "true", "download_allowed": "true",
            "must_attempt_download": "true", "result_url": url, "local_raw_path": filename,
            "current_status": "DIRECT_PUBLIC_URL_IDENTIFIED", "blocking_reason": "",
            "notes": "Direct public URL will be downloaded and classified semantically."})
    for name, category, query in MANUAL_SEARCHES:
        rows.append({"search_id": stable_id("V2BF_SEARCH", name), "target_event_id": config["priority_event_id"],
            "target_package_id": config["priority_package_id"], "source_name": name, "source_category": category,
            "search_query": query, "expected_artifact": "event-specific vector or verified geospatial support",
            "expected_geometry_type": "observed polygon", "expected_file_format": "geojson|kml|kmz|shp_zip|csv_wkt|csv_bbox",
            "source_public": "true", "download_allowed": "true", "must_attempt_download": "false", "result_url": "",
            "local_raw_path": "", "current_status": "SEARCH_COMPLETED_NO_DIRECT_VECTOR_URL",
            "blocking_reason": "NO_DIRECT_PUBLIC_VECTOR_URL_IDENTIFIED",
            "notes": "Targeted public search completed; manual access/contact may still be required."})
    return rows


def download(plan, raw_dir, allow=True):
    attempts = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for row in plan:
        if row["must_attempt_download"] != "true":
            continue
        path = raw_dir / row["local_raw_path"]
        error, content_type, status = "", "", ""
        if not path.is_file() and allow:
            try:
                request = urllib.request.Request(row["result_url"], headers={"User-Agent": "REV-P-v2bf/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    content = response.read(50 * 1024 * 1024)
                    path.write_bytes(content)
                    status, content_type = str(response.status), clean(response.headers.get("Content-Type"))
            except (OSError, urllib.error.URLError) as exc:
                error = type(exc).__name__ + ":" + clean(exc)
        if path.is_file():
            content = path.read_bytes()
            success, digest = True, hashlib.sha256(content).hexdigest()
            status = status or "LOCAL_PUBLIC_FILE_AVAILABLE"
        else:
            success, digest, content = False, "", b""
            status = status or error or "DOWNLOAD_NOT_AVAILABLE"
        attempts.append({"download_id": stable_id("V2BF_DL", row["search_id"], row["result_url"]),
            "search_id": row["search_id"], "target_event_id": row["target_event_id"], "source_name": row["source_name"],
            "url": row["result_url"], "attempted": "true", "success": b(success), "http_status_or_error": status,
            "content_type": content_type, "file_size_bytes": str(len(content)), "local_raw_path": path.as_posix(),
            "hash_sha256": digest, "retrieved_at": RETRIEVED_AT,
            "blocking_reason": "" if success else "PUBLIC_DOWNLOAD_FAILED",
            "notes": "Public source available locally and still requires semantic classification."})
    return attempts


def detect_format(path):
    suffix = path.suffix.lower()
    return {".geojson": "geojson", ".json": "json", ".csv": "csv", ".png": "png", ".html": "html",
            ".txt": "txt", ".kml": "kml", ".kmz": "kmz", ".zip": "shp_zip"}.get(suffix, suffix.lstrip("."))


def inspect_geojson(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return 0, 0, []
    features = obj.get("features", []) if obj.get("type") == "FeatureCollection" else [obj] if obj.get("type") == "Feature" else []
    types = [clean((feature.get("geometry") or {}).get("type")) for feature in features]
    return sum(t in ("Polygon", "MultiPolygon") for t in types), sum(t == "Point" for t in types), features


def inventory(plan, attempts):
    by_search = {row["search_id"]: row for row in plan}
    rows = []
    for attempt in attempts:
        if attempt["success"] != "true":
            continue
        source = by_search[attempt["search_id"]]
        path = Path(attempt["local_raw_path"])
        fmt = detect_format(path)
        polygons, points, _ = inspect_geojson(path) if fmt in ("geojson", "json") else (0, 0, [])
        context = source["source_category"] != "event_polygon_search"
        rows.append({"source_inventory_id": stable_id("V2BF_INV", attempt["download_id"]), "target_event_id": EVENT,
            "source_name": source["source_name"], "source_category": source["source_category"],
            "source_url_or_reference": source["result_url"], "local_path": path.as_posix(), "file_name": path.name,
            "file_extension": path.suffix.lower(), "file_size_bytes": attempt["file_size_bytes"],
            "hash_sha256": attempt["hash_sha256"], "detected_format": fmt, "source_public": "true",
            "access_status": "public_or_project_access", "can_parse": b(fmt in ("geojson", "json", "csv", "txt", "html")),
            "contains_geometry": b(bool(polygons or points)), "contains_polygon": b(bool(polygons)), "contains_point": b(bool(points)),
            "contains_context_only": b(context), "blocking_reason": "SOURCE_CONTEXT_NOT_OBSERVED_EVENT_POLYGON" if context else "",
            "notes": "Inventory does not authorize TP2; event-specific semantics are mandatory."})
    return rows


def classification(inventory_rows):
    rows = []
    for source in inventory_rows:
        category, fmt, path = source["source_category"], source["detected_format"], Path(source["local_path"])
        polygons, points, features = inspect_geojson(path) if fmt in ("geojson", "json") else (0, 0, [])
        if category == "risk_context":
            role, gtype, blocker = "context_risk_location_point", "Point" if points else "context_table", "RISK_CONTEXT_NOT_OBSERVED_EVENT"
        elif category == "visual_support":
            role, gtype, blocker = "visual_support_only", "image", "QUICKVIEW_NOT_GEOREFERENCED_VERIFIED_PRODUCT"
        elif category == "precipitation_context":
            role, gtype, blocker = "precipitation_context", "document", "PRECIPITATION_NOT_EVENT_POLYGON"
        elif category == "charter_product_catalog":
            role, gtype, blocker = "visual_support_only", "html_catalog", "CATALOG_HAS_NO_DIRECT_EVENT_VECTOR"
        else:
            role, gtype, blocker = "unsupported", "unknown", "NO_EVENT_SPECIFIC_POLYGON"
        count = max(1, len(features))
        for index in range(count):
            feature_type = clean((features[index].get("geometry") or {}).get("type")) if features else gtype
            is_point = feature_type == "Point"
            is_polygon = feature_type in ("Polygon", "MultiPolygon")
            rows.append({"classification_id": stable_id("V2BF_CLASS", source["source_inventory_id"], index),
                "target_event_id": EVENT, "source_inventory_id": source["source_inventory_id"], "source_name": source["source_name"],
                "geometry_id": stable_id("V2BF_GEOM", source["source_inventory_id"], index), "geometry_type": feature_type or gtype,
                "geometry_role": role, "crs": "EPSG:4326" if fmt == "geojson" else "UNKNOWN",
                "crs_status": "KNOWN_CONTEXT_CRS" if fmt == "geojson" else "UNKNOWN_OR_NOT_APPLICABLE",
                "is_point": b(is_point), "is_polygon_or_bbox": b(is_polygon), "is_observed_event_specific": "false",
                "is_contextual": "true", "is_risk_or_susceptibility": b(category == "risk_context"),
                "is_visual_only": b(role == "visual_support_only"), "can_be_tp2_candidate": "false",
                "requires_human_review": "true", "blocking_reason": blocker,
                "notes": "Context and support evidence never become an observed-event polygon automatically."})
    return rows


def geometry_metrics(geometry):
    coords = geometry.get("coordinates", [])
    rings = coords if geometry.get("type") == "Polygon" else [ring for polygon in coords for ring in polygon] if geometry.get("type") == "MultiPolygon" else []
    points = [point for ring in rings for point in ring]
    if not points:
        return [], 0, 0.0
    bbox = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
    lat = sum(p[1] for p in points) / len(points)
    radius = 6371008.8
    area = 0.0
    for ring in rings:
        xy = [(math.radians(p[0]) * radius * math.cos(math.radians(lat)), math.radians(p[1]) * radius) for p in ring]
        area += abs(sum(a[0] * b_[1] - b_[0] * a[1] for a, b_ in zip(xy, xy[1:]))) / 2
    return bbox, sum(max(0, len(ring) - 1) for ring in rings), area


def candidate_from_feature(feature, source, config, derived_path=""):
    """Accept only an explicitly event-specific Polygon/MultiPolygon with known CRS."""
    geometry = feature.get("geometry", feature)
    props = feature.get("properties", {}) if feature.get("type") == "Feature" else {}
    gtype = geometry.get("type")
    event_specific = clean(props.get("event_id")) == config["priority_event_id"] and clean(props.get("geometry_role")) == "observed_event_polygon"
    crs = clean(props.get("crs") or source.get("crs") or "EPSG:4326")
    valid = gtype in ("Polygon", "MultiPolygon") and event_specific and crs in config["accepted_crs"]
    bbox, vertices, area = geometry_metrics(geometry) if valid else ([], 0, 0.0)
    digest = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest() if valid else ""
    return {"candidate_event_polygon_id": stable_id("V2BF_CAND", config["priority_event_id"], digest),
        "event_id": config["priority_event_id"], "package_id": config["priority_package_id"], "patch_id": config["priority_patch_id"],
        "source_name": clean(source.get("source_name")), "source_url_or_reference": clean(source.get("source_url_or_reference")),
        "source_file": clean(source.get("local_path")), "local_derived_path": derived_path, "geometry_format": "geojson_file",
        "geometry_type": gtype or "UNKNOWN", "crs": crs, "crs_status": "ACCEPTED" if crs in config["accepted_crs"] else "UNKNOWN",
        "geometry_valid": b(valid), "area_m2_approx": f"{area:.2f}" if valid else "",
        "bbox_minx": bbox[0] if bbox else "", "bbox_miny": bbox[1] if bbox else "", "bbox_maxx": bbox[2] if bbox else "",
        "bbox_maxy": bbox[3] if bbox else "", "vertex_count": str(vertices), "geometry_hash": digest,
        "is_observed_event_specific": b(event_specific), "requires_human_review": "true",
        "can_feed_v2ba": b(valid), "can_feed_v2aw": b(valid), "can_feed_v2au": b(valid), "can_feed_v2az": b(valid),
        "can_support_tp2": b(valid), "blocking_reason": "" if valid else "NO_VALID_EVENT_SPECIFIC_POLYGON",
        "notes": "TP2 candidate only; human review required; never final ground truth."}


def extract_candidates(inventory_rows, config, derived_dir):
    candidates = []
    for source in inventory_rows:
        if source["source_category"] != "event_polygon_search" or source["detected_format"] not in ("geojson", "json"):
            continue
        _, _, features = inspect_geojson(Path(source["local_path"]))
        for feature in features:
            candidate = candidate_from_feature(feature, source, config)
            if candidate["geometry_valid"] != "true":
                continue
            derived = derived_dir / "event_polygon_REC_2022_05_24_30_from_public_source.geojson"
            write_text(derived, json.dumps(feature, indent=2))
            candidate["local_derived_path"] = derived.as_posix()
            candidates.append(candidate)
    return candidates


def build_feeds(candidates, config):
    rows = []
    for candidate in candidates:
        if candidate["geometry_valid"] != "true":
            continue
        rows.append({"feed_id": stable_id("V2BF_FEED", candidate["geometry_hash"]), "event_id": config["priority_event_id"],
            "patch_id": config["priority_patch_id"], "package_id": config["priority_package_id"],
            "geometry_path": candidate["local_derived_path"], "geometry_format": "geojson_file", "crs": candidate["crs"],
            "geometry_hash": candidate["geometry_hash"], "source_stage": "v2bf", "source_method": "public_event_specific_polygon",
            "source_document": candidate["source_url_or_reference"], "source_public": "true",
            "access_status": "public_or_project_access", "review_status": "provided_unreviewed",
            "requires_human_review": "true", "ready": "true", "blocking_reason": "",
            "notes": "Observed-event polygon candidate only; no operational label."})
    return rows


def build_gates(plan, attempts, candidates, feeds, context_count):
    found = bool(candidates)
    geometry_checks = [
        ("TP2_03_EVENT_POLYGON_SOURCE_FOUND", found), ("TP2_04_EVENT_POLYGON_GEOMETRY_VALID", found),
        ("TP2_05_EVENT_POLYGON_CRS_RECORDED", found), ("TP2_06_EVENT_POLYGON_PROVENANCE_RECORDED", found),
        ("TP2_07_EVENT_POLYGON_FEED_READY_FOR_V2BA", bool(feeds)), ("TP2_08_EVENT_POLYGON_FEED_READY_FOR_V2AW", bool(feeds)),
        ("TP2_09_EVENT_POLYGON_FEED_READY_FOR_V2AU", bool(feeds)), ("TP2_10_EVENT_POLYGON_FEED_READY_FOR_V2AZ", bool(feeds)),
    ]
    specs = [("TP2_01_EVENT_SOURCE_SEARCH_COMPLETED", bool(plan)), ("TP2_02_EVENT_SOURCE_DOWNLOADS_ATTEMPTED", bool(attempts))]
    specs += geometry_checks
    specs += [("TP2_11_HUMAN_REVIEW_REQUIRED", True), ("TP2_12_CONTEXT_NOT_PROMOTED", context_count > 0),
              ("TP2_13_NO_LABEL_CREATED", True), ("TP2_14_NO_MODEL_TRAINED", True)]
    return [{"gate_id": stable_id("V2BF_GATE", name), "turning_point_level": "TP2_ONE_EVENT_POLYGON_CANDIDATE_REQUIRES_HUMAN_REVIEW",
        "gate_name": name, "required_condition": name.lower(), "observed_value": f"passed={str(passed).lower()}",
        "gate_passed": b(passed), "severity": "safety" if name >= "TP2_11" else "blocking",
        "blocking_reason": "" if passed else "NO_VALID_OBSERVED_EVENT_POLYGON",
        "recommended_action": "Preserve guardrail" if passed else "Acquire or digitize a verified event-specific polygon",
        "notes": "Context cannot satisfy TP2."} for name, passed in specs]


def rejection_audit():
    specs = [
        ("400 Recife risk locations", "recife_defesa_civil_risk_locations.geojson", "400 Point features", "Risk locations are context, not observed impact extent", "manual digitization support", "event polygon|TP2", True, "POINT_CONTEXT_NOT_OBSERVED_EVENT"),
        ("Charter 758 Recife quickview", "charter_758_recife_quickview.png", "PNG visual support", "Quickview is not a verified georeferenced vector", "visual/manual support", "automatic event polygon", True, "QUICKVIEW_NOT_GEOREFERENCED_VERIFIED_PRODUCT"),
        ("SGB Recife May 2022 precipitation study", "sgb_recife_may_2022_precipitation_study.txt", "text/precipitation context", "Precipitation documents timing, not impact polygon", "temporal support", "event polygon|TP2", True, "PRECIPITATION_NOT_EVENT_POLYGON"),
        ("Charter Activation 758 page", "charter_activation_758_page.html", "HTML product catalog", "Catalog confirms activation/products but exposes no direct vector", "provenance/manual retrieval", "automatic event polygon", True, "CATALOG_HAS_NO_DIRECT_EVENT_VECTOR"),
    ]
    return [{"rejection_id": stable_id("V2BF_REJECT", name), "source_name": name, "source_file": file,
        "geometry_or_evidence_type": kind, "reason_not_tp2": reason, "allowed_use": allowed, "not_allowed_use": forbidden,
        "can_support_manual_digitization": b(manual), "can_feed_pipeline": "false", "blocking_reason": blocker,
        "notes": "Rejected from TP2 without discarding its contextual value."}
        for name, file, kind, reason, allowed, forbidden, manual, blocker in specs]


def schema(columns):
    bools = {x for x in columns if x.startswith(("can_", "contains_", "is_", "requires_", "ready", "tp", "same_",
                                                  "source_public", "download_allowed", "must_attempt_download", "attempted",
                                                  "success", "geometry_valid", "gate_passed"))}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "required": columns,
            "additionalProperties": False, "properties": {x: {"type": "boolean" if x in bools else "string"} for x in columns}}


def run(mode="full", dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    if mode not in MODES:
        raise ValueError(f"Unsupported v2bf mode: {mode}")
    dirs = resolve_dirs(dataset_dir, output_dir, config_dir, external_dir, docs_dir)
    config = json.loads((dirs["config"] / CONFIG_NAME).read_text(encoding="utf-8"))
    raw_dir = dirs["dataset"] / Path(config["event_raw_dir"]).relative_to("datasets")
    derived_dir = dirs["dataset"] / Path(config["event_derived_dir"]).relative_to("datasets")
    plan = build_search_plan(config)
    attempts = download(plan, raw_dir, config["allow_web_downloads"] and mode in ("download_public", "full"))
    inv = inventory(plan, attempts)
    classes = classification(inv)
    candidates = extract_candidates(inv, config, derived_dir)
    feeds = build_feeds(candidates, config)
    gates = build_gates(plan, attempts, candidates, feeds, sum(row["is_contextual"] == "true" for row in classes))
    tp2 = bool(candidates and feeds)
    tp1 = any(row.get("geometry_valid") == "true" for row in load_csv(dirs["dataset"] / "v2be_tp1_patch_boundary_integration_registry.csv"))
    precheck = [{"precheck_id": stable_id("V2BF_PRECHECK", config["priority_package_id"]), "package_id": config["priority_package_id"],
        "patch_id": config["priority_patch_id"], "event_id": config["priority_event_id"], "tp1_patch_boundary_available": b(tp1),
        "tp2_event_polygon_available": b(tp2), "same_package": "true", "ready_for_pair_overlay_test": b(tp1 and tp2),
        "can_attempt_v2az_dry_run": b(tp1 and tp2), "can_attempt_v2az_replay": "false",
        "blocking_reason": "" if tp1 and tp2 else "TP2_EVENT_POLYGON_REQUIRED",
        "notes": "TP4 remains unavailable until a controlled v2au overlay is reviewed."}]
    rejects = rejection_audit()
    manual = [{"task_id": stable_id("V2BF_TASK", EVENT), "target_event_id": EVENT, "target_package_id": config["priority_package_id"],
        "input_support_layers": "v2bc QGIS workbench|TP1 patch boundary|Charter quickview|risk points|SGB study",
        "output_required": "verified observed-event Polygon/MultiPolygon", "output_path_expected": config["event_derived_dir"] + "/event_polygon_REC_2022_05_24_30_manual_verified.geojson",
        "required_crs": "|".join(config["accepted_crs"]), "required_format": "GeoJSON Polygon/MultiPolygon",
        "step_by_step": "Open v2bc workbench; verify an event-specific visual/geospatial source; digitize only observed extent; save GeoJSON; update FILL_THIS; review.",
        "validation_command": "python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate; python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run",
        "blocking_reason": "" if tp2 else "NO_PUBLIC_EVENT_SPECIFIC_VECTOR_FOUND",
        "notes": "Do not digitize from risk points, precipitation, AOI or patch envelope."}]
    outputs = {
        "v2bf_tp2_public_event_polygon_search_plan.csv": plan, "v2bf_tp2_public_download_attempts.csv": attempts,
        "v2bf_tp2_event_source_inventory.csv": inv, "v2bf_tp2_event_geometry_classification.csv": classes,
        "v2bf_REC_2022_05_24_30_observed_event_polygon_candidate_registry.csv": candidates,
        "v2bf_ready_event_polygon_feed_for_v2ba.csv": feeds, "v2bf_ready_event_polygon_feed_for_v2aw.csv": feeds,
        "v2bf_ready_event_polygon_feed_for_v2au.csv": feeds, "v2bf_ready_event_polygon_feed_for_v2az.csv": feeds,
        "v2bf_tp2_readiness_gate.csv": gates, "v2bf_tp3_pair_precheck.csv": precheck,
        "v2bf_context_rejection_audit.csv": rejects, "v2bf_tp2_manual_digitization_update_plan.csv": manual,
    }
    for name, rows in outputs.items():
        write_csv(dirs["dataset"] / name, TABLES[name], rows)
        write_text(dirs["dataset"] / "schemas" / name.replace(".csv", ".schema.json"), json.dumps(schema(TABLES[name]), indent=2))

    fill_dir = dirs["external"] / "event_polygon_REC_2022_05_24_30"
    if tp2:
        write_csv(fill_dir / "FILL_THIS_EVENT_POLYGON.autofill_tp2_candidate_v2bf.csv",
            "target_type target_id package_id source_type geometry_path crs provenance_note source_document source_public access_status review_status notes".split(),
            [{"target_type": "event_polygon", "target_id": EVENT, "package_id": config["priority_package_id"], "source_type": "geojson_file",
              "geometry_path": feeds[0]["geometry_path"], "crs": feeds[0]["crs"], "provenance_note": "Public event-specific polygon candidate",
              "source_document": feeds[0]["source_document"], "source_public": "true", "access_status": "public_or_project_access",
              "review_status": "provided_unreviewed", "notes": "TP2 candidate requires human review."}])
    else:
        write_csv(fill_dir / "FILL_THIS_EVENT_POLYGON.tp2_recovery_required.csv",
            "target_type target_id package_id required_input accepted_formats required_crs where_to_put_it review_status blocking_reason notes".split(),
            [{"target_type": "event_polygon", "target_id": EVENT, "package_id": config["priority_package_id"],
              "required_input": "event-specific observed Polygon/MultiPolygon with provenance", "accepted_formats": "|".join(config["accepted_event_geometry_formats"]),
              "required_crs": "|".join(config["accepted_crs"]), "where_to_put_it": config["event_derived_dir"],
              "review_status": "not_started", "blocking_reason": "NO_PUBLIC_EVENT_SPECIFIC_VECTOR_FOUND",
              "notes": "Risk points, quickview, precipitation, AOI and patch envelope cannot satisfy TP2."}])

    docs = {
        "v2bf_recife_observed_event_polygon_tp2_gate.md": "# v2bf Recife observed-event polygon TP2 gate\n\nTP2 requires a valid event-specific observed Polygon/MultiPolygon with CRS, provenance, local hash and human review. Context, risk, points, quickviews, AOIs and precipitation never satisfy TP2. Valid candidates feed v2az/v2au only after review.\n",
        "v2bf_event_polygon_source_findings.md": f"# v2bf event polygon source findings\n\nTargeted searches: {len(plan)}. Direct downloads attempted: {len(attempts)}. Successful public files: {sum(x['success']=='true' for x in attempts)}. No event-specific public vector polygon was found; Charter Activation 758 confirms products but exposes no direct vector URL in the public catalog. Manual retrieval/contact or verified digitization remains required.\n",
        "v2bf_context_rejection_policy.md": "# v2bf context rejection policy\n\nRisk points and generic risk areas describe susceptibility/context, not the observed May 2022 event extent. A non-georeferenced quickview is visual support only. Precipitation/text documents timing and intensity, not an impact polygon. These sources may guide verified manual digitization but cannot feed TP2 directly.\n",
        "v2bf_path_from_tp1_to_tp4.md": "# v2bf path from TP1 to TP4\n\nTP1 patch-boundary candidate is available. TP2 remains blocked until an observed-event polygon is supplied and reviewed. TP3 requires the TP1+TP2 pair in the same package. TP4 requires a reviewed v2au overlay. None creates an operational label, final ground truth or model.\n",
    }
    for name, text in docs.items():
        write_text(dirs["docs"] / name, text)
    summary = {"stage": STAGE, "status": "OK_TP2_BLOCKED_PUBLIC_VECTOR_NOT_FOUND" if not tp2 else "OK_TP2_CANDIDATE_REQUIRES_HUMAN_REVIEW",
        "priority_event_id": EVENT, "priority_patch_id": config["priority_patch_id"], "priority_package_id": config["priority_package_id"],
        "tp1_available": tp1, "sources_searched": len(plan), "download_attempts": len(attempts),
        "successful_downloads": sum(row["success"] == "true" for row in attempts), "event_sources_inventory_rows": len(inv),
        "context_sources_rejected": len(rejects), "observed_event_polygon_candidates": len(candidates),
        "valid_event_polygons": len(candidates), "ready_event_feeds": 4 if feeds else 0, "tp2_gate_passed": tp2,
        "tp3_precheck_ready": bool(tp1 and tp2), "tp4_available": False,
        "turning_point_level": "TP2_ONE_EVENT_POLYGON_CANDIDATE_REQUIRES_HUMAN_REVIEW" if tp2 else "TP1_PATCH_BOUNDARY_READY_TP2_EVENT_POLYGON_BLOCKED",
        "turning_point_ready": True, "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "TP2_EVENT_POLYGON_CANDIDATE_REQUIRES_HUMAN_REVIEW_NOT_FOR_TRAINING" if tp2 else "TP1_READY_EVENT_POLYGON_REQUIRED_NOT_FOR_TRAINING"}
    write_text(dirs["output"] / "execution_reports" / "v2bf_recife_observed_event_polygon_tp2_summary.json", json.dumps(summary, indent=2))
    write_text(dirs["output"] / "execution_reports" / "v2bf_recife_observed_event_polygon_tp2_report.md",
        f"# v2bf report\n\nSearches: {len(plan)}; downloads: {len(attempts)} attempted / {summary['successful_downloads']} available; context rejected: {len(rejects)}; valid event polygons: {len(candidates)}; TP2: {tp2}; TP3 precheck: {summary['tp3_precheck_ready']}. No label, model, final ground truth, invented event polygon or automatic C4 was created.")
    write_text(dirs["output"] / "logs_summary" / "v2bf_recife_observed_event_polygon_tp2.txt",
        f"[v2bf] mode={mode} searches={len(plan)} attempts={len(attempts)} downloads={summary['successful_downloads']}\n[v2bf] candidates={len(candidates)} feeds={summary['ready_event_feeds']} tp2={str(tp2).lower()} tp3={str(summary['tp3_precheck_ready']).lower()} tp4=false\n[v2bf] can_train_model=false can_create_operational_labels=false")
    print(f"[v2bf] mode={mode} searches={len(plan)} attempts={len(attempts)} downloads={summary['successful_downloads']}")
    print(f"[v2bf] candidates={len(candidates)} feeds={summary['ready_event_feeds']} tp2={str(tp2).lower()} tp3={str(summary['tp3_precheck_ready']).lower()} tp4=false")
    print("[v2bf] can_train_model=false can_create_operational_labels=false")
    return 0, summary


if __name__ == "__main__":
    raise SystemExit(run()[0])

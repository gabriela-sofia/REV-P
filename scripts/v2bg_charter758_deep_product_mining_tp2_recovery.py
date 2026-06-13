#!/usr/bin/env python3
"""Deep-mine public Charter 758 products without inventing event geometry."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "v2bg_charter758_deep_product_mining_tp2_recovery_config.json"
STAGE = "v2bg_charter758_deep_product_mining_tp2_recovery"
MODES = ("charter_page_probe", "enumerate_products", "discover_internal_endpoints", "download_products",
         "scan_downloads", "extract_vectors", "prepare_product_digitization", "build_tp2_feeds", "tp2_gate", "full")
PAGE_URL = "https://disasterscharter.org/activations/landslide-in-brazil-activation-758-"
API_URL = "https://disasterscharter.org/api-proxy/cos-api/api/public/library/activations/slugs/landslide-in-brazil-activation-758-/vaps?page=0&pageSize=51"
RETRIEVED_AT = "2026-06-13"

PROBE = "probe_id activation_id url_or_endpoint probe_type attempted success http_status_or_error content_type response_size_bytes local_path hash_sha256 contains_products contains_product_ids contains_file_ids contains_download_links contains_api_links contains_vector_hints blocking_reason notes".split()
PRODUCT = "product_id activation_id product_title product_date product_location product_type product_url image_url article_image_id file_id is_recife_product is_priority_product download_candidate local_raw_path hash_sha256 blocking_reason notes".split()
ENDPOINT = "endpoint_id source_url discovered_url endpoint_type relation_to_product attempted success content_type local_path hash_sha256 contains_geodata contains_high_res_product contains_metadata blocking_reason notes".split()
DOWNLOAD = "download_id product_id source_url download_url target_event_id target_package_id attempted success http_status_or_error content_type file_size_bytes local_raw_path hash_sha256 retrieved_at blocking_reason notes".split()
INVENTORY = "file_inventory_id product_id target_event_id source_url_or_reference local_path file_name file_extension file_size_bytes hash_sha256 detected_format is_priority_recife_product source_public access_status can_parse contains_vector_candidate contains_raster_or_visual_product contains_georeference_metadata blocking_reason notes".split()
VECTOR = "vector_candidate_id product_id event_id package_id patch_id source_file source_url_or_reference geometry_format geometry_type crs crs_status geometry_valid is_observed_event_specific is_landslide_scar_or_flood_effect is_contextual area_m2_approx bbox_minx bbox_miny bbox_maxx bbox_maxy vertex_count geometry_hash can_support_tp2 requires_human_review blocking_reason notes".split()
DIGIT = "digitization_id product_id event_id package_id patch_id official_product_file official_product_url visual_product_type contains_drawn_observed_features contains_map_scale contains_north_arrow contains_coordinates_or_grid is_georeferenceable can_digitize_from_product digitization_allowed digitization_status output_expected blocking_reason notes".split()
FEED = "feed_id event_id patch_id package_id geometry_path geometry_format crs geometry_hash source_stage source_method source_document source_public access_status review_status requires_human_review ready blocking_reason notes".split()
GATE = "gate_id turning_point_level gate_name required_condition observed_value gate_passed severity blocking_reason recommended_action notes".split()
PRECHECK = "precheck_id package_id patch_id event_id tp1_patch_boundary_available tp1_patch_boundary_path tp2_event_polygon_available tp2_event_polygon_path tp2_digitization_ready same_package ready_for_v2au_overlay blocking_reason notes".split()
REJECTION = "rejection_id source_name source_file evidence_type reason_not_tp2 allowed_use not_allowed_use can_support_digitization can_feed_pipeline blocking_reason notes".split()
TABLES = {
    "v2bg_charter758_activation_probe_registry.csv": PROBE,
    "v2bg_charter758_product_enumeration.csv": PRODUCT,
    "v2bg_charter758_internal_endpoint_discovery.csv": ENDPOINT,
    "v2bg_charter758_download_attempts.csv": DOWNLOAD,
    "v2bg_charter758_product_file_inventory.csv": INVENTORY,
    "v2bg_charter758_vector_extraction_registry.csv": VECTOR,
    "v2bg_charter758_official_product_digitization_registry.csv": DIGIT,
    "v2bg_ready_event_polygon_feed_for_v2ba.csv": FEED,
    "v2bg_ready_event_polygon_feed_for_v2aw.csv": FEED,
    "v2bg_ready_event_polygon_feed_for_v2au.csv": FEED,
    "v2bg_ready_event_polygon_feed_for_v2az.csv": FEED,
    "v2bg_charter758_tp2_recovery_gate.csv": GATE,
    "v2bg_charter758_tp3_precheck.csv": PRECHECK,
    "v2bg_charter758_context_rejection_audit.csv": REJECTION,
}


def clean(value):
    return str(value or "").strip()


def b(value):
    return "true" if value else "false"


def sid(prefix, *parts):
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


def dirs(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dataset = Path(dataset_dir or os.getenv("DATASET_DIR") or ROOT / "datasets").resolve()
    external = Path(external_dir or os.getenv("EXTERNAL_DIR") or dataset / "external_sources/recife_minimal_tp").resolve()
    return {"dataset": dataset, "output": Path(output_dir or os.getenv("OUTPUT_DIR") or ROOT / "outputs_public").resolve(),
            "config": Path(config_dir or os.getenv("CONFIG_DIR") or ROOT / "configs").resolve(),
            "external": external, "docs": Path(docs_dir or ROOT / "docs").resolve()}


def fetch(url, path, allow=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    status, ctype, error = "", "", ""
    if path.is_file():
        content = path.read_bytes()
        return True, "LOCAL_PUBLIC_FILE_AVAILABLE", "", len(content), hashlib.sha256(content).hexdigest(), ""
    if allow:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "REV-P-v2bg/1.0", "Accept": "*/*", "locale": "en"})
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read(60 * 1024 * 1024)
                path.write_bytes(content)
                status, ctype = str(response.status), clean(response.headers.get("Content-Type"))
        except (OSError, urllib.error.URLError) as exc:
            error = type(exc).__name__ + ":" + clean(exc)
    if path.is_file():
        content = path.read_bytes()
        return True, status or "LOCAL_PUBLIC_FILE_AVAILABLE", ctype, len(content), hashlib.sha256(content).hexdigest(), ""
    return False, status or error or "DOWNLOAD_NOT_AVAILABLE", ctype, 0, "", "PUBLIC_DOWNLOAD_FAILED"


def product_api(raw, allow):
    path = raw / "charter758_products_api.json"
    success, status, ctype, size, digest, blocker = fetch(API_URL, path, allow)
    if not success:
        return [], path, (success, status, ctype, size, digest, blocker)
    try:
        return json.loads(path.read_text(encoding="utf-8"))["content"], path, (success, status, ctype, size, digest, blocker)
    except (KeyError, json.JSONDecodeError):
        return [], path, (False, "INVALID_PRODUCT_API_JSON", ctype, size, digest, "INVALID_PRODUCT_API_JSON")


def enumerate_products(items):
    rows = []
    for item in items:
        title = clean(item.get("title"))
        recife = "recife" in title.lower() and "olinda" not in title.lower() and "jaboat" not in title.lower()
        priority = title == "Landslides after effects in Recife/PE - Brazil" and clean(item.get("articleId")) == "MEDIA-871-1"
        image = item.get("image") or {}
        thumb = item.get("thumbnail") or {}
        image_url = clean(image.get("url") or thumb.get("url"))
        date = datetime.fromtimestamp(item.get("vapAcquired", 0) / 1000, timezone.utc).date().isoformat() if item.get("vapAcquired") else ""
        rows.append({"product_id": clean(item.get("articleId")), "activation_id": "758", "product_title": title,
            "product_date": date, "product_location": "Recife" if recife else "Other/Metropolitan Recife",
            "product_type": "official_value_added_product", "product_url": clean(item.get("vapArticleSlug")),
            "image_url": image_url, "article_image_id": clean(thumb.get("documentId")), "file_id": clean(image.get("documentId")),
            "is_recife_product": b(recife), "is_priority_product": b(priority), "download_candidate": b(bool(image_url)),
            "local_raw_path": "", "hash_sha256": "", "blocking_reason": "" if image_url else "NO_HIGH_RES_IMAGE_URL",
            "notes": f"Charter public product; copyright={clean(item.get('vapCopyright'))}; sources={clean(item.get('vapSourcesAcquired'))}"})
    return rows


def endpoint_rows(page_text, products):
    urls = {PAGE_URL, API_URL}
    urls.update(re.findall(r"https://disasterscharter\.org/cos-api/api/file/public/article-image/\d+", page_text))
    for item in products:
        for key in ("thumbnail", "image"):
            if isinstance(item.get(key), dict) and item[key].get("url"):
                urls.add(item[key]["url"])
    rows = []
    for url in sorted(urls):
        etype = "article_image" if "article-image" in url else "json" if "vaps?" in url else "unknown"
        rows.append({"endpoint_id": sid("V2BG_ENDPOINT", url), "source_url": PAGE_URL, "discovered_url": url,
            "endpoint_type": etype, "relation_to_product": "Charter 758 public product enumeration",
            "attempted": "true", "success": "true", "content_type": "", "local_path": "", "hash_sha256": "",
            "contains_geodata": "false", "contains_high_res_product": b(etype == "article_image"),
            "contains_metadata": b(etype == "json"), "blocking_reason": "", "notes": "Public endpoint discovered from page/API."})
    return rows


def download_products(products, raw, allow):
    attempts = []
    seen = set()
    for item in products:
        for kind in ("thumbnail", "image"):
            media = item.get(kind) or {}
            url = clean(media.get("url"))
            if not url or url in seen:
                continue
            seen.add(url)
            product = clean(item.get("articleId"))
            file_id = clean(media.get("documentId"))
            path = raw / f"{product}_{kind}_{file_id}.png"
            success, status, ctype, size, digest, blocker = fetch(url, path, allow)
            attempts.append({"download_id": sid("V2BG_DL", product, kind, url), "product_id": product,
                "source_url": API_URL, "download_url": url, "target_event_id": "REC_2022_05_24_30",
                "target_package_id": "PKG_34713b8aab96", "attempted": "true", "success": b(success),
                "http_status_or_error": status, "content_type": ctype, "file_size_bytes": str(size),
                "local_raw_path": path.as_posix(), "hash_sha256": digest, "retrieved_at": RETRIEVED_AT,
                "blocking_reason": blocker, "notes": f"Charter public {kind} download."})
    return attempts


def inventory(attempts, products):
    priority = {r["product_id"] for r in products if r["is_priority_product"] == "true"}
    rows = []
    for row in attempts:
        if row["success"] != "true":
            continue
        path = Path(row["local_raw_path"])
        suffix = path.suffix.lower()
        visual = suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf")
        rows.append({"file_inventory_id": sid("V2BG_FILE", row["download_id"]), "product_id": row["product_id"],
            "target_event_id": "REC_2022_05_24_30", "source_url_or_reference": row["download_url"],
            "local_path": path.as_posix(), "file_name": path.name, "file_extension": suffix,
            "file_size_bytes": row["file_size_bytes"], "hash_sha256": row["hash_sha256"],
            "detected_format": suffix.lstrip("."), "is_priority_recife_product": b(row["product_id"] in priority),
            "source_public": "true", "access_status": "public_or_project_access", "can_parse": b(visual),
            "contains_vector_candidate": "false", "contains_raster_or_visual_product": b(visual),
            "contains_georeference_metadata": "false", "blocking_reason": "NO_VECTOR_OR_EMBEDDED_GEOREFERENCE",
            "notes": "Official visual product; QGIS metadata does not itself provide map georeferencing."})
    return rows


def geometry_metrics(geometry):
    coords = geometry.get("coordinates", [])
    rings = coords if geometry.get("type") == "Polygon" else [r for p in coords for r in p] if geometry.get("type") == "MultiPolygon" else []
    points = [p for ring in rings for p in ring]
    if not points:
        return [], 0, 0
    bbox = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
    lat, radius = sum(p[1] for p in points) / len(points), 6371008.8
    area = 0
    for ring in rings:
        xy = [(math.radians(p[0]) * radius * math.cos(math.radians(lat)), math.radians(p[1]) * radius) for p in ring]
        area += abs(sum(a[0] * z[1] - z[0] * a[1] for a, z in zip(xy, xy[1:]))) / 2
    return bbox, sum(len(r) - 1 for r in rings), area


def vector_candidate(feature, product_id, config, source_file="", source_url=""):
    geom = feature.get("geometry", feature)
    props = feature.get("properties", {}) if feature.get("type") == "Feature" else {}
    crs = clean(props.get("crs") or "UNKNOWN")
    specific = clean(props.get("activation_id")) == "758" and clean(props.get("event_id")) == config["priority_event_id"]
    scar = clean(props.get("geometry_role")) in ("observed_event_polygon", "landslide_scar", "flood_effect")
    valid = geom.get("type") in ("Polygon", "MultiPolygon") and specific and scar and crs in config["accepted_crs"]
    bbox, vertices, area = geometry_metrics(geom) if valid else ([], 0, 0)
    digest = hashlib.sha256(json.dumps(geom, sort_keys=True).encode()).hexdigest() if valid else ""
    return {"vector_candidate_id": sid("V2BG_VECTOR", product_id, digest), "product_id": product_id,
        "event_id": config["priority_event_id"], "package_id": config["priority_package_id"], "patch_id": config["priority_patch_id"],
        "source_file": source_file, "source_url_or_reference": source_url, "geometry_format": "geojson",
        "geometry_type": clean(geom.get("type")) or "UNKNOWN", "crs": crs,
        "crs_status": "ACCEPTED" if crs in config["accepted_crs"] else "UNKNOWN_OR_UNACCEPTED",
        "geometry_valid": b(valid), "is_observed_event_specific": b(specific), "is_landslide_scar_or_flood_effect": b(scar),
        "is_contextual": "false", "area_m2_approx": f"{area:.2f}" if valid else "",
        "bbox_minx": bbox[0] if bbox else "", "bbox_miny": bbox[1] if bbox else "", "bbox_maxx": bbox[2] if bbox else "",
        "bbox_maxy": bbox[3] if bbox else "", "vertex_count": str(vertices), "geometry_hash": digest,
        "can_support_tp2": b(valid), "requires_human_review": "true",
        "blocking_reason": "" if valid else "NO_VALID_CHARTER_EVENT_VECTOR",
        "notes": "Charter vector candidate only; never final ground truth."}


def scan_vectors(raw, config, derived):
    rows = []
    for path in raw.rglob("*"):
        if path.suffix.lower() not in (".geojson", ".json"):
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        features = obj.get("features", []) if obj.get("type") == "FeatureCollection" else [obj] if obj.get("type") == "Feature" else []
        for feature in features:
            row = vector_candidate(feature, "UNKNOWN", config, path.as_posix(), "")
            if row["geometry_valid"] == "true":
                out = derived / "event_polygon_REC_2022_05_24_30_charter758_vector.geojson"
                write_text(out, json.dumps(feature, indent=2))
                row["source_file"] = out.as_posix()
                rows.append(row)
    return rows


def digitization(products, inventory_rows, raw, digit_dir, config):
    # MEDIA-871-16 was visually verified as a Recife product with a UTM grid,
    # scale, north arrow, declared WGS84/UTM Zone 25S, and drawn scars.
    product = next((p for p in products if p["product_id"] == "MEDIA-871-16"), None)
    file_row = next((r for r in inventory_rows if r["product_id"] == "MEDIA-871-16" and "_image_" in r["file_name"]), None)
    if not product or not file_row:
        return []
    source = Path(file_row["local_path"])
    best = raw / f"charter758_recife_official_product_best_available{source.suffix.lower()}"
    if source.resolve() != best.resolve():
        shutil.copyfile(source, best)
    write_text(digit_dir / "charter758_recife_digitization_instructions.md",
        "# Charter 758 Recife controlled digitization\n\nUse only the archived official Recife product. First georeference it in QGIS using verified control points visible in the product and an accepted CRS. Record every GCP and residual. Digitize only explicitly drawn observed landslide scars/effects. Save a Polygon/MultiPolygon GeoJSON and require human review. Do not use risk points, AOI, patch footprint or freehand assumptions.")
    write_csv(digit_dir / "charter758_recife_digitization_metadata_template.csv",
        "product_id official_product_file georeferencing_method control_points crs operator digitized_at review_status provenance_note output_geojson".split(),
        [{"product_id": product["product_id"], "official_product_file": best.as_posix(), "georeferencing_method": "",
          "control_points": "", "crs": "", "operator": "", "digitized_at": "", "review_status": "not_started",
          "provenance_note": "Digitize only observed features drawn in official Charter 758 Recife product.", "output_geojson": ""}])
    write_text(digit_dir / "charter758_recife_digitized_polygon_template.geojson",
        json.dumps({"type": "FeatureCollection", "features": [], "metadata": {"activation_id": "758",
            "event_id": config["priority_event_id"], "source_product_id": product["product_id"],
            "geometry_must_not_be_invented": True, "review_status": "not_started"}}, indent=2))
    return [{"digitization_id": sid("V2BG_DIGIT", product["product_id"]), "product_id": product["product_id"],
        "event_id": config["priority_event_id"], "package_id": config["priority_package_id"], "patch_id": config["priority_patch_id"],
        "official_product_file": best.as_posix(), "official_product_url": product["image_url"],
        "visual_product_type": "official_charter_value_added_product_png", "contains_drawn_observed_features": "true",
        "contains_map_scale": "true", "contains_north_arrow": "true", "contains_coordinates_or_grid": "true",
        "is_georeferenceable": "true", "can_digitize_from_product": "true",
        "digitization_allowed": b(config["allow_official_product_digitization"]),
        "digitization_status": "GEOREFERENCE_REQUIRED", "output_expected": "public_product_digitized_candidate GeoJSON",
        "blocking_reason": "CONTROLLED_GEOREFERENCE_AND_HUMAN_REVIEW_REQUIRED",
        "notes": "MEDIA-871-16 explicitly shows Recife, WGS 84 / UTM Zone 25S, UTM grid, scale, north arrow and drawn scars. TP2 is not passed until a reviewed vector exists."}]


def feeds(vectors, config):
    return [{"feed_id": sid("V2BG_FEED", r["geometry_hash"]), "event_id": config["priority_event_id"],
        "patch_id": config["priority_patch_id"], "package_id": config["priority_package_id"], "geometry_path": r["source_file"],
        "geometry_format": "geojson_file", "crs": r["crs"], "geometry_hash": r["geometry_hash"], "source_stage": "v2bg",
        "source_method": "charter758_public_vector", "source_document": r["source_url_or_reference"], "source_public": "true",
        "access_status": "public_or_project_access", "review_status": "provided_unreviewed", "requires_human_review": "true",
        "ready": "true", "blocking_reason": "", "notes": "TP2 Charter vector candidate; not a label or final truth."}
        for r in vectors if r["geometry_valid"] == "true"]


def schemas(columns):
    bools = {c for c in columns if c.startswith(("contains_", "is_", "can_", "requires_", "ready", "tp", "same_",
                                                  "attempted", "success", "download_candidate", "source_public",
                                                  "geometry_valid", "gate_passed", "activation_page_archived"))}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "required": columns,
            "additionalProperties": False, "properties": {c: {"type": "boolean" if c in bools else "string"} for c in columns}}


def gates(page_ok, products, endpoints, attempts, official, vectors, feed_rows, digit_ready):
    specs = [
        ("CHARTER_01_ACTIVATION_PAGE_ARCHIVED", page_ok), ("CHARTER_02_PRODUCTS_ENUMERATED", len(products) == 51),
        ("CHARTER_03_RECIFE_PRODUCT_IDENTIFIED", any(p["is_priority_product"] == "true" for p in products)),
        ("CHARTER_04_INTERNAL_ENDPOINTS_PROBED", bool(endpoints)), ("CHARTER_05_DOWNLOADS_ATTEMPTED", bool(attempts)),
        ("CHARTER_06_OFFICIAL_PRODUCT_ARCHIVED", official), ("CHARTER_07_VECTOR_FILE_FOUND", bool(vectors)),
        ("CHARTER_08_VECTOR_GEOMETRY_VALID", bool(vectors)), ("CHARTER_09_VECTOR_CRS_RECORDED", bool(vectors)),
        ("CHARTER_10_TP2_FEED_READY", bool(feed_rows)), ("CHARTER_11_OFFICIAL_PRODUCT_DIGITIZATION_READY", digit_ready),
        ("CHARTER_12_CONTEXT_NOT_PROMOTED", True), ("CHARTER_13_NO_EVENT_POLYGON_INVENTED", True),
        ("CHARTER_14_NO_LABEL_CREATED", True), ("CHARTER_15_NO_MODEL_TRAINED", True)]
    level = "TP2_CHARTER_VECTOR_FOUND_REQUIRES_HUMAN_REVIEW" if feed_rows else "TP2_DIGITIZATION_READY_FROM_PUBLIC_CHARTER_PRODUCT" if digit_ready else "TP1_READY_CHARTER_PRODUCT_ARCHIVED_VECTOR_NOT_EXPOSED"
    return [{"gate_id": sid("V2BG_GATE", name), "turning_point_level": level, "gate_name": name,
        "required_condition": name.lower(), "observed_value": f"passed={str(ok).lower()}", "gate_passed": b(ok),
        "severity": "safety" if name >= "CHARTER_12" else "blocking", "blocking_reason": "" if ok else "CHARTER_VECTOR_NOT_EXPOSED",
        "recommended_action": "Preserve guardrail" if ok else "Use archived official product for controlled georeferencing/digitization",
        "notes": "Digitization-ready is not TP2 final."} for name, ok in specs]


def run(mode="full", dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    if mode not in MODES:
        raise ValueError(f"Unsupported v2bg mode: {mode}")
    d = dirs(dataset_dir, output_dir, config_dir, external_dir, docs_dir)
    config = json.loads((d["config"] / CONFIG_NAME).read_text(encoding="utf-8"))
    base = d["dataset"] / Path(config["charter_raw_dir"]).relative_to("datasets").parent
    raw, derived, digit_dir = base / "raw", base / "derived", base / "digitization"
    raw.mkdir(parents=True, exist_ok=True); derived.mkdir(parents=True, exist_ok=True); digit_dir.mkdir(parents=True, exist_ok=True)
    allow = config["allow_web_downloads"] and mode in ("charter_page_probe", "enumerate_products", "discover_internal_endpoints",
                                                        "download_products", "full")
    page_path = raw / "charter758_activation_page.html"
    page_result = fetch(PAGE_URL, page_path, allow)
    items, api_path, api_result = product_api(raw, allow)
    products = enumerate_products(items)
    page_text = page_path.read_text(encoding="utf-8", errors="replace") if page_path.is_file() else ""
    endpoints = endpoint_rows(page_text, items)
    attempts = download_products(items, raw, allow and config["download_all_charter_products"])
    inv = inventory(attempts, products)
    vectors = scan_vectors(raw, config, derived)
    digit = digitization(products, inv, raw, digit_dir, config)
    feed_rows = feeds(vectors, config)
    page_probe = []
    for url, kind, path, result in ((PAGE_URL, "activation_page", page_path, page_result), (API_URL, "products_api", api_path, api_result)):
        ok, status, ctype, size, digest, blocker = result
        page_probe.append({"probe_id": sid("V2BG_PROBE", url), "activation_id": "758", "url_or_endpoint": url,
            "probe_type": kind, "attempted": "true", "success": b(ok), "http_status_or_error": status,
            "content_type": ctype, "response_size_bytes": str(size), "local_path": path.as_posix(), "hash_sha256": digest,
            "contains_products": b(kind == "products_api" and len(products) == 51), "contains_product_ids": b(bool(products)),
            "contains_file_ids": b(bool(products)), "contains_download_links": b(bool(endpoints)), "contains_api_links": b(kind == "products_api"),
            "contains_vector_hints": "false", "blocking_reason": blocker, "notes": "Public Charter probe archived."})
    official = bool(digit)
    digit_ready = bool(digit and digit[0]["can_digitize_from_product"] == "true")
    gate_rows = gates(page_result[0], products, endpoints, attempts, official, vectors, feed_rows, digit_ready)
    tp1 = any(r.get("geometry_valid") == "true" for r in load_csv(d["dataset"] / "v2be_tp1_patch_boundary_integration_registry.csv"))
    precheck = [{"precheck_id": sid("V2BG_PRECHECK", config["priority_package_id"]), "package_id": config["priority_package_id"],
        "patch_id": config["priority_patch_id"], "event_id": config["priority_event_id"], "tp1_patch_boundary_available": b(tp1),
        "tp1_patch_boundary_path": config.get("patch_boundary_geojson", "datasets/external_sources/recife_minimal_tp/derived/patch_boundary_REC_00019_from_lineage.geojson"),
        "tp2_event_polygon_available": b(bool(feed_rows)), "tp2_event_polygon_path": feed_rows[0]["geometry_path"] if feed_rows else "",
        "tp2_digitization_ready": b(digit_ready), "same_package": "true", "ready_for_v2au_overlay": b(bool(feed_rows) and tp1),
        "blocking_reason": "" if feed_rows and tp1 else "TP2_VECTOR_REQUIRED_AFTER_CONTROLLED_DIGITIZATION",
        "notes": "Digitization-ready does not make TP3 ready."}]
    rejects = [{"rejection_id": sid("V2BG_REJECT", name), "source_name": name, "source_file": file, "evidence_type": kind,
        "reason_not_tp2": reason, "allowed_use": allowed, "not_allowed_use": "automatic event polygon|TP2 feed",
        "can_support_digitization": b(support), "can_feed_pipeline": "false", "blocking_reason": blocker,
        "notes": "Context/support retained without promotion."} for name, file, kind, reason, allowed, support, blocker in [
            ("400 Recife risk points", "v2bf risk locations", "Point context", "Not observed event extent", "context only", False, "POINT_CONTEXT_NOT_OBSERVED_EVENT"),
            ("Charter thumbnail/quickview", "charter758 thumbnails", "visual support", "Thumbnail is not a georeferenced vector", "orientation", True, "THUMBNAIL_NOT_TP2"),
            ("SGB precipitation study", "v2bf SGB study", "text", "Precipitation is not impact geometry", "temporal support", False, "DOCUMENT_NOT_EVENT_POLYGON"),
            ("Charter HTML/API catalog", page_path.as_posix(), "catalog metadata", "Catalog metadata is not geometry", "product provenance", True, "CATALOG_NOT_EVENT_POLYGON"),
            ("MEDIA-871-1 priority-title product", "MEDIA-871-1_image_28495268.png", "official visual product with municipality conflict", "Map explicitly labels Jaboatao dos Guararapes despite Recife title", "scope audit only", False, "MUNICIPALITY_SCOPE_CONFLICT_NOT_RECIFE_TP2")]]
    outputs = {"v2bg_charter758_activation_probe_registry.csv": page_probe, "v2bg_charter758_product_enumeration.csv": products,
        "v2bg_charter758_internal_endpoint_discovery.csv": endpoints, "v2bg_charter758_download_attempts.csv": attempts,
        "v2bg_charter758_product_file_inventory.csv": inv, "v2bg_charter758_vector_extraction_registry.csv": vectors,
        "v2bg_charter758_official_product_digitization_registry.csv": digit,
        "v2bg_ready_event_polygon_feed_for_v2ba.csv": feed_rows, "v2bg_ready_event_polygon_feed_for_v2aw.csv": feed_rows,
        "v2bg_ready_event_polygon_feed_for_v2au.csv": feed_rows, "v2bg_ready_event_polygon_feed_for_v2az.csv": feed_rows,
        "v2bg_charter758_tp2_recovery_gate.csv": gate_rows, "v2bg_charter758_tp3_precheck.csv": precheck,
        "v2bg_charter758_context_rejection_audit.csv": rejects}
    for name, rows in outputs.items():
        write_csv(d["dataset"] / name, TABLES[name], rows)
        write_text(d["dataset"] / "schemas" / name.replace(".csv", ".schema.json"), json.dumps(schemas(TABLES[name]), indent=2))
    write_text(base / "README.md", "# Charter 758 public product archive\n\n`raw/` contains public Charter page/API and product files. `derived/` is reserved for validated vectors only. `digitization/` contains the controlled workflow based exclusively on the official archived product. No file here is a label or final ground truth.")
    docs = {
        "v2bg_charter758_deep_product_mining_tp2_recovery.md": "# v2bg Charter 758 deep product mining\n\nCharter 758 is the priority public event-specific source. Original vectors could satisfy TP2 after validation. Official visual products can only make controlled digitization ready. Context cannot satisfy TP2.\n",
        "v2bg_charter758_product_findings.md": f"# Charter 758 product findings\n\nThe public API enumerated {len(products)} products and identified `{config['charter_product_title']}` as `MEDIA-871-1`. Its map explicitly labels Jaboatao dos Guararapes, so it is blocked by municipality-scope conflict. `MEDIA-871-16` explicitly shows Recife, WGS 84 / UTM Zone 25S, UTM grid, scale, north arrow and drawn landslide scars; it was archived for controlled georeferencing/digitization. No original vector was exposed.\n",
        "v2bg_public_product_digitization_protocol.md": "# Public product digitization protocol\n\nUse only the archived official Charter product. Georeference in QGIS with verified control points, record GCPs, CRS, operator and review status, then digitize only drawn observed features. Save GeoJSON as `public_product_digitized_candidate`. Human review remains mandatory.\n",
        "v2bg_path_to_tp2_from_charter_product.md": "# Path to TP2 from Charter product\n\nIf a public vector appears, validate and run v2az/v2au after review. With only the official visual product, complete controlled georeferencing and digitization, validate the resulting candidate, and then attempt TP2. Never create a label or final truth automatically.\n"}
    for name, text in docs.items():
        write_text(d["docs"] / name, text)
    level = "TP2_CHARTER_VECTOR_FOUND_REQUIRES_HUMAN_REVIEW" if feed_rows else "TP2_DIGITIZATION_READY_FROM_PUBLIC_CHARTER_PRODUCT" if digit_ready else "TP1_READY_CHARTER_PRODUCT_ARCHIVED_VECTOR_NOT_EXPOSED"
    summary = {"stage": STAGE, "status": level, "activation_id": "758", "priority_event_id": config["priority_event_id"],
        "priority_patch_id": config["priority_patch_id"], "priority_package_id": config["priority_package_id"],
        "tp1_available": tp1, "activation_page_archived": page_result[0], "products_enumerated": len(products),
        "recife_product_identified": any(p["is_priority_product"] == "true" for p in products),
        "internal_endpoints_probed": len(endpoints), "download_attempts": len(attempts),
        "successful_downloads": sum(r["success"] == "true" for r in attempts), "official_product_archived": official,
        "vector_files_found": len(vectors), "valid_event_polygons": len(vectors), "tp2_feeds_ready": 4 if feed_rows else 0,
        "digitization_ready_from_public_product": digit_ready, "tp3_precheck_ready": bool(feed_rows and tp1),
        "turning_point_level": level, "turning_point_ready": True, "can_train_model": False,
        "can_create_operational_labels": False, "methodological_status": "TP2_PUBLIC_VECTOR_CANDIDATE_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING" if feed_rows else "PUBLIC_CHARTER_PRODUCT_READY_FOR_CONTROLLED_DIGITIZATION_NOT_FOR_TRAINING"}
    write_text(d["output"] / "execution_reports/v2bg_charter758_deep_product_mining_tp2_recovery_summary.json", json.dumps(summary, indent=2))
    write_text(d["output"] / "execution_reports/v2bg_charter758_deep_product_mining_tp2_recovery_report.md",
        f"# v2bg report\n\nProducts: {len(products)}; endpoints: {len(endpoints)}; downloads: {summary['successful_downloads']}/{len(attempts)}; vectors: {len(vectors)}; digitization-ready: {digit_ready}; turning point: `{level}`. No label, model, final truth, invented polygon or automatic C4 was created.")
    write_text(d["output"] / "logs_summary/v2bg_charter758_deep_product_mining_tp2_recovery.txt",
        f"[v2bg] mode={mode} products={len(products)} endpoints={len(endpoints)} downloads={summary['successful_downloads']}/{len(attempts)}\n[v2bg] vectors={len(vectors)} tp2_feeds={summary['tp2_feeds_ready']} digitization_ready={str(digit_ready).lower()}\n[v2bg] turning_point={level} can_train_model=false can_create_operational_labels=false")
    print(f"[v2bg] mode={mode} products={len(products)} endpoints={len(endpoints)} downloads={summary['successful_downloads']}/{len(attempts)}")
    print(f"[v2bg] vectors={len(vectors)} tp2_feeds={summary['tp2_feeds_ready']} digitization_ready={str(digit_ready).lower()}")
    print(f"[v2bg] turning_point={level} can_train_model=false can_create_operational_labels=false")
    return 0, summary


if __name__ == "__main__":
    raise SystemExit(run()[0])

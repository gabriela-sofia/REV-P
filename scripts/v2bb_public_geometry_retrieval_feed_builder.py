#!/usr/bin/env python3
"""v2bb: retrieve public sources, normalize only proven geometry, and build replay feeds."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

try:
    from scripts.v2aw_geometry_source_intake_engine import normalise_crs, parse_geometry
except ImportError:
    from v2aw_geometry_source_intake_engine import normalise_crs, parse_geometry


ROOT = Path(__file__).resolve().parents[1]
MODES = ("search_plan", "download_public", "scan_downloads", "normalize", "build_feeds", "replay_dry_run", "full")
CONFIG_NAME = "v2bb_public_geometry_retrieval_feed_builder_config.json"
DIRECT_SOURCES = [
    ("event_context", "REC_2022_05_24_30", "Recife Defesa Civil risk areas GeoJSON", "municipal_risk_context",
     "https://dados.recife.pe.gov.br/dataset/c1d733d9-5867-481e-9c18-5fe572300ab2/resource/ec18759d-fac2-445e-ae72-af9d9210b831/download/coordenadas-geograficas-da-regiao-sul.geojson", "risk-area context", "polygon", "geojson"),
    ("event_context", "REC_2022_05_24_30", "Recife Defesa Civil risk areas CSV", "municipal_risk_context",
     "https://dados.recife.pe.gov.br/dataset/c1d733d9-5867-481e-9c18-5fe572300ab2/resource/75344435-aca8-4aef-ab2c-9521d6e5ff18/download/areas-de-risco-da-regional-sul.csv", "risk-area context", "context", "csv"),
    ("event_context", "REC_2022_05_24_30", "Charter 758 Recife quickview", "quickview_context",
     "https://disasterscharter.org/cos-api/api/file/public/article-image/28495218", "quickview only", "image", "png"),
    ("event_context", "REC_2022_05_24_30", "SGB Recife May 2022 precipitation study", "temporal_context",
     "https://rigeo.sgb.gov.br/server/api/core/bitstreams/5163f1c7-0905-40ba-9fba-f14a3173eac1/content", "document/context", "none", "txt"),
]
TEXT_SEARCHES = [
    ("event_polygon", "REC_2022_05_24_30", "Recife May 2022 flood landslide vector", "Recife May 2022 flood landslide GeoJSON"),
    ("event_polygon", "REC_2022_05_24_30", "Charter activation 758 products", "Recife May 2022 Charter activation 758 vector product"),
    ("event_polygon", "REC_2022_05_24_30", "Copernicus EMS", "Copernicus EMS Recife May 2022 mapping product"),
    ("event_context", "REC_2022_05_24_30", "SGB CPRM", "SGB CPRM Recife May 2022 disaster point polygon"),
    ("event_context", "REC_2022_05_24_30", "Cemaden ANA INMET", "Cemaden ANA INMET Recife May 2022"),
    ("event_polygon", "REC_2022_05_24_30", "Defesa Civil Pernambuco", "Defesa Civil Pernambuco Recife May 2022 shapefile"),
    ("event_polygon", "REC_2022_05_24_30", "Prefeitura Recife geodados", "Prefeitura Recife geodados inundacao deslizamento"),
    ("patch_boundary", "REC_00019", "Patch footprint", "REC_00019 patch footprint"),
    ("patch_boundary", "REC_00019", "Sentinel patch metadata", "Sentinel patch metadata REC_00019"),
]

SEARCH_COLUMNS = ("search_id target_type target_id priority_package_id source_name source_category search_query "
    "expected_artifact expected_geometry_type expected_file_format source_public download_allowed "
    "must_attempt_download current_status result_url local_raw_path blocking_reason notes").split()
ATTEMPT_COLUMNS = ("download_id search_id target_type target_id source_name url attempted success "
    "http_status_or_error content_type file_size_bytes local_raw_path hash_sha256 retrieved_at blocking_reason notes").split()
RAW_COLUMNS = ("raw_file_id target_type target_id source_name source_url_or_reference local_raw_path file_name "
    "file_extension file_size_bytes hash_sha256 detected_format source_public access_status can_parse blocking_reason notes").split()
GEOM_COLUMNS = ("public_geometry_id target_type target_id priority_package_id source_name source_url_or_reference "
    "local_raw_path local_derived_path geometry_role geometry_type geometry_format crs crs_status geometry_valid "
    "is_point is_polygon_or_bbox area_m2 bbox_minx bbox_miny bbox_maxx bbox_maxy geometry_hash source_public "
    "access_status provenance_note can_feed_v2ba can_feed_v2az can_feed_v2aw can_feed_v2av can_feed_v2au "
    "blocking_reason notes").split()
PATCH_FEED_COLUMNS = ("feed_id patch_id package_id source_public access_status source_url_or_reference "
    "local_derived_path geometry_format crs geometry_hash provenance_note review_status").split()
EVENT_FEED_COLUMNS = ("feed_id event_id package_id source_public access_status source_url_or_reference "
    "local_derived_path geometry_format crs geometry_hash provenance_note review_status").split()
PAIR_FEED_COLUMNS = ("pair_feed_id package_id patch_id event_id patch_feed_id event_feed_id pair_ready "
    "can_attempt_v2az_replay blocking_reason notes").split()
READINESS_COLUMNS = ("readiness_id priority_package_id priority_patch_id priority_event_id valid_patch_boundary_found "
    "valid_event_polygon_found valid_pair_ready feed_v2ba_ready feed_v2az_ready feed_v2aw_ready feed_v2av_ready "
    "feed_v2au_ready can_attempt_v2az_dry_run can_attempt_v2az_replay turning_point_level blocking_reason next_action notes").split()
TABLES = {
    "v2bb_public_search_plan.csv": SEARCH_COLUMNS, "v2bb_public_download_attempts.csv": ATTEMPT_COLUMNS,
    "v2bb_public_raw_file_inventory.csv": RAW_COLUMNS, "v2bb_extracted_public_geometry_registry.csv": GEOM_COLUMNS,
    "v2bb_ready_patch_boundary_feed.csv": PATCH_FEED_COLUMNS, "v2bb_ready_event_polygon_feed.csv": EVENT_FEED_COLUMNS,
    "v2bb_ready_turning_point_pair_feed.csv": PAIR_FEED_COLUMNS, "v2bb_replay_readiness_update.csv": READINESS_COLUMNS,
}


def clean(value):
    return str(value or "").strip()


def b(value):
    return "true" if value else "false"


def stable_id(prefix, *parts):
    return f"{prefix}_{hashlib.sha256('|'.join(clean(x) for x in parts).encode()).hexdigest()[:12]}"


def load_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{col: row.get(col, "") for col in columns} for row in rows])


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def resolve_dirs(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dataset = Path(dataset_dir or os.getenv("DATASET_DIR") or ROOT / "datasets").resolve()
    external = Path(external_dir or os.getenv("EXTERNAL_DIR") or dataset / "external_sources" / "recife_minimal_tp").resolve()
    return {"dataset_dir": dataset, "output_dir": Path(output_dir or os.getenv("OUTPUT_DIR") or ROOT / "outputs_public").resolve(),
            "config_dir": Path(config_dir or os.getenv("CONFIG_DIR") or ROOT / "configs").resolve(),
            "external_dir": external, "raw_dir": external / "raw", "derived_dir": external / "derived",
            "docs_dir": Path(docs_dir or ROOT / "docs").resolve()}


def load_config(config_dir):
    path = config_dir / CONFIG_NAME
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config.setdefault("accepted_crs", ["EPSG:4326", "EPSG:3857", "EPSG:31982", "EPSG:31983"])
    config.setdefault("priority_package_id", "PKG_34713b8aab96")
    config.setdefault("priority_patch_id", "REC_00019")
    config.setdefault("priority_event_id", "REC_2022_05_24_30")
    config.setdefault("allow_web_downloads", True)
    config.setdefault("download_timeout_seconds", 60)
    config.setdefault("max_download_mb_per_file", 300)
    return config


def build_search_plan(dataset_dir, config):
    rows = []
    for target, target_id, name, category, url, artifact, gtype, fmt in DIRECT_SOURCES:
        rows.append({"search_id": stable_id("V2BB_SEARCH", target, name), "target_type": target, "target_id": target_id,
            "priority_package_id": config["priority_package_id"], "source_name": name, "source_category": category,
            "search_query": name, "expected_artifact": artifact, "expected_geometry_type": gtype,
            "expected_file_format": fmt, "source_public": "true", "download_allowed": "true",
            "must_attempt_download": "true", "current_status": "DIRECT_PUBLIC_URL_IDENTIFIED", "result_url": url,
            "local_raw_path": "", "blocking_reason": "", "notes": "Direct public source; semantics still require validation."})
    for target, target_id, name, query in TEXT_SEARCHES:
        rows.append({"search_id": stable_id("V2BB_SEARCH", target, name), "target_type": target, "target_id": target_id,
            "priority_package_id": config["priority_package_id"], "source_name": name, "source_category": "manual_search",
            "search_query": query, "expected_artifact": "documented public geometry or metadata",
            "expected_geometry_type": "polygon/bbox" if target == "patch_boundary" else "observed polygon",
            "expected_file_format": "geojson|kml|kmz|zip|csv|gpkg", "source_public": "true",
            "download_allowed": "true", "must_attempt_download": "false", "current_status": "MANUAL_SEARCH_REQUIRED",
            "result_url": "", "local_raw_path": "", "blocking_reason": "NO_DIRECT_PUBLIC_URL_IDENTIFIED",
            "notes": "Use query and record exact public URL before promotion."})
    seen = {(r["target_type"], r["target_id"]) for r in rows}
    for source_file in ("v2ay_geometry_acquisition_targets.csv", "v2ba_external_source_acquisition_manifest.csv"):
        for row in load_csv(dataset_dir / source_file):
            target = clean(row.get("target_type"))
            target_id = clean(row.get("target_id"))
            if not target_id or (target, target_id) in seen:
                continue
            seen.add((target, target_id))
            rows.append({"search_id": stable_id("V2BB_SEARCH", target, target_id), "target_type": target, "target_id": target_id,
                "priority_package_id": config["priority_package_id"], "source_name": clean(row.get("source_name") or row.get("recommended_source_name")),
                "source_category": "diagnosed_target", "search_query": clean(row.get("suggested_search_terms") or target_id),
                "expected_artifact": clean(row.get("what_to_acquire") or row.get("expected_artifact")),
                "expected_geometry_type": clean(row.get("expected_geometry_type") or "polygon/bbox"),
                "expected_file_format": clean(row.get("accepted_formats") or "geojson|wkt|bbox"), "source_public": "true",
                "download_allowed": "true", "must_attempt_download": "false", "current_status": "MANUAL_SEARCH_REQUIRED",
                "result_url": clean(row.get("source_url_or_reference")), "local_raw_path": "",
                "blocking_reason": "NO_DIRECT_PUBLIC_URL_IDENTIFIED", "notes": "Imported diagnosed target."})
    return sorted(rows, key=lambda x: x["search_id"])


def safe_name(row):
    url_path = urllib.parse.urlparse(row["result_url"]).path
    suffix = Path(url_path).suffix or f".{row['expected_file_format'].split('|')[0]}"
    if row["source_category"] == "quickview_context":
        suffix = ".png"
    if row["source_category"] == "temporal_context":
        suffix = ".txt"
    return re.sub(r"[^a-z0-9]+", "_", row["source_name"].lower()).strip("_") + suffix


def download_sources(plan, dirs, config):
    dirs["raw_dir"].mkdir(parents=True, exist_ok=True)
    attempts = []
    for row in plan:
        if row["must_attempt_download"] != "true" or not row["result_url"]:
            continue
        path = dirs["raw_dir"] / safe_name(row)
        result = {"download_id": stable_id("V2BB_DL", row["search_id"], row["result_url"]), "search_id": row["search_id"],
            "target_type": row["target_type"], "target_id": row["target_id"], "source_name": row["source_name"],
            "url": row["result_url"], "attempted": "true", "success": "false", "http_status_or_error": "",
            "content_type": "", "file_size_bytes": "", "local_raw_path": "", "hash_sha256": "",
            "retrieved_at": date.today().isoformat(), "blocking_reason": "", "notes": ""}
        try:
            request = urllib.request.Request(row["result_url"], headers={"User-Agent": "REV-P-v2bb/1.0"})
            with urllib.request.urlopen(request, timeout=config["download_timeout_seconds"]) as response:
                limit = config["max_download_mb_per_file"] * 1024 * 1024
                content = response.read(limit + 1)
                if len(content) > limit:
                    raise ValueError("MAX_DOWNLOAD_SIZE_EXCEEDED")
                path.write_bytes(content)
                result.update({"success": "true", "http_status_or_error": str(response.status),
                    "content_type": clean(response.headers.get("Content-Type")), "file_size_bytes": str(len(content)),
                    "local_raw_path": path.relative_to(dirs["external_dir"]).as_posix(),
                    "hash_sha256": hashlib.sha256(content).hexdigest(), "notes": "Public source downloaded."})
        except Exception as exc:
            result["http_status_or_error"] = f"{type(exc).__name__}: {exc}"[:300]
            result["blocking_reason"] = "PUBLIC_DOWNLOAD_FAILED"
            result["notes"] = f"Manual command: Invoke-WebRequest -Uri '{row['result_url']}' -OutFile '{path}'"
        attempts.append(result)
    return attempts


def detected_format(path):
    suffix = path.suffix.lower()
    return {".geojson": "geojson", ".json": "json", ".csv": "csv", ".wkt": "wkt", ".kml": "kml",
            ".kmz": "kmz", ".zip": "shp_zip", ".gpkg": "gpkg", ".pdf": "pdf", ".jpg": "image"}.get(suffix, suffix.lstrip("."))


def inventory_raw(dirs, plan):
    by_path = {row["local_raw_path"]: row for row in plan if row.get("local_raw_path")}
    by_name = {safe_name(row): row for row in plan if row.get("result_url")}
    rows = []
    for path in sorted(dirs["raw_dir"].glob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(dirs["external_dir"]).as_posix()
        source = by_path.get(rel) or by_name.get(path.name, {})
        fmt = detected_format(path)
        parse = fmt in ("geojson", "json", "csv", "wkt")
        rows.append({"raw_file_id": stable_id("V2BB_RAW", rel), "target_type": clean(source.get("target_type")) or "unknown",
            "target_id": clean(source.get("target_id")), "source_name": clean(source.get("source_name")) or path.stem,
            "source_url_or_reference": clean(source.get("result_url")), "local_raw_path": rel, "file_name": path.name,
            "file_extension": path.suffix.lower(), "file_size_bytes": str(path.stat().st_size),
            "hash_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "detected_format": fmt,
            "source_public": "true", "access_status": "public_or_project_access", "can_parse": b(parse),
            "blocking_reason": "" if parse else "FORMAT_OR_CONTEXT_NOT_DIRECTLY_PARSEABLE",
            "notes": "Raw inventory does not imply event-observed or patch-boundary semantics."})
    return rows


def geojson_crs(obj):
    name = clean(obj.get("crs", {}).get("properties", {}).get("name"))
    match = re.search(r"EPSG[:/]{1,2}(\d+)", name, re.I)
    return f"EPSG:{match.group(1)}" if match else ""


def geometry_objects(path, fmt):
    if fmt in ("geojson", "json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        crs = geojson_crs(obj)
        if obj.get("type") == "FeatureCollection":
            result = []
            for feature in obj.get("features", []):
                geometry = dict(feature.get("geometry", {}))
                properties = feature.get("properties", {})
                for key in ("target_type", "target_id", "package_id", "crs"):
                    if properties.get(key):
                        geometry[f"_{key}"] = properties[key]
                result.append((geometry, clean(properties.get("crs")) or crs))
            return result
        if obj.get("type") == "Feature":
            geometry = dict(obj.get("geometry", {}))
            properties = obj.get("properties", {})
            for key in ("target_type", "target_id", "package_id", "crs"):
                if properties.get(key):
                    geometry[f"_{key}"] = properties[key]
            return [(geometry, clean(properties.get("crs")) or crs)]
        return [(obj, crs)]
    if fmt == "csv":
        result = []
        for row in load_csv(path):
            stype = clean(row.get("source_type") or row.get("geometry_format")).lower()
            value = clean(row.get("geometry_value") or row.get("wkt") or row.get("bbox"))
            if stype and value:
                result.append(({"_source_type": stype, "_value": value, "_target_type": clean(row.get("target_type")),
                                "_target_id": clean(row.get("target_id")), "_crs": clean(row.get("crs")),
                                "_package_id": clean(row.get("package_id"))}, clean(row.get("crs"))))
        return result
    if fmt == "wkt":
        return [({"_source_type": "wkt", "_value": path.read_text(encoding="utf-8").strip()}, "")]
    return []


def bbox_geojson(obj):
    coords = []
    def collect(value):
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(x, (int, float)) for x in value[:2]):
            coords.append(value[:2])
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(obj.get("coordinates", []))
    if not coords:
        return ["", "", "", ""]
    xs, ys = [p[0] for p in coords], [p[1] for p in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def extract_geometries(dirs, inventory, config, normalize=False):
    rows, accepted = [], set(config["accepted_crs"])
    for raw in inventory:
        path = dirs["external_dir"] / raw["local_raw_path"]
        for index, (obj, obj_crs) in enumerate(geometry_objects(path, raw["detected_format"])):
            source_type = clean(obj.get("_source_type")) or "geojson_inline"
            value = clean(obj.get("_value")) or json.dumps(obj, sort_keys=True, separators=(",", ":"))
            target_type = clean(obj.get("_target_type")) or raw["target_type"]
            target_id = clean(obj.get("_target_id")) or raw["target_id"]
            crs = normalise_crs(clean(obj.get("_crs")) or obj_crs)
            gtype, valid_parse = parse_geometry(source_type, value, "")
            point = gtype == "point"
            polygon = valid_parse and gtype in ("polygon", "bbox")
            role_ok = target_type in ("patch_boundary", "event_polygon")
            valid = polygon and role_ok and crs in accepted
            blockers = []
            if target_type == "event_context":
                blockers.append("CONTEXT_NOT_OBSERVED_EVENT_POLYGON")
            if point:
                blockers.append("POINT_ANCHOR_NOT_OVERLAY")
            if not polygon:
                blockers.append("GEOMETRY_NOT_POLYGON_OR_BBOX")
            if crs not in accepted:
                blockers.append("CRS_UNKNOWN_OR_UNACCEPTED")
            derived = ""
            if valid and normalize:
                name = "patch_boundary_REC_00019_normalized.geojson" if target_type == "patch_boundary" else "event_polygon_REC_2022_05_24_30_normalized.geojson"
                target = dirs["derived_dir"] / name
                target.parent.mkdir(parents=True, exist_ok=True)
                geometry = {key: value for key, value in obj.items() if not key.startswith("_")}
                geo = {"type": "Feature", "properties": {"target_type": target_type, "target_id": target_id,
                    "package_id": clean(obj.get("_package_id")) or config["priority_package_id"], "crs": crs},
                    "geometry": geometry}
                write_text(target, json.dumps(geo, indent=2))
                derived = target.relative_to(dirs["external_dir"]).as_posix()
            bounds = bbox_geojson(obj) if source_type == "geojson_inline" else ["", "", "", ""]
            rows.append({"public_geometry_id": stable_id("V2BB_GEOM", raw["raw_file_id"], index), "target_type": target_type,
                "target_id": target_id, "priority_package_id": clean(obj.get("_package_id")) or config["priority_package_id"],
                "source_name": raw["source_name"], "source_url_or_reference": raw["source_url_or_reference"],
                "local_raw_path": raw["local_raw_path"], "local_derived_path": derived, "geometry_role": target_type,
                "geometry_type": gtype or "unknown", "geometry_format": source_type, "crs": crs or "UNKNOWN",
                "crs_status": "ACCEPTED" if crs in accepted else "UNKNOWN_OR_UNACCEPTED", "geometry_valid": b(valid),
                "is_point": b(point), "is_polygon_or_bbox": b(polygon), "area_m2": "", "bbox_minx": bounds[0],
                "bbox_miny": bounds[1], "bbox_maxx": bounds[2], "bbox_maxy": bounds[3],
                "geometry_hash": hashlib.sha256(f"{source_type}|{value}|{crs}".encode()).hexdigest(),
                "source_public": "true", "access_status": "public_or_project_access",
                "provenance_note": f"Downloaded public source: {raw['source_name']}", "can_feed_v2ba": b(valid),
                "can_feed_v2az": b(valid), "can_feed_v2aw": b(valid), "can_feed_v2av": b(valid and target_type == "patch_boundary"),
                "can_feed_v2au": b(valid and target_type == "event_polygon"), "blocking_reason": "|".join(dict.fromkeys(blockers)),
                "notes": "Context, point and unknown-CRS geometry remain fail-closed."})
    return rows


def build_feeds(geometries, config):
    patch, event = [], []
    for row in geometries:
        if row["geometry_valid"] != "true":
            continue
        common = {"package_id": config["priority_package_id"], "source_public": "true",
            "access_status": "public_or_project_access", "source_url_or_reference": row["source_url_or_reference"],
            "local_derived_path": row["local_derived_path"], "geometry_format": row["geometry_format"], "crs": row["crs"],
            "geometry_hash": row["geometry_hash"], "provenance_note": row["provenance_note"], "review_status": "provided_unreviewed"}
        if row["target_type"] == "patch_boundary":
            patch.append(dict(common, feed_id=stable_id("V2BB_PATCH", row["public_geometry_id"]), patch_id=config["priority_patch_id"]))
        elif row["target_type"] == "event_polygon":
            event.append(dict(common, feed_id=stable_id("V2BB_EVENT", row["public_geometry_id"]), event_id=config["priority_event_id"]))
    pairs = []
    if patch and event:
        pairs.append({"pair_feed_id": stable_id("V2BB_PAIR", patch[0]["feed_id"], event[0]["feed_id"]),
            "package_id": config["priority_package_id"], "patch_id": config["priority_patch_id"], "event_id": config["priority_event_id"],
            "patch_feed_id": patch[0]["feed_id"], "event_feed_id": event[0]["feed_id"], "pair_ready": "true",
            "can_attempt_v2az_replay": "true", "blocking_reason": "", "notes": "TP4 still requires v2az/v2au replay and human review."})
    return patch, event, pairs


def readiness(config, patch, event, pairs):
    if pairs:
        level = "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY"
    elif patch:
        level = "TP1_ONE_PATCH_BOUNDARY_VALIDATED"
    elif event:
        level = "TP2_ONE_EVENT_POLYGON_VALIDATED"
    else:
        level = "TP0_DOCUMENTED_ABSENCE_WITH_PUBLIC_SEARCH_DOSSIER"
    return [{"readiness_id": stable_id("V2BB_READY", config["priority_package_id"]), "priority_package_id": config["priority_package_id"],
        "priority_patch_id": config["priority_patch_id"], "priority_event_id": config["priority_event_id"],
        "valid_patch_boundary_found": b(bool(patch)), "valid_event_polygon_found": b(bool(event)), "valid_pair_ready": b(bool(pairs)),
        "feed_v2ba_ready": b(bool(patch or event)), "feed_v2az_ready": b(bool(pairs)), "feed_v2aw_ready": b(bool(patch or event)),
        "feed_v2av_ready": b(bool(patch)), "feed_v2au_ready": b(bool(event)), "can_attempt_v2az_dry_run": b(bool(pairs)),
        "can_attempt_v2az_replay": b(bool(pairs)), "turning_point_level": level,
        "blocking_reason": "" if pairs else "REAL_PATCH_AND_EVENT_GEOMETRY_REQUIRED",
        "next_action": "Run v2az dry_run then controlled replay" if pairs else "Collect a proven patch boundary and observed-event polygon",
        "notes": "No replay readiness creates a label or final ground truth."}]


def update_autofill(dirs):
    paths = []
    specs = [
        ("v2bb_ready_patch_boundary_feed.csv", "patch_boundary_REC_00019/FILL_THIS_PATCH_BOUNDARY.autofill_candidate.csv", "patch_boundary", "REC_00019"),
        ("v2bb_ready_event_polygon_feed.csv", "event_polygon_REC_2022_05_24_30/FILL_THIS_EVENT_POLYGON.autofill_candidate.csv", "event_polygon", "REC_2022_05_24_30"),
    ]
    columns = "target_type target_id package_id source_type geometry_value geometry_path crs provenance_type provenance_note source_document source_public access_status review_status instructions".split()
    for feed_name, rel, target_type, target_id in specs:
        feeds = load_csv(dirs["dataset_dir"] / feed_name)
        if not feeds:
            continue
        row = feeds[0]
        target = dirs["external_dir"] / rel
        write_csv(target, columns, [{"target_type": target_type, "target_id": target_id, "package_id": row["package_id"],
            "source_type": "geojson_file", "geometry_path": row["local_derived_path"], "crs": row["crs"],
            "provenance_type": "public_source", "provenance_note": row["provenance_note"],
            "source_document": row["source_url_or_reference"], "source_public": "true",
            "access_status": "public_or_project_access", "review_status": "provided_unreviewed",
            "instructions": "Human review required before use."}])
        paths.append(target)
    return paths


def schema(columns):
    bools = {x for x in columns if x.startswith("can_") or x.startswith("valid_") or x.startswith("feed_")}
    bools |= {"source_public", "download_allowed", "must_attempt_download", "attempted", "success", "geometry_valid",
              "is_point", "is_polygon_or_bbox", "pair_ready", "turning_point_ready"}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "required": list(columns),
            "additionalProperties": False, "properties": {x: {"type": "boolean" if x in bools else "string"} for x in columns}}


def write_support(dirs, config, plan, attempts, inventory, geometries, patch, event, pairs, ready):
    for name, columns in TABLES.items():
        rows = {"v2bb_public_search_plan.csv": plan, "v2bb_public_download_attempts.csv": attempts,
            "v2bb_public_raw_file_inventory.csv": inventory, "v2bb_extracted_public_geometry_registry.csv": geometries,
            "v2bb_ready_patch_boundary_feed.csv": patch, "v2bb_ready_event_polygon_feed.csv": event,
            "v2bb_ready_turning_point_pair_feed.csv": pairs, "v2bb_replay_readiness_update.csv": ready}[name]
        write_csv(dirs["dataset_dir"] / name, columns, rows)
        write_text(dirs["dataset_dir"] / "schemas" / f"{Path(name).stem}.schema.json", json.dumps(schema(columns), indent=2))
    write_text(dirs["docs_dir"] / "v2bb_public_geometry_retrieval_feed_builder.md",
        "# v2bb public geometry retrieval feed builder\n\nv2bb extends v2ba with executable public downloads. Public access is not blocked by license, but provenance, semantics and CRS remain mandatory. Valid public geometry is normalized and fed forward; context and quickviews never become labels.\n")
    search_lines = "\n".join(f"- {r['source_name']}: {r['current_status']} {r['result_url'] or r['search_query']}" for r in plan)
    write_text(dirs["docs_dir"] / "v2bb_public_source_search_log.md", f"# v2bb public source search log\n\n{search_lines}\n\nDownloads attempted: {len(attempts)}; successful: {sum(x['success']=='true' for x in attempts)}.\n")
    write_text(dirs["docs_dir"] / "v2bb_after_public_geometry_found.md",
        "# After public geometry is found\n\nRun v2bb `normalize`, `build_feeds`, and the autofill updater. Then run v2ba validation and v2az `dry_run`; use v2az `replay` only after review. Stop at `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`.\n")
    summary = {"stage": "v2bb_public_geometry_retrieval_feed_builder", "status": "OK_WITH_EXPECTED_BLOCKERS",
        "priority_package_id": config["priority_package_id"], "priority_patch_id": config["priority_patch_id"],
        "priority_event_id": config["priority_event_id"], "web_downloads_allowed": bool(config["allow_web_downloads"]),
        "download_all_diagnosed_sources": True, "public_sources_searched": len(plan), "download_attempts": len(attempts),
        "successful_downloads": sum(x["success"] == "true" for x in attempts), "raw_files_found": len(inventory),
        "candidate_geometries_found": len(geometries), "valid_patch_boundaries": len(patch), "valid_event_polygons": len(event),
        "ready_patch_feed_rows": len(patch), "ready_event_feed_rows": len(event), "ready_pair_feed_rows": len(pairs),
        "turning_point_level": ready[0]["turning_point_level"], "turning_point_ready": bool(pairs),
        "can_attempt_v2az_replay": bool(pairs), "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "PUBLIC_SEARCH_COMPLETE_WAITING_FOR_PROVEN_REAL_GEOMETRY_NOT_FOR_TRAINING" if not pairs else "PUBLIC_PAIR_READY_FOR_CONTROLLED_REPLAY_NOT_FOR_TRAINING"}
    write_text(dirs["output_dir"] / "execution_reports" / "v2bb_public_geometry_retrieval_feed_builder_summary.json", json.dumps(summary, indent=2))
    write_text(dirs["output_dir"] / "execution_reports" / "v2bb_public_geometry_retrieval_feed_builder_report.md",
        f"# v2bb report\n\nSources: {len(plan)}; attempts: {len(attempts)}; downloads: {summary['successful_downloads']}; raw files: {len(inventory)}; valid patch/event/pair: {len(patch)}/{len(event)}/{len(pairs)}; turning point: `{summary['turning_point_level']}`.\n\nNo model, label, final ground truth, invented geometry or automatic C4 was created.\n")
    write_text(dirs["output_dir"] / "logs_summary" / "v2bb_public_geometry_retrieval_feed_builder.txt",
        "\n".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in summary.items()))
    return summary


def run(mode="search_plan", dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dirs = resolve_dirs(dataset_dir, output_dir, config_dir, external_dir, docs_dir)
    config = load_config(dirs["config_dir"])
    dirs["raw_dir"].mkdir(parents=True, exist_ok=True)
    dirs["derived_dir"].mkdir(parents=True, exist_ok=True)
    plan = build_search_plan(dirs["dataset_dir"], config)
    attempts = load_csv(dirs["dataset_dir"] / "v2bb_public_download_attempts.csv")
    if mode in ("download_public", "full") and config["allow_web_downloads"]:
        attempts = download_sources(plan, dirs, config)
        by_id = {x["search_id"]: x for x in attempts if x["success"] == "true"}
        for row in plan:
            if row["search_id"] in by_id:
                row["current_status"] = "DOWNLOADED_PUBLIC_SOURCE"
                row["local_raw_path"] = by_id[row["search_id"]]["local_raw_path"]
    inventory = inventory_raw(dirs, plan)
    normalize = mode in ("normalize", "build_feeds", "replay_dry_run", "full")
    geometries = extract_geometries(dirs, inventory, config, normalize=normalize)
    patch, event, pairs = build_feeds(geometries, config)
    ready = readiness(config, patch, event, pairs)
    summary = write_support(dirs, config, plan, attempts, inventory, geometries, patch, event, pairs, ready)
    if mode in ("build_feeds", "replay_dry_run", "full"):
        update_autofill(dirs)
    print(f"[v2bb] mode={mode} sources={len(plan)} attempts={len(attempts)} downloads={summary['successful_downloads']}")
    print(f"[v2bb] raw={len(inventory)} candidates={len(geometries)} valid_patch={len(patch)} valid_event={len(event)} pairs={len(pairs)}")
    print(f"[v2bb] turning_point={summary['turning_point_level']} can_attempt_v2az_replay={str(bool(pairs)).lower()}")
    return 0, summary


if __name__ == "__main__":
    raise SystemExit(run()[0])

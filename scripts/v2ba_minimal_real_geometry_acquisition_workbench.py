#!/usr/bin/env python3
"""v2ba: fail-closed acquisition workbench for the first real Recife geometry pair."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

try:
    from scripts.v2aw_geometry_source_intake_engine import normalise_crs, parse_geometry
except ImportError:
    from v2aw_geometry_source_intake_engine import normalise_crs, parse_geometry


ROOT = Path(__file__).resolve().parents[1]
MODES = ("source_scan", "ingest", "validate", "replay_ready_check")
CONFIG_NAME = "v2ba_minimal_real_geometry_acquisition_workbench_config.json"

DEFAULT_CONFIG = {
    "offline_mode": True, "allow_web_downloads": False,
    "allow_manual_external_files": True, "allow_public_source_ingestion": True,
    "priority_region": "Recife", "priority_patch_id": "REC_00019",
    "priority_event_id": "REC_2022_05_24_30", "priority_package_id": "PKG_34713b8aab96",
    "external_source_dir": "datasets/external_sources/recife_minimal_tp",
    "accepted_crs": ["EPSG:4326", "EPSG:3857", "EPSG:31982", "EPSG:31983"],
    "accepted_patch_geometry_formats": ["bbox", "wkt", "geojson", "geojson_file", "kml", "gpkg", "shp_zip"],
    "accepted_event_geometry_formats": ["wkt", "geojson", "geojson_file", "kml", "gpkg", "shp_zip"],
    "allow_point_as_patch_boundary": False, "allow_point_as_event_polygon": False,
    "allow_quickview_as_verified_product": False, "allow_auto_geometry_generation": False,
    "can_train_model": False, "can_create_operational_labels": False,
}

MANIFEST_COLUMNS = ("acquisition_source_id target_type target_id priority_package_id region city "
    "source_category source_name source_url_or_reference expected_artifact expected_geometry_type "
    "expected_crs_metadata source_public access_status provenance_requirement can_promote_alone "
    "current_status local_path blocking_reason recommended_action notes").split()
SEARCH_COLUMNS = ("query_id target_type target_id source_name query_goal suggested_query expected_result "
    "expected_geometry manual_download_needed download_destination source_public access_status "
    "validation_after_download notes").split()
INVENTORY_COLUMNS = ("external_file_id target_type target_id file_path file_name file_extension file_size_bytes "
    "detected_format geometry_candidate crs_detected crs_value hash_sha256 source_public access_status "
    "can_parse can_validate_geometry blocking_reason notes").split()
GEOMETRY_COLUMNS = ("candidate_geometry_id target_type target_id source_file_id geometry_role geometry_type "
    "geometry_format crs crs_status geometry_valid is_point is_polygon_or_bbox area_m2 bbox_minx bbox_miny "
    "bbox_maxx bbox_maxy geometry_hash source_public access_status provenance_note can_feed_v2aw can_feed_v2av "
    "can_feed_v2au blocking_reason notes").split()
PATCH_ADAPTER_COLUMNS = ("intake_id patch_id region city priority_rank package_count required_geometry_kind "
    "source_type geometry_value geometry_path crs provenance_type provenance_note source_document "
    "source_document_page source_document_url_or_path digitized_by digitized_at source_confidence review_status "
    "validation_status blocking_reason notes source_public access_status next_action").split()
EVENT_ADAPTER_COLUMNS = ("event_intake_id event_id region city hazard_type linked_packages_count "
    "required_geometry_kind source_type geometry_value geometry_path crs event_geometry_role source_id "
    "source_name provenance_type provenance_note source_document source_document_page source_document_url_or_path "
    "digitized_by digitized_at source_confidence review_status validation_status blocking_reason notes "
    "source_public access_status next_action").split()
PATCH_FEED_COLUMNS = ("feed_id patch_id package_id region city source_type geometry_value geometry_path crs "
    "provenance_note source_public access_status source_file hash_sha256").split()
EVENT_FEED_COLUMNS = ("feed_id event_id package_id region city geometry_role source_type geometry_value "
    "geometry_path crs provenance_note source_public access_status source_file hash_sha256").split()
PAIR_FEED_COLUMNS = ("pair_feed_id package_id patch_id event_id patch_feed_id event_feed_id pair_ready "
    "can_attempt_v2az_replay blocking_reason notes").split()
GATE_COLUMNS = ("gate_id turning_point_level gate_name required_condition observed_value gate_passed severity "
    "blocking_reason recommended_action notes").split()
FILL_COLUMNS = ("target_type target_id package_id source_type geometry_value geometry_path crs provenance_type "
    "provenance_note source_document source_public access_status review_status instructions").split()

OUTPUT_TABLES = {
    "v2ba_external_source_acquisition_manifest.csv": MANIFEST_COLUMNS,
    "v2ba_external_search_and_download_plan.csv": SEARCH_COLUMNS,
    "v2ba_external_file_inventory.csv": INVENTORY_COLUMNS,
    "v2ba_candidate_geometry_registry.csv": GEOMETRY_COLUMNS,
    "v2ba_minimal_candidate_patch_intake_adapter.csv": PATCH_ADAPTER_COLUMNS,
    "v2ba_minimal_candidate_event_intake_adapter.csv": EVENT_ADAPTER_COLUMNS,
    "v2ba_ready_patch_boundary_feed.csv": PATCH_FEED_COLUMNS,
    "v2ba_ready_event_polygon_feed.csv": EVENT_FEED_COLUMNS,
    "v2ba_ready_turning_point_pair_feed.csv": PAIR_FEED_COLUMNS,
    "v2ba_minimal_tp_acquisition_gate.csv": GATE_COLUMNS,
}


def clean(value):
    return str(value or "").strip()


def b(value):
    return "true" if value else "false"


def stable_id(prefix, *parts):
    digest = hashlib.sha256("|".join(clean(x) for x in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_dirs(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dataset = Path(dataset_dir or os.getenv("DATASET_DIR") or ROOT / "datasets").resolve()
    output = Path(output_dir or os.getenv("OUTPUT_DIR") or ROOT / "outputs_public").resolve()
    config = Path(config_dir or os.getenv("CONFIG_DIR") or ROOT / "configs").resolve()
    external_env = external_dir or os.getenv("EXTERNAL_DIR")
    external = Path(external_env).resolve() if external_env else dataset / "external_sources" / "recife_minimal_tp"
    return {"dataset_dir": dataset, "output_dir": output, "config_dir": config,
            "external_dir": external, "docs_dir": Path(docs_dir or ROOT / "docs").resolve()}


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


def load_config(config_dir):
    config = dict(DEFAULT_CONFIG)
    path = config_dir / CONFIG_NAME
    if path.is_file():
        config.update(json.loads(path.read_text(encoding="utf-8")))
    config["accepted_crs"] = [normalise_crs(x) for x in config["accepted_crs"]]
    return config


def ensure_external_structure(external_dir):
    paths = [
        external_dir, external_dir / "patch_boundary_REC_00019",
        external_dir / "event_polygon_REC_2022_05_24_30", external_dir / "source_documents",
        external_dir / "raw", external_dir / "derived",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    write_text(external_dir / "README.md", """# Recife minimal turning-point external intake

Place the real `REC_00019` patch boundary under `patch_boundary_REC_00019/` and the real observed
`REC_2022_05_24_30` event polygon under `event_polygon_REC_2022_05_24_30/`.

Accepted intake formats are bbox, WKT, GeoJSON, KML, GPKG and zipped SHP. Geometry, explicit CRS,
source reference and provenance note are mandatory. These sources are treated as public/project-accessible;
access and origin remain auditable. Nothing in this directory becomes a label automatically.
""")
    return paths


def generate_fill_files(external_dir):
    ensure_external_structure(external_dir)
    rows = [
        (external_dir / "patch_boundary_REC_00019" / "FILL_THIS_PATCH_BOUNDARY.csv", {
            "target_type": "patch_boundary", "target_id": "REC_00019", "package_id": "PKG_34713b8aab96",
            "source_type": "missing", "crs": "UNKNOWN", "source_public": "true",
            "access_status": "public_or_project_access", "review_status": "not_started",
            "instructions": "Fill geometry_value or geometry_path, CRS, provenance_note and source_document.",
        }),
        (external_dir / "event_polygon_REC_2022_05_24_30" / "FILL_THIS_EVENT_POLYGON.csv", {
            "target_type": "event_polygon", "target_id": "REC_2022_05_24_30", "package_id": "PKG_34713b8aab96",
            "source_type": "missing", "crs": "UNKNOWN", "source_public": "true",
            "access_status": "public_or_project_access", "review_status": "not_started",
            "instructions": "Fill observed polygon geometry_value or geometry_path, CRS, provenance_note and source_document.",
        }),
    ]
    for path, row in rows:
        write_csv(path, FILL_COLUMNS, [row])
    return [path for path, _ in rows]


def build_manifest(config):
    patch, event, package = config["priority_patch_id"], config["priority_event_id"], config["priority_package_id"]
    specs = [
        ("patch_boundary", patch, "patch_metadata", "Sentinel patch generation metadata", "verified vector or bounds", "polygon or bbox", True),
        ("patch_boundary", patch, "local_gis", "Local GIS export", "verified vector export", "polygon or bbox", True),
        ("patch_boundary", patch, "manual_digitization", "VHR/manual digitization with provenance", "auditable digitized vector", "polygon", False),
        ("event_polygon", event, "operational_mapping", "Charter/EMS operational product", "verified operational vector product", "polygon", True),
        ("event_polygon", event, "manual_digitization", "VHR/manual digitization with provenance", "auditable observed-event polygon", "polygon", False),
        ("event_polygon", event, "support_anchor", "SGB/CPRM", "context/point anchor", "point", False),
        ("event_polygon", event, "temporal_context", "ANA/INMET/Cemaden", "temporal/context evidence", "none", False),
        ("event_polygon", event, "context_only", "Media/EM-DAT/social", "context evidence", "none", False),
    ]
    rows = []
    for target_type, target_id, category, name, artifact, geom, promote in specs:
        strong = promote and geom != "point"
        rows.append({
            "acquisition_source_id": stable_id("V2BA_SRC", target_type, name), "target_type": target_type,
            "target_id": target_id, "priority_package_id": package, "region": "Recife", "city": "Recife",
            "source_category": category, "source_name": name, "source_url_or_reference": "",
            "expected_artifact": artifact, "expected_geometry_type": geom, "expected_crs_metadata": "explicit CRS",
            "source_public": "true", "access_status": "not_acquired",
            "provenance_requirement": "source reference and provenance note required",
            "can_promote_alone": b(strong), "current_status": "PLANNED_NOT_ACQUIRED", "local_path": "",
            "blocking_reason": "REAL_GEOMETRY_FILE_REQUIRED",
            "recommended_action": "Acquire public/project-accessible source and run v2ba validation",
            "notes": "Point/context/quickview cannot close an observed polygon or patch boundary.",
        })
    return rows


def build_search_plan(config):
    external = config["external_source_dir"].replace("\\", "/")
    specs = [
        ("event_polygon", config["priority_event_id"], "Recife May 2022 flood/landslide", "Recife maio 2022 inundacao deslizamento vetor"),
        ("event_polygon", config["priority_event_id"], "Charter/EMS", "Charter EMS Recife May 2022 flood landslide Brazil vector"),
        ("event_polygon", config["priority_event_id"], "SGB/CPRM", "SGB CPRM Recife 2022 pontos eventos"),
        ("event_polygon", config["priority_event_id"], "Operational vector products", "Recife May 2022 flood extent shapefile geojson"),
        ("patch_boundary", config["priority_patch_id"], "Sentinel patch metadata", "REC_00019 patch footprint Sentinel metadata CRS"),
    ]
    return [{
        "query_id": stable_id("V2BA_QUERY", target, source), "target_type": target, "target_id": target_id,
        "source_name": source, "query_goal": "Acquire real geometry or authoritative metadata",
        "suggested_query": query, "expected_result": "public/project-accessible documented artifact",
        "expected_geometry": "polygon or bbox" if target == "patch_boundary" else "observed polygon",
        "manual_download_needed": b(not config["allow_web_downloads"]),
        "download_destination": f"{external}/raw/", "source_public": "true", "access_status": "not_acquired",
        "validation_after_download": "python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate",
        "notes": "No automatic download while allow_web_downloads=false.",
    } for target, target_id, source, query in specs]


def infer_target(path, config, row=None):
    text = str(path).lower()
    row = row or {}
    target = clean(row.get("target_type")).lower()
    target_id = clean(row.get("target_id"))
    if target in ("patch_boundary", "patch") or config["priority_patch_id"].lower() in text:
        return "patch_boundary", target_id or config["priority_patch_id"]
    if target in ("event_polygon", "event") or config["priority_event_id"].lower() in text:
        return "event_polygon", target_id or config["priority_event_id"]
    return "unknown", target_id


def source_files(external_dir):
    ignored_names = {"README.md", "FILL_THIS_PATCH_BOUNDARY.csv", "FILL_THIS_EVENT_POLYGON.csv"}
    ignored_parts = {"derived"}
    return sorted(path for path in external_dir.rglob("*") if path.is_file()
                  and path.name not in ignored_names and not ignored_parts.intersection(path.relative_to(external_dir).parts))


def detect_format(path):
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        return "geojson_file"
    if suffix in (".wkt",):
        return "wkt"
    if suffix == ".kml":
        return "kml"
    if suffix == ".gpkg":
        return "gpkg"
    if suffix == ".zip":
        return "shp_zip"
    if suffix == ".csv":
        return "csv"
    return suffix.lstrip(".") or "unknown"


def inventory_files(external_dir, config):
    rows = []
    for path in source_files(external_dir):
        fmt = detect_format(path)
        target_type, target_id = infer_target(path, config)
        parseable = fmt in ("geojson_file", "wkt", "csv")
        rows.append({
            "external_file_id": stable_id("V2BA_FILE", path.relative_to(external_dir)),
            "target_type": target_type, "target_id": target_id,
            "file_path": path.relative_to(external_dir).as_posix(), "file_name": path.name,
            "file_extension": path.suffix.lower(), "file_size_bytes": str(path.stat().st_size),
            "detected_format": fmt, "geometry_candidate": b(fmt in ("geojson_file", "wkt", "csv", "kml", "gpkg", "shp_zip")),
            "crs_detected": "false", "crs_value": "", "hash_sha256": sha256(path),
            "source_public": "true", "access_status": "public_or_project_access",
            "can_parse": b(parseable), "can_validate_geometry": b(parseable),
            "blocking_reason": "" if parseable else "FORMAT_REQUIRES_EXTERNAL_GIS_VALIDATION",
            "notes": "Inventory does not imply geometry validity.",
        })
    return rows


def candidate_records(path, inventory_row):
    fmt = inventory_row["detected_format"]
    if fmt == "csv":
        return load_csv(path)
    if fmt == "geojson_file":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            crs = clean(obj.get("crs", {}).get("properties", {}).get("name"))
            return [{"source_type": "geojson_inline", "geometry_value": json.dumps(obj, sort_keys=True),
                     "crs": crs, "provenance_note": f"External GeoJSON: {path.name}"}]
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
    if fmt == "wkt":
        try:
            return [{"source_type": "wkt", "geometry_value": path.read_text(encoding="utf-8").strip(),
                     "crs": "", "provenance_note": f"External WKT: {path.name}"}]
        except UnicodeDecodeError:
            return []
    return []


def bbox_values(source_type, value):
    if source_type == "bbox":
        try:
            vals = [float(x.strip()) for x in value.split(",")]
            return vals if len(vals) == 4 else ["", "", "", ""]
        except ValueError:
            pass
    return ["", "", "", ""]


def extract_geometries(external_dir, inventory, config, mode):
    rows = []
    accepted = set(config["accepted_crs"])
    by_rel = {row["file_path"]: row for row in inventory}
    for rel, inv in by_rel.items():
        path = external_dir / rel
        for index, source in enumerate(candidate_records(path, inv)):
            target_type, target_id = infer_target(path, config, source)
            source_type = clean(source.get("source_type") or source.get("geometry_format")).lower()
            if source_type == "geojson":
                source_type = "geojson_inline"
            value = clean(source.get("geometry_value"))
            crs = normalise_crs(source.get("crs"))
            gtype, valid = parse_geometry(source_type, value, "")
            point = gtype == "point"
            polygon = gtype in ("polygon", "bbox") and valid
            role_valid = polygon and target_type in ("patch_boundary", "event_polygon")
            crs_ok = crs in accepted
            valid_geometry = role_valid and crs_ok
            blockers = []
            if not value:
                blockers.append("GEOMETRY_MISSING")
            if point:
                blockers.append("POINT_NOT_POLYGON_OR_BOUNDARY")
            if not valid or not polygon:
                blockers.append("GEOMETRY_INVALID_OR_UNSUPPORTED")
            if not crs_ok:
                blockers.append("CRS_MISSING_OR_UNACCEPTED")
            if target_type == "unknown":
                blockers.append("TARGET_TYPE_UNKNOWN")
            candidate_id = stable_id("V2BA_GEOM", inv["external_file_id"], index, target_type, target_id)
            bounds = bbox_values(source_type, value)
            geom_hash = hashlib.sha256(f"{source_type}|{value}|{crs}".encode("utf-8")).hexdigest() if value else ""
            row = {
                "candidate_geometry_id": candidate_id, "target_type": target_type, "target_id": target_id,
                "source_file_id": inv["external_file_id"], "geometry_role": target_type,
                "geometry_type": gtype or "unknown", "geometry_format": source_type, "crs": crs or "UNKNOWN",
                "crs_status": "ACCEPTED" if crs_ok else "UNKNOWN_OR_UNACCEPTED",
                "geometry_valid": b(valid_geometry), "is_point": b(point), "is_polygon_or_bbox": b(polygon),
                "area_m2": "", "bbox_minx": bounds[0], "bbox_miny": bounds[1],
                "bbox_maxx": bounds[2], "bbox_maxy": bounds[3], "geometry_hash": geom_hash,
                "source_public": clean(source.get("source_public")) or "true",
                "access_status": clean(source.get("access_status")) or "public_or_project_access",
                "provenance_note": clean(source.get("provenance_note")),
                "can_feed_v2aw": b(valid_geometry), "can_feed_v2av": b(valid_geometry and target_type == "patch_boundary"),
                "can_feed_v2au": b(valid_geometry and target_type == "event_polygon"),
                "blocking_reason": "|".join(dict.fromkeys(blockers)), "notes": "No geometry is inferred from text.",
                "_source_type": source_type, "_geometry_value": value, "_geometry_path": "",
                "_source_file": rel, "_hash_sha256": inv["hash_sha256"],
                "_package_id": clean(source.get("package_id")) or config["priority_package_id"],
            }
            rows.append(row)
            if valid_geometry and mode == "ingest":
                destination = external_dir / "derived" / path.name
                if path.resolve() != destination.resolve():
                    shutil.copy2(path, destination)
    return rows


def adapters(config):
    shared = {"region": "Recife", "city": "Recife", "source_type": "missing", "geometry_value": "",
              "geometry_path": "", "crs": "UNKNOWN", "provenance_type": "unknown", "provenance_note": "",
              "source_document": "", "source_public": "true", "access_status": "public_or_project_access",
              "review_status": "not_started", "validation_status": "BLOCKED_PENDING_REAL_GEOMETRY",
              "blocking_reason": "REAL_GEOMETRY_REQUIRED",
              "notes": "No geometry is auto-generated.",
              "next_action": "Fill real geometry, CRS, provenance note and source document; run v2ba validate."}
    patch = dict(shared, intake_id=stable_id("V2BA_PATCH", config["priority_patch_id"]),
                 patch_id=config["priority_patch_id"], priority_rank="1", package_count="1",
                 required_geometry_kind="patch_boundary_polygon")
    event = dict(shared, event_intake_id=stable_id("V2BA_EVENT", config["priority_event_id"]),
                 event_id=config["priority_event_id"], hazard_type="urban_flood", linked_packages_count="55",
                 required_geometry_kind="observed_event_polygon", event_geometry_role="observed_event_polygon")
    return [patch], [event]


def build_feeds(geometries, config):
    patch_rows, event_rows = [], []
    for row in geometries:
        if row["geometry_valid"] != "true":
            continue
        common = {
            "package_id": row["_package_id"], "region": "Recife", "city": "Recife",
            "source_type": row["_source_type"], "geometry_value": row["_geometry_value"],
            "geometry_path": row["_geometry_path"], "crs": row["crs"], "provenance_note": row["provenance_note"],
            "source_public": row["source_public"], "access_status": row["access_status"],
            "source_file": row["_source_file"], "hash_sha256": row["_hash_sha256"],
        }
        if row["target_type"] == "patch_boundary":
            patch_rows.append(dict(common, feed_id=stable_id("V2BA_PATCH_FEED", row["candidate_geometry_id"]),
                                   patch_id=config["priority_patch_id"]))
        elif row["target_type"] == "event_polygon":
            event_rows.append(dict(common, feed_id=stable_id("V2BA_EVENT_FEED", row["candidate_geometry_id"]),
                                   event_id=config["priority_event_id"], geometry_role="observed_event_polygon"))
    patch_rows.sort(key=lambda x: x["feed_id"])
    event_rows.sort(key=lambda x: x["feed_id"])
    pairs = []
    matching_patch = next((row for row in patch_rows if row["package_id"] == config["priority_package_id"]), None)
    matching_event = next((row for row in event_rows if row["package_id"] == config["priority_package_id"]), None)
    if matching_patch and matching_event:
        pairs.append({
            "pair_feed_id": stable_id("V2BA_PAIR", matching_patch["feed_id"], matching_event["feed_id"]),
            "package_id": config["priority_package_id"], "patch_id": config["priority_patch_id"],
            "event_id": config["priority_event_id"], "patch_feed_id": matching_patch["feed_id"],
            "event_feed_id": matching_event["feed_id"], "pair_ready": "true",
            "can_attempt_v2az_replay": "true", "blocking_reason": "",
            "notes": "Pair is ready for v2az dry-run/replay; TP4 still requires confirmed v2au overlay.",
        })
    return patch_rows, event_rows, pairs


def build_gates(patch_rows, event_rows, pair_rows):
    patch = bool(patch_rows)
    event = bool(event_rows)
    pair = bool(pair_rows)
    specs = [
        ("TP1", "TP1_PATCH_BOUNDARY_FILE_PRESENT", patch, "Real patch boundary file"),
        ("TP1", "TP1_PATCH_BOUNDARY_CRS_VALID", patch, "Accepted explicit patch CRS"),
        ("TP1", "TP1_PATCH_BOUNDARY_GEOMETRY_VALID", patch, "Valid polygon/bbox patch boundary"),
        ("TP2", "TP2_EVENT_POLYGON_FILE_PRESENT", event, "Real observed-event polygon file"),
        ("TP2", "TP2_EVENT_POLYGON_CRS_VALID", event, "Accepted explicit event CRS"),
        ("TP2", "TP2_EVENT_POLYGON_GEOMETRY_VALID", event, "Valid observed-event polygon"),
        ("TP3", "TP3_PATCH_EVENT_PAIR_LINKED", pair, "Patch and event linked to minimal package"),
        ("TP3", "TP3_PAIR_FEED_READY", pair, "Pair feed ready"),
        ("TP4", "TP4_REPLAY_READY", pair, "Pair available for v2az replay"),
        ("TP4", "TP4_OVERLAY_NOT_YET_CONFIRMED", False, "Confirmed v2au overlay plus human review"),
        ("GUARDRAIL", "NO_LABEL_CREATED", True, "No operational label"),
        ("GUARDRAIL", "NO_MODEL_TRAINED", True, "No model training"),
        ("GUARDRAIL", "NO_GEOMETRY_INVENTED", True, "Only supplied files parsed"),
    ]
    return [{
        "gate_id": stable_id("V2BA_GATE", name), "turning_point_level": level, "gate_name": name,
        "required_condition": requirement, "observed_value": b(passed), "gate_passed": b(passed),
        "severity": "BLOCKER" if not passed else "PASS",
        "blocking_reason": "" if passed else ("OVERLAY_REPLAY_AND_HUMAN_REVIEW_REQUIRED" if "OVERLAY" in name else "REAL_GEOMETRY_REQUIRED"),
        "recommended_action": "Preserve guardrail" if passed else "Supply/validate real geometry or run controlled replay",
        "notes": "Maximum permitted decision is C4_CANDIDATE_REQUIRES_HUMAN_REVIEW.",
    } for level, name, passed, requirement in specs]


def turning_level(patch_rows, event_rows, pair_rows):
    if pair_rows:
        return "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY"
    if patch_rows and event_rows:
        return "TP2_ONE_EVENT_POLYGON_VALIDATED"
    if patch_rows:
        return "TP1_ONE_PATCH_BOUNDARY_VALIDATED"
    if event_rows:
        return "TP2_ONE_EVENT_POLYGON_VALIDATED"
    return "TP0_DOCUMENTED_ABSENCE_WITH_ACQUISITION_DOSSIER"


def schema(columns):
    bools = {"source_public", "can_promote_alone", "manual_download_needed", "geometry_candidate",
             "crs_detected", "can_parse", "can_validate_geometry", "geometry_valid", "is_point",
             "is_polygon_or_bbox", "can_feed_v2aw", "can_feed_v2av", "can_feed_v2au", "pair_ready",
             "can_attempt_v2az_replay", "gate_passed"}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "required": list(columns), "additionalProperties": False,
            "properties": {col: {"type": "boolean" if col in bools else "string"} for col in columns}}


def write_schemas(dataset_dir):
    for name, columns in OUTPUT_TABLES.items():
        path = dataset_dir / "schemas" / f"{Path(name).stem}.schema.json"
        write_text(path, json.dumps(schema(columns), indent=2))


def write_docs(docs_dir):
    write_text(docs_dir / "v2ba_minimal_real_geometry_acquisition_workbench.md", """# v2ba minimal real geometry acquisition workbench

v2ba targets the audit-selected minimal pair `REC_00019` + `REC_2022_05_24_30` in
`PKG_34713b8aab96`. TP1 needs a real patch boundary; TP2 needs a real observed-event polygon;
TP3 needs both linked; TP4 needs a confirmed v2au replay and human review.

The workbench receives public/project-accessible external files, inventories them, requires explicit
CRS and provenance, and emits feeds only after technical validation. No real geometry is currently
present. Absence stays an explicit blocker and never becomes a label.
""")
    write_text(docs_dir / "v2ba_how_to_fill_REC_00019_and_REC_2022_05_24_30.md", """# Fill the minimal Recife pair

Put a real patch boundary in `datasets/external_sources/recife_minimal_tp/patch_boundary_REC_00019/`
or fill `FILL_THIS_PATCH_BOUNDARY.csv`. Put a real observed-event polygon in
`event_polygon_REC_2022_05_24_30/` or fill `FILL_THIS_EVENT_POLYGON.csv`.

Fill `source_type`, either `geometry_value` or `geometry_path`, explicit `crs`, `provenance_note`,
`source_document`, `source_public`, `access_status`, and `review_status`. Accepted examples are bbox
`minx,miny,maxx,maxy`, polygon WKT, and polygon GeoJSON. Run v2ba `validate`, then v2az `dry_run`;
use v2az `replay` only when feeds are valid.
""")
    write_text(docs_dir / "v2ba_source_quality_policy.md", """# v2ba source quality policy

Official sources and validated operational products may support promotion after geometry, CRS, semantics,
access, and provenance checks. A quickview, media, benchmark, social source, EM-DAT record, temporal
series, or CPRM point is context only and cannot close an observed polygon or patch boundary.

License is not a blocker in v2ba because inputs are treated as public/project-accessible. Origin,
source documentation, access status and provenance remain mandatory. Missing CRS or provenance blocks.
""")
    write_text(docs_dir / "v2ba_turning_point_execution_after_fill.md", """# Execution after filling

1. Place the real external file and provenance.
2. Run `python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate`.
3. If feeds exist, run `python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run`.
4. Run v2az `replay` only after reviewing inputs.
5. Review v2au output.
6. Stop at `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`; create no label.
""")


def build_summary(config, inventory, geometries, patch_rows, event_rows, pair_rows):
    level = turning_level(patch_rows, event_rows, pair_rows)
    return {
        "stage": "v2ba_minimal_real_geometry_acquisition_workbench", "status": "OK_WITH_EXPECTED_BLOCKERS",
        "priority_package_id": config["priority_package_id"], "priority_patch_id": config["priority_patch_id"],
        "priority_event_id": config["priority_event_id"], "external_files_found": len(inventory),
        "candidate_geometries_found": len(geometries), "valid_patch_boundaries": len(patch_rows),
        "valid_event_polygons": len(event_rows), "ready_patch_feed_rows": len(patch_rows),
        "ready_event_feed_rows": len(event_rows), "ready_pair_feed_rows": len(pair_rows),
        "turning_point_level": level, "turning_point_ready": bool(pair_rows),
        "can_attempt_v2az_replay": bool(pair_rows), "can_train_model": False,
        "can_create_operational_labels": False,
        "methodological_status": "MINIMAL_REAL_GEOMETRY_ACQUISITION_WORKBENCH_READY_WAITING_FOR_EXTERNAL_FILES"
        if not pair_rows else "MINIMAL_REAL_GEOMETRY_PAIR_READY_FOR_CONTROLLED_REPLAY_NOT_FOR_TRAINING",
    }


def write_public(output_dir, summary):
    report = f"""# v2ba minimal real geometry acquisition workbench report

- Status: `{summary['status']}`
- External files found: `{summary['external_files_found']}`
- Candidate geometries: `{summary['candidate_geometries_found']}`
- Valid patch boundaries: `{summary['valid_patch_boundaries']}`
- Valid event polygons: `{summary['valid_event_polygons']}`
- Turning point: `{summary['turning_point_level']}`
- Can attempt v2az replay: `{str(summary['can_attempt_v2az_replay']).lower()}`

No geometry is invented. No model, operational label, final ground truth, or automatic C4 is created.
"""
    write_text(output_dir / "execution_reports" / "v2ba_minimal_real_geometry_acquisition_workbench_report.md", report)
    write_text(output_dir / "execution_reports" / "v2ba_minimal_real_geometry_acquisition_workbench_summary.json",
               json.dumps(summary, indent=2))
    write_text(output_dir / "logs_summary" / "v2ba_minimal_real_geometry_acquisition_workbench.txt",
               "\n".join(f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in summary.items()))


def run(mode="source_scan", dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    if mode not in MODES:
        raise ValueError(f"Unsupported v2ba mode: {mode}")
    dirs = resolve_dirs(dataset_dir, output_dir, config_dir, external_dir, docs_dir)
    config = load_config(dirs["config_dir"])
    ensure_external_structure(dirs["external_dir"])
    generate_fill_files(dirs["external_dir"])
    manifest = build_manifest(config)
    search = build_search_plan(config)
    inventory = inventory_files(dirs["external_dir"], config)
    geometries = extract_geometries(dirs["external_dir"], inventory, config, mode)
    patch_adapter, event_adapter = adapters(config)
    patch_rows, event_rows, pair_rows = build_feeds(geometries, config)
    gates = build_gates(patch_rows, event_rows, pair_rows)
    outputs = {
        "v2ba_external_source_acquisition_manifest.csv": manifest,
        "v2ba_external_search_and_download_plan.csv": search,
        "v2ba_external_file_inventory.csv": inventory,
        "v2ba_candidate_geometry_registry.csv": geometries,
        "v2ba_minimal_candidate_patch_intake_adapter.csv": patch_adapter,
        "v2ba_minimal_candidate_event_intake_adapter.csv": event_adapter,
        "v2ba_ready_patch_boundary_feed.csv": patch_rows,
        "v2ba_ready_event_polygon_feed.csv": event_rows,
        "v2ba_ready_turning_point_pair_feed.csv": pair_rows,
        "v2ba_minimal_tp_acquisition_gate.csv": gates,
    }
    for name, rows in outputs.items():
        write_csv(dirs["dataset_dir"] / name, OUTPUT_TABLES[name], rows)
    write_schemas(dirs["dataset_dir"])
    write_docs(dirs["docs_dir"])
    summary = build_summary(config, inventory, geometries, patch_rows, event_rows, pair_rows)
    write_public(dirs["output_dir"], summary)
    print(f"[v2ba] mode={mode} external_files={len(inventory)} candidates={len(geometries)}")
    print(f"[v2ba] valid_patch={len(patch_rows)} valid_event={len(event_rows)} pairs={len(pair_rows)}")
    print(f"[v2ba] turning_point={summary['turning_point_level']} replay={str(bool(pair_rows)).lower()}")
    print("[v2ba] can_train_model=false can_create_operational_labels=false")
    return 0, summary


def main():
    code, _ = run()
    raise SystemExit(code)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v2bi Recife Charter and temporal manual intake, fail-closed."""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import zipfile

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2bi_recife_charter_temporal_intake")
CACHE_DIR = os.environ.get("V2BI_CACHE_DIR", os.path.join(DOCS_DIR, "evidence_cache"))
CHARTER_CACHE = os.path.join(CACHE_DIR, "manual_charter_758")
TEMPORAL_CACHE = os.path.join(CACHE_DIR, "manual_temporal")
NETWORK_ENABLED = os.environ.get("V2BI_NETWORK", "0") == "1"
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false", "can_create_label": "false",
    "can_create_negative": "false", "can_train_model": "false",
    "charter_vector_is_not_final_ground_truth": "true", "charter_geometry_requires_human_review": "true",
    "temporal_series_is_not_spatial_geometry": "true", "rainfall_is_not_flood_extent": "true",
    "landslide_geometry_is_not_flood_extent": "true", "crs_missing_blocks_geometry_promotion": "true",
    "raw_data_versioned": "false",
}
INPUTS = {
    "inventory": "v2bh_charter_758_product_inventory.csv", "classification": "v2bh_recife_olinda_product_classification.csv",
    "access": "v2bh_product_access_vector_crs_license_audit.csv", "hazard": "v2bh_product_hazard_geometry_type_classification.csv",
    "candidate": "v2bh_candidate_geometry_source_registry.csv", "gates": "v2bh_recife_gate_status_update.csv",
    "tasks": "v2bh_manual_access_request_tasks.csv", "v2bg_gates": "v2bg_recife_protocol_gate_status.csv",
    "gap_selection": "v2bg_recife_gap_package_selection.csv", "temporal_readiness": "v2ay_event_patch_temporal_readiness_update.csv",
    "prior_metrics": "v2ay_window_precipitation_metrics.csv", "recife_queue": "v2az_recife_gap_review_queue.csv",
}
OUTPUTS = [
    "v2bi_manual_intake_cache_inventory.csv", "v2bi_charter_file_audit.csv", "v2bi_charter_vector_metadata.csv",
    "v2bi_charter_crs_geometry_validation.csv", "v2bi_charter_candidate_geometry_readiness.csv",
    "v2bi_temporal_series_cache_inventory.csv", "v2bi_apac_cemaden_series_parse_report.csv",
    "v2bi_recife_temporal_metrics.csv", "v2bi_recife_protocol_gate_update.csv", "v2bi_manual_blocker_tasks.csv",
    "v2bi_guardrail_regression.csv",
]
TYPE_MAP = {".zip": "ZIP", ".shp": "SHP", ".geojson": "GEOJSON", ".json": "GEOJSON", ".kml": "KML", ".gpkg": "GPKG",
            ".pdf": "PDF", ".png": "PNG", ".jpg": "JPG", ".jpeg": "JPG", ".csv": "CSV", ".xlsx": "XLSX",
            ".xls": "XLSX", ".html": "HTML", ".htm": "HTML", ".txt": "CSV"}
VECTOR_TYPES = {"SHP", "GEOJSON", "KML", "GPKG"}
MAP_TYPES = {"PDF", "PNG", "JPG"}


def parse_args(argv=None): return argparse.ArgumentParser().parse_args(argv)
def dataset_path(name): return os.path.join(DATASET_DIR, name)
def doc_path(*parts): return os.path.join(DOCS_DIR, *parts)
def with_invariants(row): return {**row, **INVARIANTS}
def is_true(value): return str(value or "").strip().lower() == "true"
def clean(value): return str(value or "").strip()


def load_csv(path):
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows: raise ValueError(f"Refusing empty output: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle: handle.write(text)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def by(rows, key): return {r.get(key, ""): r for r in rows}
def detect_type(path): return TYPE_MAP.get(os.path.splitext(path)[1].lower(), "UNKNOWN")


def source_candidate(name):
    value = clean(name).upper()
    if "APAC" in value: return "APAC"
    if "CEMADEN" in value: return "CEMADEN"
    if "HIDROWEB" in value or "ANA" in value: return "ANA_HIDROWEB"
    if "INMET" in value: return "INMET_PROXY"
    return "UNKNOWN"


def cache_files(folder):
    if not os.path.isdir(folder): return []
    return sorted(path for root, _, names in os.walk(folder) for name in names
                  if name != ".gitignore" for path in [os.path.join(root, name)])


def inventory_file(path, group, number):
    kind = detect_type(path)
    return with_invariants({
        "cache_item_id": f"CACHE_v2bi_{number:04d}", "cache_group": group, "file_path": path.replace("\\", "/"),
        "file_name": os.path.basename(path), "file_extension": os.path.splitext(path)[1].lower(),
        "file_size_bytes": str(os.path.getsize(path)), "sha256": sha256(path), "detected_type": kind,
        "usable_for_intake": str(kind != "UNKNOWN").lower(), "note": "Manual raw payload remains in ignored cache.",
    })


def run_scan_manual_intake_cache(args=None):
    rows = []; files = [(p, "CHARTER_758") for p in cache_files(CHARTER_CACHE)] + [(p, "TEMPORAL_APAC_CEMADEN") for p in cache_files(TEMPORAL_CACHE)]
    for number, (path, group) in enumerate(files, 1): rows.append(inventory_file(path, group, number))
    if not rows: rows = [with_invariants({"cache_item_id": "CACHE_v2bi_0000", "cache_group": "OTHER", "file_path": "", "file_name": "", "file_extension": "", "file_size_bytes": "0", "sha256": "", "detected_type": "UNKNOWN", "usable_for_intake": "false", "note": "NO_MANUAL_INTAKE_FILES_FOUND"})]
    write_csv(dataset_path(OUTPUTS[0]), rows); return rows


def archive_summary(path):
    try:
        with zipfile.ZipFile(path) as archive: return "|".join(archive.namelist()), any(n.lower().endswith((".shp", ".geojson", ".json", ".kml", ".gpkg")) for n in archive.namelist())
    except (zipfile.BadZipFile, OSError): return "ZIP_READ_FAILED", False


def charter_audit_status(kind, vector=False):
    if vector: return "VECTOR_CANDIDATE_FOUND"
    if kind == "PDF": return "MAP_ONLY_FOUND"
    if kind in {"PNG", "JPG"}: return "PREVIEW_ONLY_FOUND"
    return "UNSUPPORTED_FILE"


def run_audit_charter_files(args=None):
    rows = []
    for item in load_csv(dataset_path(OUTPUTS[0])):
        if item["cache_group"] != "CHARTER_758": continue
        kind, path = item["detected_type"], item["file_path"]; summary, zip_vector = ("", False)
        if kind == "ZIP": summary, zip_vector = archive_summary(path)
        vector = kind in VECTOR_TYPES or zip_vector; map_found = kind in MAP_TYPES
        status = charter_audit_status(kind, vector)
        rows.append(with_invariants({"charter_file_audit_id": f"CHAUD_v2bi_{len(rows)+1:03d}", "product_id": "CH758_RECIFE_20220602_001",
            "file_path": path, "file_type": kind, "archive_contents_summary": summary, "vector_candidate_found": str(vector).lower(),
            "raster_or_map_found": str(map_found).lower(), "pdf_or_image_found": str(map_found).lower(), "metadata_file_found": "false",
            "audit_status": status, "note": "Candidate detection only; geometry, CRS, license, and municipal scope require validation."}))
    if not rows: rows = [with_invariants({"charter_file_audit_id": "CHAUD_v2bi_000", "product_id": "CH758_RECIFE_20220602_001", "file_path": "", "file_type": "", "archive_contents_summary": "", "vector_candidate_found": "false", "raster_or_map_found": "false", "pdf_or_image_found": "false", "metadata_file_found": "false", "audit_status": "NO_MANUAL_CHARTER_FILE_FOUND", "note": "BLOCKED_MANUAL_ACCESS_REQUIRED"})]
    write_csv(dataset_path(OUTPUTS[1]), rows); return rows


def flatten_coords(value):
    if not isinstance(value, list): return []
    if len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]): return [(float(value[0]), float(value[1]))]
    result = []
    for child in value: result.extend(flatten_coords(child))
    return result


def geojson_metadata(path):
    try:
        with open(path, encoding="utf-8-sig") as handle: data = json.load(handle)
        features = data.get("features", []) if data.get("type") == "FeatureCollection" else [data] if data.get("type") == "Feature" else []
        coords, types, fields = [], set(), set()
        for feature in features:
            geometry = feature.get("geometry") or {}; types.add(clean(geometry.get("type"))); coords.extend(flatten_coords(geometry.get("coordinates")))
            fields.update((feature.get("properties") or {}).keys())
        bbox = [min(v[i] for v in coords) for i in (0, 1)] + [max(v[i] for v in coords) for i in (0, 1)] if coords else ["", "", "", ""]
        crs = data.get("crs") or {}; crs_value = clean((crs.get("properties") or {}).get("name"))
        return {"crs": crs_value, "count": len(features), "geometry": "|".join(sorted(t for t in types if t)), "bbox": bbox, "fields": "|".join(sorted(fields)), "status": "VECTOR_METADATA_EXTRACTED" if coords else "READ_FAILED"}
    except (OSError, ValueError, TypeError) as exc:
        return {"crs": "", "count": 0, "geometry": "", "bbox": ["", "", "", ""], "fields": "", "status": "READ_FAILED", "error": type(exc).__name__}


def run_extract_charter_vector_metadata(args=None):
    rows = []
    for audit in load_csv(dataset_path(OUTPUTS[1])):
        if not is_true(audit["vector_candidate_found"]): continue
        path, kind = audit["file_path"], audit["file_type"]
        meta = geojson_metadata(path) if kind == "GEOJSON" else {"crs": "", "count": 0, "geometry": "", "bbox": ["", "", "", ""], "fields": "", "status": "CRS_NOT_DETECTED"}
        rows.append(with_invariants({"product_id": audit["product_id"], "vector_file_detected": "true", "vector_file_type": kind, "layer_name": "",
            "crs_detected": str(bool(meta["crs"])).lower(), "crs_value": meta["crs"], "feature_count": str(meta["count"]), "geometry_type": meta["geometry"],
            "bbox_minx": str(meta["bbox"][0]), "bbox_miny": str(meta["bbox"][1]), "bbox_maxx": str(meta["bbox"][2]), "bbox_maxy": str(meta["bbox"][3]),
            "attribute_fields_summary": meta["fields"], "metadata_status": meta["status"]}))
    if not rows: rows = [with_invariants({"product_id": "CH758_RECIFE_20220602_001", "vector_file_detected": "false", "vector_file_type": "", "layer_name": "", "crs_detected": "false", "crs_value": "", "feature_count": "0", "geometry_type": "", "bbox_minx": "", "bbox_miny": "", "bbox_maxx": "", "bbox_maxy": "", "attribute_fields_summary": "", "metadata_status": "VECTOR_NOT_AVAILABLE"})]
    write_csv(dataset_path(OUTPUTS[2]), rows); return rows


def bbox_checks(meta):
    try:
        minx, miny, maxx, maxy = [float(meta[k]) for k in ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy")]
    except (ValueError, TypeError): return False, False
    brazil = -75 <= minx <= -30 and -35 <= miny <= 6 and -75 <= maxx <= -30 and -35 <= maxy <= 6
    recife = maxx >= -35.1 and minx <= -34.7 and maxy >= -8.2 and miny <= -7.8
    return brazil, recife


def validation_status(meta):
    if not is_true(meta["vector_file_detected"]): return "NOT_AVAILABLE"
    if not clean(meta["geometry_type"]) or int(meta.get("feature_count") or 0) < 1: return "GEOMETRY_MISSING"
    if not is_true(meta["crs_detected"]): return "CRS_MISSING"
    brazil, recife = bbox_checks(meta)
    if not brazil or not recife: return "OUTSIDE_EXPECTED_AREA"
    return "VALID_FOR_HUMAN_REVIEW"


def run_validate_charter_crs_geometry(args=None):
    rows = []
    for meta in load_csv(dataset_path(OUTPUTS[2])):
        brazil, recife = bbox_checks(meta); status = validation_status(meta)
        rows.append(with_invariants({"product_id": meta["product_id"], "crs_present": meta["crs_detected"], "crs_value": meta["crs_value"],
            "geometry_present": str(bool(clean(meta["geometry_type"]) and int(meta.get("feature_count") or 0) > 0)).lower(), "geometry_type": meta["geometry_type"],
            "feature_count": meta["feature_count"], "bbox_within_brazil": str(brazil).lower(), "bbox_intersects_recife_context": str(recife).lower(),
            "geometry_validity_status": status, "validation_note": "Human review only; validation does not create ground truth."}))
    write_csv(dataset_path(OUTPUTS[3]), rows); return rows


def run_build_charter_candidate_geometry_readiness(args=None):
    previous = next(r for r in load_csv(dataset_path(INPUTS["candidate"])) if r["product_id"] == "CH758_RECIFE_20220602_001")
    validation = load_csv(dataset_path(OUTPUTS[3]))[0]; audit = load_csv(dataset_path(OUTPUTS[1]))[0]
    if validation["geometry_validity_status"] == "VALID_FOR_HUMAN_REVIEW": status, reason, action = "CANDIDATE_GEOMETRY_READY_FOR_HUMAN_REVIEW", "Vector, CRS, geometry and Recife bbox validated.", "HUMAN_REVIEW_CHARTER_RECIFE_CANDIDATE_GEOMETRY"
    elif audit["audit_status"] == "MAP_ONLY_FOUND": status, reason, action = "MAP_ONLY_REVIEW", "Map/PDF found without validated vector.", "REQUEST_CHARTER_VECTOR_CRS_LICENSE"
    elif audit["audit_status"] == "PREVIEW_ONLY_FOUND": status, reason, action = "PREVIEW_ONLY_NOT_READY", "Preview found without validated vector.", "REQUEST_CHARTER_VECTOR_CRS_LICENSE"
    elif audit["audit_status"] == "NO_MANUAL_CHARTER_FILE_FOUND": status, reason, action = "NO_FILE_AVAILABLE", "No manual Charter product file available.", "MANUALLY_DOWNLOAD_CHARTER_758_PRODUCT_AND_APAC_CEMADEN_SERIES"
    else: status, reason, action = "PENDING_VECTOR_CRS", f"Validation status: {validation['geometry_validity_status']}.", "REQUEST_OR_MANUALLY_ACCESS_CHARTER_PRODUCT_VECTOR_CRS"
    rows = [with_invariants({"product_id": previous["product_id"], "recife_package_id": previous["recife_package_id"], "event_patch_package_id": previous["event_patch_package_id"],
        "previous_candidate_status": previous["candidate_status"], "updated_candidate_status": status, "readiness_reason": reason, "required_next_action": action})]
    write_csv(dataset_path(OUTPUTS[4]), rows); return rows


def run_scan_temporal_series_cache(args=None):
    rows = []
    for number, path in enumerate(cache_files(TEMPORAL_CACHE), 1):
        kind = detect_type(path)
        rows.append(with_invariants({"temporal_file_id": f"TEMP_v2bi_{number:03d}", "source_candidate": source_candidate(os.path.basename(path)),
            "file_path": path.replace("\\", "/"), "file_name": os.path.basename(path), "file_type": kind, "sha256": sha256(path),
            "usable_for_parse": str(kind in {"CSV", "XLSX", "ZIP"}).lower(), "note": "Manual temporal payload remains in ignored cache."}))
    if not rows: rows = [with_invariants({"temporal_file_id": "TEMP_v2bi_000", "source_candidate": "UNKNOWN", "file_path": "", "file_name": "", "file_type": "", "sha256": "", "usable_for_parse": "false", "note": "NO_TEMPORAL_SERIES_FOUND"})]
    write_csv(dataset_path(OUTPUTS[5]), rows); return rows


def normalize_key(value): return re.sub(r"[^a-z0-9]", "", clean(value).lower().replace("ç", "c").replace("ã", "a").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o"))
def find_column(fields, terms): return next((f for f in fields if any(term in normalize_key(f) for term in terms)), "")


def parse_number(value):
    raw = clean(value).replace(" ", "")
    if "," in raw and "." not in raw: raw = raw.replace(",", ".")
    elif "," in raw and "." in raw: raw = raw.replace(".", "").replace(",", ".")
    try: return float(raw)
    except ValueError: return None


def parse_datetime(value, hour=""):
    raw = f"{clean(value)} {clean(hour)}".strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try: return dt.datetime.strptime(raw, fmt)
        except ValueError: pass
    try: return dt.datetime.fromisoformat(raw)
    except ValueError: return None


def parse_temporal_text(text, source="UNKNOWN"):
    try:
        try: dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;")
        except csv.Error: dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect); fields = reader.fieldnames or []
        date_col = find_column(fields, ("timestamp", "datahora", "datetime", "data", "date"))
        hour_col = find_column(fields, ("hora", "hour")) if "timestamp" not in normalize_key(date_col) else ""
        precip_col = find_column(fields, ("precipitacao", "precipitation", "chuva", "acumulado", "rain"))
        station_col = find_column(fields, ("estacao", "station", "posto"))
        municipality_col = find_column(fields, ("municipio", "cidade", "municipality"))
        if not date_col or not precip_col: return [], {"status": "UNSUPPORTED_SCHEMA", "date": date_col, "precip": precip_col, "station": station_col}
        rows = []
        for raw in reader:
            timestamp, value = parse_datetime(raw.get(date_col), raw.get(hour_col)) if hour_col else parse_datetime(raw.get(date_col)), parse_number(raw.get(precip_col))
            if timestamp is not None and value is not None:
                rows.append({"timestamp": timestamp, "precipitation": value, "station": clean(raw.get(station_col)), "municipality": clean(raw.get(municipality_col)), "source": source})
        return rows, {"status": "PARSED" if rows else "READ_FAILED", "date": date_col, "precip": precip_col, "station": station_col}
    except (OSError, csv.Error):
        return [], {"status": "READ_FAILED", "date": "", "precip": "", "station": ""}


def parse_temporal_csv(path, source="UNKNOWN"):
    try:
        return parse_temporal_text(open(path, "rb").read().decode("utf-8-sig", errors="replace"), source)
    except OSError:
        return [], {"status": "READ_FAILED", "date": "", "precip": "", "station": ""}


def parse_temporal_file(path, kind, source="UNKNOWN"):
    if kind == "CSV": return parse_temporal_csv(path, source)
    if kind == "XLSX":
        try:
            import openpyxl
            sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).active
            values = list(sheet.iter_rows(values_only=True))
            if not values: return [], {"status": "UNSUPPORTED_SCHEMA", "date": "", "precip": "", "station": ""}
            stream = io.StringIO(); writer = csv.writer(stream); writer.writerows(values)
            return parse_temporal_text(stream.getvalue(), source)
        except (ImportError, OSError, ValueError, zipfile.BadZipFile):
            return [], {"status": "READ_FAILED", "date": "", "precip": "", "station": ""}
    if kind == "ZIP":
        try:
            rows, infos = [], []
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if not name.lower().endswith((".csv", ".txt")): continue
                    parsed, info = parse_temporal_text(archive.read(name).decode("utf-8-sig", errors="replace"), source)
                    rows.extend(parsed); infos.append(info)
            if rows:
                info = next((i for i in infos if i["status"] == "PARSED"), infos[0])
                return rows, info
            return [], infos[0] if infos else {"status": "UNSUPPORTED_SCHEMA", "date": "", "precip": "", "station": ""}
        except (OSError, zipfile.BadZipFile):
            return [], {"status": "READ_FAILED", "date": "", "precip": "", "station": ""}
    return [], {"status": "READ_FAILED", "date": "", "precip": "", "station": ""}


def run_parse_apac_cemaden_series(args=None):
    rows = []; parsed_store = {}
    for item in load_csv(dataset_path(OUTPUTS[5])):
        if not is_true(item["usable_for_parse"]): continue
        parsed, info = parse_temporal_file(item["file_path"], item["file_type"], item["source_candidate"])
        parsed_store[item["temporal_file_id"]] = parsed
        rows.append(with_invariants({"temporal_file_id": item["temporal_file_id"], "source_candidate": item["source_candidate"], "parse_status": info["status"],
            "records_parsed": str(len(parsed)), "stations_detected": str(len({r["station"] for r in parsed if r["station"]})),
            "date_min": min((r["timestamp"].isoformat() for r in parsed), default=""), "date_max": max((r["timestamp"].isoformat() for r in parsed), default=""),
            "precipitation_column_detected": info["precip"], "timestamp_column_detected": info["date"], "station_column_detected": info["station"],
            "note": "Defensive parse; temporal series does not prove spatial geometry."}))
    if not rows: rows = [with_invariants({"temporal_file_id": "TEMP_v2bi_000", "source_candidate": "UNKNOWN", "parse_status": "NO_TEMPORAL_SERIES_FOUND", "records_parsed": "0", "stations_detected": "0", "date_min": "", "date_max": "", "precipitation_column_detected": "", "timestamp_column_detected": "", "station_column_detected": "", "note": "BLOCKED_MANUAL_ACCESS_REQUIRED"})]
    write_csv(dataset_path(OUTPUTS[6]), rows); return rows


def rolling_24h(rows):
    maximum = None
    for i, current in enumerate(rows):
        total = sum(r["precipitation"] for r in rows if current["timestamp"] - dt.timedelta(hours=23, minutes=59) <= r["timestamp"] <= current["timestamp"])
        maximum = total if maximum is None else max(maximum, total)
    return maximum


def temporal_metrics(parsed, start, end):
    selected = sorted([r for r in parsed if start <= r["timestamp"].date() <= end], key=lambda r: r["timestamp"])
    if not selected: return {"count": 0, "stations": 0, "total": "", "max1h": "", "max24h": "", "missing": "1.000", "status": "NO_SERIES_AVAILABLE"}
    values = [r["precipitation"] for r in selected]; expected = ((end - start).days + 1) * 24
    missing = max(0, 1 - len(values) / expected); status = "TEMPORAL_EVIDENCE_READY_FOR_REVIEW" if missing <= 0.2 else "TEMPORAL_EVIDENCE_PARTIAL"
    return {"count": len(values), "stations": len({r["station"] for r in selected if r["station"]}), "total": f"{sum(values):.3f}", "max1h": f"{max(values):.3f}", "max24h": f"{rolling_24h(selected):.3f}", "missing": f"{missing:.3f}", "status": status}


def all_parsed_temporal():
    result = []
    for item in load_csv(dataset_path(OUTPUTS[5])):
        if is_true(item["usable_for_parse"]):
            parsed, _ = parse_temporal_file(item["file_path"], item["file_type"], item["source_candidate"]); result.extend(parsed)
    return result


def run_compute_recife_temporal_metrics(args=None):
    parsed = all_parsed_temporal(); rows = []
    for packet in load_csv(dataset_path(INPUTS["recife_queue"])):
        metric = temporal_metrics(parsed, dt.date.fromisoformat(packet["window_start"]), dt.date.fromisoformat(packet["window_end"]))
        source = next((r["source"] for r in parsed), "UNKNOWN"); station = next((r["station"] for r in parsed if r["station"]), "")
        rows.append(with_invariants({"recife_package_id": packet["review_packet_id"], "event_patch_package_id": packet["event_patch_package_id"], "event_date": packet["event_date"],
            "window_start": packet["window_start"], "window_end": packet["window_end"], "source_candidate": source, "station_id": station, "station_name": station,
            "station_role": "LOCAL" if station and any(clean(r["municipality"]).upper() == "RECIFE" for r in parsed) else "UNKNOWN",
            "records_in_window": str(metric["count"]), "station_count": str(metric["stations"]), "precip_total_window": metric["total"], "precip_max_1h": metric["max1h"],
            "precip_max_24h": metric["max24h"], "missing_rate": metric["missing"], "temporal_status": metric["status"]}))
    write_csv(dataset_path(OUTPUTS[7]), rows); return rows


def run_update_recife_protocol_gates(args=None):
    readiness = load_csv(dataset_path(OUTPUTS[4]))[0]; metrics = by(load_csv(dataset_path(OUTPUTS[7])), "event_patch_package_id"); rows = []
    for previous in load_csv(dataset_path(INPUTS["gates"])):
        metric = metrics.get(previous["event_patch_package_id"], {}); status = previous["updated_gate_status"]; evidence = previous["evidence_used"]; blocker = previous["blocker_remaining"]; reason = "No new intake evidence; previous fail-closed status retained."
        if previous["candidate_id"] == "REC_2022_05_24_30":
            if previous["previous_gate_id"] == "C3_SPATIAL_ANCHOR": status, reason = "PASS", "Charter Recife product remains confirmed."
            elif previous["previous_gate_id"] == "C4_CANDIDATE_GEOMETRY":
                status = "PASS_FOR_HUMAN_REVIEW_ONLY" if readiness["updated_candidate_status"] == "CANDIDATE_GEOMETRY_READY_FOR_HUMAN_REVIEW" else "PENDING_VECTOR_CRS"
                reason, blocker = readiness["readiness_reason"], "" if status.startswith("PASS") else "VALIDATED_VECTOR_CRS_GEOMETRY_MISSING"
            elif previous["previous_gate_id"] in {"C1_TEMPORALITY", "C2_VALID_SERIES_OR_STATION"} and metric.get("temporal_status") == "TEMPORAL_EVIDENCE_READY_FOR_REVIEW":
                status, evidence, reason, blocker = "PASS_FOR_REVIEW", f"{metric.get('source_candidate')}:{previous['event_patch_package_id']}", "Parsed temporal series covers the event window.", ""
            elif previous["previous_gate_id"] == "C7_FINAL_GROUND_TRUTH": status, blocker = "BLOCKED", "FINAL_GROUND_TRUTH_BLOCKED"
        rows.append(with_invariants({"recife_package_id": previous["recife_package_id"], "event_patch_package_id": previous["event_patch_package_id"], "candidate_id": previous["candidate_id"],
            "gate_id": previous["previous_gate_id"], "previous_status": previous["updated_gate_status"], "updated_status": status, "evidence_used": evidence,
            "update_reason": reason, "blocker_remaining": blocker,
            "next_action_rank_1": "HUMAN_REVIEW_CHARTER_RECIFE_CANDIDATE_GEOMETRY" if readiness["updated_candidate_status"] == "CANDIDATE_GEOMETRY_READY_FOR_HUMAN_REVIEW" else "MANUALLY_DOWNLOAD_CHARTER_758_PRODUCT_AND_APAC_CEMADEN_SERIES"}))
    write_csv(dataset_path(OUTPUTS[8]), rows); return rows


def run_build_manual_blocker_tasks(args=None):
    definitions = [
        ("CHARTER_VECTOR_CRS_ACCESS", "Download or request the Recife Charter product vector and CRS.", "CHARTER_VECTOR_WITH_CRS_METADATA", "C4_CANDIDATE_GEOMETRY"),
        ("APAC_CEMADEN_SERIES_ACCESS", "Download APAC/Cemaden Recife May 2022 series into manual_temporal.", "PARSEABLE_LOCAL_TEMPORAL_SERIES", "C1_TEMPORALITY|C2_VALID_SERIES_OR_STATION"),
        ("FEATURE_TYPE_CONFIRMATION", "Confirm mapped Charter feature semantics.", "FEATURE_TYPE_REVIEW_RECORD", "C4_CANDIDATE_GEOMETRY|C5_HUMAN_REVIEW"),
        ("LICENSE_TERMS_CONFIRMATION", "Confirm Charter product license and redistribution terms.", "LICENSE_TERMS_RECORD", "C4_CANDIDATE_GEOMETRY"),
        ("HUMAN_REVIEW_REQUIRED", "Human-review any validated Charter geometry.", "HUMAN_REVIEW_DECISION", "C5_HUMAN_REVIEW|C6_CANDIDATE_REFERENCE"),
    ]
    rows = [with_invariants({"task_id": f"TASK_v2bi_{i:03d}", "blocker_type": kind, "required_action": action, "expected_artifact": artifact,
        "priority": "P0" if i <= 2 else "P1", "blocks_gate": gate, "resolved_by_file_or_metadata": "false"}) for i, (kind, action, artifact, gate) in enumerate(definitions, 1)]
    write_csv(dataset_path(OUTPUTS[9]), rows); return rows


def run_generate_recife_intake_report(args=None):
    readiness = load_csv(dataset_path(OUTPUTS[4]))[0]; temporal = load_csv(dataset_path(OUTPUTS[6]))[0]
    overall = "BLOCKED_MANUAL_ACCESS_REQUIRED" if readiness["updated_candidate_status"] != "CANDIDATE_GEOMETRY_READY_FOR_HUMAN_REVIEW" and temporal["parse_status"] != "PARSED" else "INTAKE_EVIDENCE_AVAILABLE_FOR_REVIEW"
    write_text(doc_path("README.md"), f"""# v2bi Recife Charter and Temporal Intake

Eu/equipe executou uma etapa de intake, nao de promocao final. Status geral: `{overall}`.

Sem arquivo manual, Charter permanece sem vetor/CRS validado e APAC/Cemaden permanece sem serie parseada. Com vetor, CRS e geometria valida, C4 pode avancar apenas para human review. Com serie APAC/Cemaden/ANA valida na janela, C1/C2 podem avancar para review.

C3 permanece PASS para maio de 2022. C7 permanece BLOCKED. Ground truth final, labels, negativos e treino continuam zero.
""")
    write_text(doc_path("intake_reports", "recife_may_2022.md"), f"# Recife May 2022 Intake Report\n\nOverall: `{overall}`\n\nCharter: `{readiness['updated_candidate_status']}`.\nTemporal: `{temporal['parse_status']}`.\nNext action: `MANUALLY_DOWNLOAD_CHARTER_758_PRODUCT_AND_APAC_CEMADEN_SERIES`.\n")
    for name in (OUTPUTS[1], OUTPUTS[2], OUTPUTS[3], OUTPUTS[4]): write_csv(doc_path("charter_audit", name), load_csv(dataset_path(name)))
    for name in (OUTPUTS[5], OUTPUTS[6], OUTPUTS[7]): write_csv(doc_path("temporal_series_audit", name), load_csv(dataset_path(name)))
    return [{"overall_status": overall}]


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}; rows = []
    for number, name in enumerate(OUTPUTS[:10], 1):
        violations = sum(r.get(field, "").lower() == "true" for r in load_csv(dataset_path(name)) for field in forbidden)
        rows.append({"regression_id": f"GR_v2bi_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    c7 = [r for r in load_csv(dataset_path(OUTPUTS[8])) if r["gate_id"] == "C7_FINAL_GROUND_TRUTH"]; passed = c7 and all(r["updated_status"] == "BLOCKED" for r in c7)
    rows.append({"regression_id": "GR_v2bi_011", "artifact_path": "C7_FINAL_GROUND_TRUTH", "violation_count": "0" if passed else "1", "status": "PASS" if passed else "FAIL"})
    if any(r["status"] != "PASS" for r in rows): raise ValueError("v2bi guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[10]), rows); return rows


STEPS = [
    ("scan_manual_intake_cache", run_scan_manual_intake_cache, OUTPUTS[0]), ("audit_charter_files", run_audit_charter_files, OUTPUTS[1]),
    ("extract_charter_vector_metadata", run_extract_charter_vector_metadata, OUTPUTS[2]), ("validate_charter_crs_geometry", run_validate_charter_crs_geometry, OUTPUTS[3]),
    ("build_charter_candidate_geometry_readiness", run_build_charter_candidate_geometry_readiness, OUTPUTS[4]),
    ("scan_temporal_series_cache", run_scan_temporal_series_cache, OUTPUTS[5]), ("parse_apac_cemaden_series", run_parse_apac_cemaden_series, OUTPUTS[6]),
    ("compute_recife_temporal_metrics", run_compute_recife_temporal_metrics, OUTPUTS[7]), ("update_recife_protocol_gates", run_update_recife_protocol_gates, OUTPUTS[8]),
    ("build_manual_blocker_tasks", run_build_manual_blocker_tasks, OUTPUTS[9]), ("generate_recife_intake_report", run_generate_recife_intake_report, None),
    ("run_guardrail_regression", run_guardrail_regression, OUTPUTS[10]),
]


def ensure_structure():
    for folder in ("intake_reports", "charter_audit", "temporal_series_audit", "evidence_cache", "evidence_cache/manual_charter_758", "evidence_cache/manual_temporal"): os.makedirs(doc_path(folder), exist_ok=True)
    for folder in (CACHE_DIR, CHARTER_CACHE, TEMPORAL_CACHE): write_text(os.path.join(folder, ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure(); manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args); path = dataset_path(output) if output else doc_path("README.md")
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK", "output": path.replace("\\", "/"), "output_hash": sha256(path)[:16], "notes": "Fail-closed manual intake; raw payload remains ignored."})
    write_csv(dataset_path("v2bi_orchestrator_manifest.csv"), manifest); return manifest

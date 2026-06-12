#!/usr/bin/env python3
"""v2ay Public Hydrometeorological Series Ingestion and Normalization."""

import argparse
import csv
import datetime as dt
import hashlib
import io
import os
import re
import zipfile

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2ay_hydromet_series_ingestion")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DOCS_DIR, "evidence_cache"))
RAW_DIR = os.environ.get("RAW_DIR", os.path.join(CACHE_DIR, "raw"))
MISSING_RATE_THRESHOLD = 0.20
ACCESS_DATE = "2026-06-10"
OFFICIAL_PAGES = {
    "INMET": "https://portal.inmet.gov.br/dadoshistoricos",
    "CEMADEN": "https://www.cemaden.gov.br/mapainterativo/",
    "ANA_HIDROWEB": "https://www.snirh.gov.br/hidroweb/",
}
INMET_RELEVANT_STATIONS = ("_RECIFE_", "_CURITIBA_", "_PICO DO COUTO_")
INPUTS = {
    "sources": "v2ax_hydromet_source_registry.csv",
    "windows": "v2ax_event_temporal_window_registry.csv",
    "acquisition": "v2ax_timeseries_acquisition_manifest.csv",
    "linkage": "v2ax_event_patch_temporal_linkage.csv",
    "readiness": "v2ax_temporal_evidence_readiness.csv",
    "provenance": "v2aw_public_source_provenance_registry.csv",
}
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false",
    "can_train_model": "false", "hydromet_time_series_is_not_geometry": "true",
    "precipitation_signal_is_not_flood_truth": "true",
    "temporal_readiness_is_not_truth_readiness": "true", "raw_data_versioned": "false",
}
ALIASES = {
    "timestamp": ["timestamp", "datahora", "data_hora", "datetime", "data", "date"],
    "value": ["observed_value", "valor", "precipitacao", "precipitation", "chuva", "value"],
    "station_id": ["station_id", "codigoestacao", "codestacao", "estacao", "station"],
    "station_name": ["station_name", "nomeestacao", "nome", "stationname"],
    "municipality": ["municipality", "municipio", "cidade"],
    "state": ["state", "uf", "estado"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
}


def parse_args(argv=None): return argparse.ArgumentParser().parse_args(argv)
def clean(value): return str(value or "").strip()
def is_true(value): return clean(value).lower() == "true"
def boolean(value): return "true" if bool(value) else "false"
def dataset_path(name): return os.path.join(DATASET_DIR, name)
def doc_path(name): return os.path.join(DOCS_DIR, name)
def rel_dataset(name): return f"datasets/protocolo_c/{name}"
def rel_doc(name): return f"docs/protocolo_c/v2ay_hydromet_series_ingestion/{name}"


def with_invariants(row): return {**row, **INVARIANTS}


def load_csv(path):
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not os.path.basename(path).startswith("v2ay_"): raise ValueError(f"Refusing non-v2ay output: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_markdown(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle: handle.write("\n".join(lines) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def load_inputs():
    stack = {}
    for key, name in INPUTS.items():
        path = dataset_path(name)
        if not os.path.exists(path): raise FileNotFoundError(f"Required input missing: {path}")
        stack[key] = load_csv(path)
    return stack


def ensure_cache_policy():
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, ".gitignore"), "w", encoding="utf-8") as handle:
        handle.write("*\n!.gitignore\n!raw/\n!raw/.gitignore\n")
    with open(os.path.join(RAW_DIR, ".gitignore"), "w", encoding="utf-8") as handle:
        handle.write("*\n!.gitignore\n")


def infer_source(filename):
    low = filename.lower()
    if "inmet" in low: return "INMET"
    if "cemaden" in low: return "CEMADEN"
    if "hidro" in low or "ana" in low: return "ANA_HIDROWEB"
    return "UNKNOWN"


def discover_raw_files():
    ensure_cache_policy()
    files = []
    for root, _, names in os.walk(RAW_DIR):
        files.extend(os.path.join(root, name) for name in names if name != ".gitignore")
    return sorted(files)


def inventory_status(path):
    ext = os.path.splitext(path)[1].lower()
    return "SUPPORTED_RAW_CSV" if ext == ".csv" else "SUPPORTED_RAW_ZIP" if ext == ".zip" else "UNSUPPORTED_RAW_FILE"


def _normalize_key(value): return re.sub(r"[^a-z0-9]+", "", clean(value).lower())


def _find_value(row, field):
    normalized = {_normalize_key(key): value for key, value in row.items()}
    for alias in ALIASES[field]:
        if _normalize_key(alias) in normalized: return clean(normalized[_normalize_key(alias)])
    return ""


def _float(value):
    try: return float(clean(value).replace(",", "."))
    except ValueError: return None


def normalize_row(row, source, raw_hash):
    timestamp, value = _find_value(row, "timestamp"), _float(_find_value(row, "value"))
    return {
        "source_name": source, "station_id": _find_value(row, "station_id"),
        "station_name": _find_value(row, "station_name"), "municipality": _find_value(row, "municipality"),
        "state": _find_value(row, "state"), "latitude": _find_value(row, "latitude"),
        "longitude": _find_value(row, "longitude"), "timestamp": timestamp,
        "variable_name": "precipitation", "variable_unit": "mm",
        "observed_value": "" if value is None else str(value), "quality_flag": "",
        "raw_file_hash": raw_hash, "parse_status": "PARSED" if timestamp and value is not None else "MISSING_REQUIRED_FIELDS",
    }


def _decode_payload(payload):
    for encoding in ("utf-8-sig", "latin1"):
        try: return payload.decode(encoding)
        except UnicodeDecodeError: pass
    return payload.decode("utf-8", errors="replace")


def parse_inmet_bytes(payload, raw_hash):
    text = _decode_payload(payload)
    lines = text.splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.lower().startswith("data;")), -1)
    if header_index < 0: return []
    metadata = {}
    for line in lines[:header_index]:
        parts = line.split(";", 1)
        if len(parts) == 2: metadata[_normalize_key(parts[0])] = clean(parts[1])
    rows = []
    for raw in csv.DictReader(io.StringIO("\n".join(lines[header_index:])), delimiter=";"):
        date = clean(raw.get("Data"))
        hour = clean(raw.get("Hora UTC")).replace(" UTC", "")
        timestamp = f"{date.replace('/', '-')} {hour[:2]}:{hour[2:4]}" if date and len(hour) >= 4 else date
        precip_key = next((key for key in raw if "PRECIPITA" in clean(key).upper()), "")
        value = _float(raw.get(precip_key))
        rows.append({
            "source_name": "INMET", "station_id": metadata.get("codigowmo", ""),
            "station_name": metadata.get("estacao", ""), "municipality": metadata.get("estacao", ""),
            "state": metadata.get("uf", ""), "latitude": metadata.get("latitude", ""),
            "longitude": metadata.get("longitude", ""), "timestamp": timestamp,
            "variable_name": "precipitation", "variable_unit": "mm",
            "observed_value": "" if value is None else str(value), "quality_flag": "",
            "raw_file_hash": raw_hash, "parse_status": "PARSED" if timestamp and value is not None else "MISSING_REQUIRED_FIELDS",
        })
    return rows


def parse_csv_bytes(payload, source, raw_hash):
    text = _decode_payload(payload)
    if source == "INMET" and any(line.lower().startswith("data;") for line in text.splitlines()[:20]):
        return parse_inmet_bytes(payload, raw_hash)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;")
    except csv.Error:
        dialect = csv.excel
    return [normalize_row(row, source, raw_hash) for row in csv.DictReader(io.StringIO(text), dialect=dialect)]


def parse_raw_file(path):
    source, raw_hash = infer_source(os.path.basename(path)), sha256_file(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "rb") as handle: return parse_csv_bytes(handle.read(), source, raw_hash)
    if ext == ".zip":
        rows = []
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            for name in names:
                if not name.lower().endswith(".csv"): continue
                if source == "INMET" and len(names) > 50 and not any(token in name.upper() for token in INMET_RELEVANT_STATIONS): continue
                rows.extend(parse_csv_bytes(archive.read(name), source, raw_hash))
        return rows
    return []


def all_normalized_rows():
    rows = []
    for path in discover_raw_files(): rows.extend(parse_raw_file(path))
    return rows


def parse_date(value):
    raw = clean(value).replace("T", " ").split(" ")[0]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try: return dt.datetime.strptime(raw, fmt).date()
        except ValueError: pass
    return None


def parse_timestamp(value):
    raw = clean(value).replace("T", " ").replace("Z", "")
    try:
        return dt.datetime.fromisoformat(raw)
    except ValueError:
        return None


def max_rolling_24h(rows):
    observations = sorted(
        (timestamp, float(row["observed_value"]))
        for row in rows
        if (timestamp := parse_timestamp(row.get("timestamp"))) is not None and clean(row.get("observed_value"))
    )
    maximum, start, running = None, 0, 0.0
    for end, (timestamp, value) in enumerate(observations):
        running += value
        while observations[start][0] < timestamp - dt.timedelta(hours=23):
            running -= observations[start][1]
            start += 1
        maximum = running if maximum is None else max(maximum, running)
    return maximum


def window_metrics(rows, window):
    start, end = parse_date(window.get("window_start")), parse_date(window.get("window_end"))
    selected = [row for row in rows if parse_date(row.get("timestamp")) and start and end
                and start <= parse_date(row["timestamp"]) <= end and row["parse_status"] == "PARSED"]
    values = [float(row["observed_value"]) for row in selected if clean(row.get("observed_value"))]
    max_24h = max_rolling_24h(selected)
    days = (end - start).days + 1 if start and end else 0
    expected = days * 24
    available = len(values)
    missing = 1.0 if expected == 0 else max(0.0, 1 - available / expected)
    signal = "UNKNOWN" if not values else "PRECIPITATION_PRESENT" if sum(values) > 0 else "NO_PRECIPITATION_OBSERVED"
    return {"records_expected": str(expected), "records_available": str(available), "missing_rate": f"{missing:.3f}",
            "precip_total_window": f"{sum(values):.3f}" if values else "", "precip_max_1h": f"{max(values):.3f}" if values else "",
            "precip_max_24h": f"{max_24h:.3f}" if max_24h is not None else "", "precip_signal_status": signal}


def readiness(parsed, window_defined, missing_rate, precipitation_available, signal):
    ready = parsed and window_defined and float(missing_rate) <= MISSING_RATE_THRESHOLD and precipitation_available and signal != "UNKNOWN"
    return "TEMPORAL_EVIDENCE_READY_FOR_REVIEW" if ready else "TEMPORAL_EVIDENCE_NOT_READY"


def station_name_for_candidate(candidate_id):
    prefix = clean(candidate_id).upper()[:3]
    return {"REC": "RECIFE", "PET": "PICO DO COUTO", "CTB": "CURITIBA"}.get(prefix, "")


def run_discover_cached_timeseries(args=None):
    rows = []
    for number, path in enumerate(discover_raw_files(), 1):
        source = infer_source(os.path.basename(path))
        relative = os.path.relpath(path, RAW_DIR).replace("\\", "/")
        period = next((token for token in re.findall(r"\d{4}", os.path.basename(path)) if token.startswith("20")), "")
        rows.append(with_invariants({"inventory_id": f"RAW_v2ay_{number:03d}",
            "relative_path": f"docs/protocolo_c/v2ay_hydromet_series_ingestion/evidence_cache/raw/{relative}",
            "file_size_bytes": str(os.path.getsize(path)), "sha256": sha256_file(path),
            "inferred_source": source, "agency": source, "official_page": OFFICIAL_PAGES.get(source, ""),
            "access_date": ACCESS_DATE, "period_or_year": period, "primary_variable": "precipitation",
            "inventory_status": inventory_status(path)}))
    if not rows: rows = [with_invariants({"inventory_id": "RAW_v2ay_000", "relative_path": "", "file_size_bytes": "0",
        "sha256": "", "inferred_source": "", "agency": "", "official_page": "", "access_date": ACCESS_DATE,
        "period_or_year": "", "primary_variable": "", "inventory_status": "NO_RAW_TIMESERIES_AVAILABLE"})]
    write_csv(dataset_path("v2ay_cached_timeseries_inventory.csv"), rows); return rows


def run_parse_timeseries(args=None):
    rows = []
    for number, path in enumerate(discover_raw_files(), 1):
        parsed = parse_raw_file(path)
        rows.append(with_invariants({"parse_id": f"PARSE_v2ay_{number:03d}", "raw_file_hash": sha256_file(path),
            "source_name": infer_source(os.path.basename(path)), "records_total": str(len(parsed)),
            "records_parsed": str(sum(1 for row in parsed if row["parse_status"] == "PARSED")),
            "parse_status": "PARSED" if parsed and all(row["parse_status"] == "PARSED" for row in parsed) else "PARSE_INCOMPLETE"}))
    if not rows: rows = [with_invariants({"parse_id": "PARSE_v2ay_000", "raw_file_hash": "", "source_name": "",
        "records_total": "0", "records_parsed": "0", "parse_status": "NO_RAW_TIMESERIES_AVAILABLE"})]
    write_csv(dataset_path("v2ay_timeseries_parse_audit.csv"), rows); return rows


def run_normalize_timeseries_schema(args=None):
    normalized = all_normalized_rows()
    manifests, stations = [], {}
    for row in normalized:
        key = (row["source_name"], row["raw_file_hash"])
        stations[(row["source_name"], row["station_id"])] = row
        if key not in {(m["source_name"], m["raw_file_hash"]) for m in manifests}:
            manifests.append(with_invariants({"manifest_id": f"NORM_v2ay_{len(manifests)+1:03d}", "source_name": row["source_name"],
                "raw_file_hash": row["raw_file_hash"], "normalized_schema_status": "NORMALIZED_SCHEMA_AVAILABLE",
                "normalized_records": str(sum(1 for item in normalized if item["source_name"] == row["source_name"] and item["raw_file_hash"] == row["raw_file_hash"]))}))
    if not manifests: manifests = [with_invariants({"manifest_id": "NORM_v2ay_000", "source_name": "", "raw_file_hash": "",
        "normalized_schema_status": "NO_RAW_TIMESERIES_AVAILABLE", "normalized_records": "0"})]
    station_rows = [with_invariants({"station_metadata_id": f"ST_v2ay_{i:03d}", "source_name": row["source_name"],
        "station_id": row["station_id"], "station_name": row["station_name"], "municipality": row["municipality"],
        "state": row["state"], "latitude": row["latitude"], "longitude": row["longitude"]})
        for i, row in enumerate(stations.values(), 1)]
    if not station_rows: station_rows = [with_invariants({"station_metadata_id": "ST_v2ay_000", "source_name": "", "station_id": "",
        "station_name": "", "municipality": "", "state": "", "latitude": "", "longitude": ""})]
    write_csv(dataset_path("v2ay_normalized_timeseries_manifest.csv"), manifests)
    write_csv(dataset_path("v2ay_station_metadata_registry.csv"), station_rows); return manifests


def run_compute_window_precipitation_metrics(args=None):
    normalized, rows = all_normalized_rows(), []
    for number, window in enumerate(load_inputs()["windows"], 1):
        station_name = station_name_for_candidate(window["candidate_id"])
        regional = [row for row in normalized if row["station_name"].upper() == station_name]
        metrics = window_metrics(regional, window)
        support = readiness(bool(regional), is_true(window["temporal_window_defined"]), metrics["missing_rate"],
                            bool(regional), metrics["precip_signal_status"])
        station_id = next((row["station_id"] for row in regional if row["station_id"]), "")
        rows.append(with_invariants({"metric_id": f"METRIC_v2ay_{number:04d}", "event_patch_package_id": window["assertion_id"],
            "source_name": "INMET", "station_id": station_id, "window_start": window["window_start"], "event_date": window["event_date"],
            "window_end": window["window_end"], **metrics, "temporal_support_status": support,
            "absence_of_precipitation_creates_negative": "false"}))
    write_csv(dataset_path("v2ay_window_precipitation_metrics.csv"), rows); return rows


def run_audit_timeseries_quality(args=None):
    rows = []
    for number, metric in enumerate(load_csv(dataset_path("v2ay_window_precipitation_metrics.csv")), 1):
        status = "QUALITY_ACCEPTABLE_FOR_REVIEW" if float(metric["missing_rate"]) <= MISSING_RATE_THRESHOLD else "QUALITY_INCOMPLETE"
        rows.append(with_invariants({"quality_id": f"QUALITY_v2ay_{number:04d}", "event_patch_package_id": metric["event_patch_package_id"],
            "missing_rate": metric["missing_rate"], "precip_signal_status": metric["precip_signal_status"], "quality_status": status}))
    write_csv(dataset_path("v2ay_timeseries_quality_report.csv"), rows); return rows


def run_update_temporal_readiness(args=None):
    metrics = {r["event_patch_package_id"]: r for r in load_csv(dataset_path("v2ay_window_precipitation_metrics.csv"))}
    rows = []
    for number, link in enumerate(load_inputs()["linkage"], 1):
        metric = metrics.get(link["assertion_id"], {})
        status = clean(metric.get("temporal_support_status")) or "TEMPORAL_EVIDENCE_NOT_READY"
        rows.append(with_invariants({"readiness_id": f"READY_v2ay_{number:04d}", "assertion_id": link["assertion_id"],
            "candidate_id": link["candidate_id"], "temporal_readiness_status": status,
            "geometry_still_blocking_truth": "true"}))
    write_csv(dataset_path("v2ay_event_patch_temporal_readiness_update.csv"), rows); return rows


def run_generate_ingestion_report(args=None):
    ready = load_csv(dataset_path("v2ay_event_patch_temporal_readiness_update.csv"))
    ready_n = sum(1 for row in ready if row["temporal_readiness_status"] == "TEMPORAL_EVIDENCE_READY_FOR_REVIEW")
    no_raw = not discover_raw_files()
    rows = [with_invariants({"report_id": "GAP_v2ay_001", "metric": "raw_timeseries_status",
        "value": "NO_RAW_TIMESERIES_AVAILABLE" if no_raw else "RAW_TIMESERIES_DISCOVERED", "status": "RECORDED"}),
        with_invariants({"report_id": "GAP_v2ay_002", "metric": "temporal_evidence_ready_count", "value": str(ready_n), "status": "RECORDED"}),
        with_invariants({"report_id": "GAP_v2ay_003", "metric": "next_action_rank_1",
        "value": "MANUALLY_DOWNLOAD_PUBLIC_HYDROMET_TIMESERIES" if ready_n == 0 else "BUILD_MANUAL_GEOMETRY_REVIEW_PACKETS_FOR_TEMPORALLY_SUPPORTED_EVENTS",
        "status": "SAFE_NEXT_ACTION"})]
    write_csv(dataset_path("v2ay_ingestion_gap_report.csv"), rows); return rows


def scan_guardrails(path):
    bad = []
    for row in load_csv(path):
        for field, expected in INVARIANTS.items():
            if field in row and clean(row[field]).lower() != expected: bad.append(field)
    return bad


def run_guardrail_regression(args=None):
    rows, failures = [], 0
    for name in sorted(os.listdir(DATASET_DIR)):
        if name.startswith("v2ay_") and name.endswith(".csv"):
            bad = scan_guardrails(dataset_path(name)); failures += len(bad)
            rows.append({"regression_id": f"GR_v2ay_{len(rows):03d}", "artifact_path": rel_dataset(name),
                "violation_count": str(len(bad)), "status": "PASS" if not bad else "FAIL"})
    cache_ok = (open(os.path.join(CACHE_DIR, ".gitignore"), encoding="utf-8").read()
                == "*\n!.gitignore\n!raw/\n!raw/.gitignore\n"
                and open(os.path.join(RAW_DIR, ".gitignore"), encoding="utf-8").read() == "*\n!.gitignore\n")
    rows.append({"regression_id": f"GR_v2ay_{len(rows):03d}", "artifact_path": rel_doc("evidence_cache/.gitignore|evidence_cache/raw/.gitignore"),
        "violation_count": "0" if cache_ok else "1", "status": "PASS" if cache_ok else "FAIL"}); failures += int(not cache_ok)
    write_csv(dataset_path("v2ay_guardrail_regression.csv"), rows)
    if failures: raise ValueError(f"v2ay guardrail regression failed: {failures}")
    return rows


def _write_docs():
    ensure_cache_policy()
    write_markdown(doc_path("README.md"), ["# v2ay Hydromet Series Ingestion", "", "Place manual raw CSV/ZIP files in evidence_cache/raw/. Raw data remains ignored."])
    write_markdown(os.path.join(DOCS_DIR, "manual_input_instructions", "README.md"), ["# Manual input", "", "Name files with INMET, CEMADEN, HIDROWEB or ANA to support source inference."])
    write_markdown(os.path.join(DOCS_DIR, "normalized_schema", "README.md"), ["# Normalized schema", "", "Only manifests, station metadata, audits and derived metrics are versionable."])


ORCHESTRATION = [
    ("discover_cached_timeseries", run_discover_cached_timeseries, "v2ay_cached_timeseries_inventory.csv"),
    ("parse_timeseries", run_parse_timeseries, "v2ay_timeseries_parse_audit.csv"),
    ("normalize_timeseries_schema", run_normalize_timeseries_schema, "v2ay_normalized_timeseries_manifest.csv"),
    ("compute_window_precipitation_metrics", run_compute_window_precipitation_metrics, "v2ay_window_precipitation_metrics.csv"),
    ("audit_timeseries_quality", run_audit_timeseries_quality, "v2ay_timeseries_quality_report.csv"),
    ("update_temporal_readiness", run_update_temporal_readiness, "v2ay_event_patch_temporal_readiness_update.csv"),
    ("generate_ingestion_report", run_generate_ingestion_report, "v2ay_ingestion_gap_report.csv"),
    ("guardrail_regression", run_guardrail_regression, "v2ay_guardrail_regression.csv"),
]


def run_orchestrator(args=None):
    _write_docs(); rows = []
    for order, (name, function, output) in enumerate(ORCHESTRATION, 1):
        try: function(args); status, notes = "OK", "Completed."
        except Exception as exc: status, notes = "FAIL", f"{type(exc).__name__}: {exc}"
        rows.append({"step_order": str(order), "step_name": name, "status": status, "output": rel_dataset(output),
            "output_hash": sha256_file(dataset_path(output))[:16] if os.path.exists(dataset_path(output)) else "", "notes": notes})
        if status == "FAIL": write_csv(dataset_path("v2ay_orchestrator_manifest.csv"), rows); raise ValueError(notes)
    write_csv(dataset_path("v2ay_orchestrator_manifest.csv"), rows); return rows


# Source-specific parser entry points retained for thin wrappers and tests.
def run_parse_inmet_timeseries(args=None): return run_parse_timeseries(args)
def run_parse_cemaden_timeseries(args=None): return run_parse_timeseries(args)
def run_parse_hidroweb_timeseries(args=None): return run_parse_timeseries(args)

#!/usr/bin/env python3
"""v2ax Public Hydrometeorological Time-Series Acquisition and Temporal Evidence Pack."""

import argparse
import csv
import datetime as dt
import hashlib
import os
import re
import urllib.request

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2ax_hydrometeorological_temporal_evidence")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DOCS_DIR, "evidence_cache"))
NETWORK_ENV = "V2AX_NETWORK"
HTTP_TIMEOUT = 10
MAX_CACHE_BYTES = 2 * 1024 * 1024
MISSING_RATE_THRESHOLD = 0.20
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
INPUTS = {
    "targets": "v2aw_observational_data_target_registry.csv",
    "plan": "v2aw_hydrometeorological_acquisition_plan.csv",
    "windows": "v2aw_temporal_window_candidate_registry.csv",
    "packages": "v2aw_event_patch_observation_package_index.csv",
    "provenance": "v2aw_public_source_provenance_registry.csv",
}
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false",
    "can_train_model": "false", "hydromet_time_series_is_not_geometry": "true",
    "precipitation_signal_is_not_flood_truth": "true",
    "temporal_readiness_is_not_truth_readiness": "true", "raw_data_versioned": "false",
}


def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def boolean(value):
    return "true" if bool(value) else "false"


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/protocolo_c/v2ax_hydrometeorological_temporal_evidence/{name}"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not os.path.basename(path).startswith("v2ax_"):
        raise ValueError(f"Refusing non-v2ax output: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    columns = list(rows[0]) if rows else ["id"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write_markdown(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def sha256_file(path):
    if not os.path.exists(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def with_invariants(row):
    return {**row, **INVARIANTS}


def load_inputs():
    stack = {}
    for key, name in INPUTS.items():
        path = dataset_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required v2aw input missing: {path}")
        stack[key] = load_csv(path)
    return stack


def ensure_cache_policy():
    os.makedirs(CACHE_DIR, exist_ok=True)
    marker = cache_path(".gitignore")
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("*\n!.gitignore\n")
    return marker


def is_network_enabled():
    return clean(os.environ.get(NETWORK_ENV)) == "1"


def fetch_light_series(url, source_id):
    ensure_cache_policy()
    if not is_network_enabled():
        return {"acquisition_status": "NETWORK_DISABLED_DETERMINISTIC_RUN", "cache_path": "",
                "cache_sha256": "", "file_size_bytes": "", "raw_data_versioned": "false"}
    request = urllib.request.Request(clean(url), headers={"User-Agent": "REV-P-v2ax/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            payload = response.read(MAX_CACHE_BYTES + 1)
        if len(payload) > MAX_CACHE_BYTES:
            return {"acquisition_status": "PAYLOAD_TOO_LARGE_NOT_CACHED", "cache_path": "",
                    "cache_sha256": "", "file_size_bytes": str(len(payload)), "raw_data_versioned": "false"}
        name = re.sub(r"[^a-z0-9]+", "-", source_id.lower()).strip("-") + ".timeseries"
        path = cache_path(name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return {"acquisition_status": "LIGHT_PUBLIC_TIMESERIES_CACHED", "cache_path": rel_doc(f"evidence_cache/{name}"),
                "cache_sha256": sha256_file(path), "file_size_bytes": str(len(payload)), "raw_data_versioned": "false"}
    except Exception as exc:
        return {"acquisition_status": f"ACQUISITION_FAILED_{type(exc).__name__.upper()}",
                "cache_path": "", "cache_sha256": "", "file_size_bytes": "", "raw_data_versioned": "false"}


def parse_event_window(event_id):
    parts = clean(event_id).split("_")
    if len(parts) < 4:
        return {"window_start": "", "event_date": "", "window_end": "",
                "temporal_precision": "UNKNOWN", "temporal_window_defined": "false",
                "temporal_window_source": "UNRESOLVED_EVENT_ID"}
    try:
        year, month = int(parts[1]), int(parts[2])
        days = [int(value) for value in parts[3:] if value.isdigit()]
        if not days:
            raise ValueError
        start_date = dt.date(year, month, days[0])
        event_date = start_date
        end_date = dt.date(year, month, days[-1]) if len(days) > 1 else start_date
        precision = "DATE_RANGE" if len(days) > 1 else "EXACT_DATE"
        return {"window_start": (start_date - dt.timedelta(days=7)).isoformat(),
                "event_date": event_date.isoformat(), "window_end": (end_date + dt.timedelta(days=3)).isoformat(),
                "temporal_precision": precision, "temporal_window_defined": "true",
                "temporal_window_source": "EVENT_ID_DEFAULT_MINUS7_PLUS3"}
    except (ValueError, IndexError):
        return {"window_start": "", "event_date": "", "window_end": "",
                "temporal_precision": "UNKNOWN", "temporal_window_defined": "false",
                "temporal_window_source": "UNRESOLVED_EVENT_ID"}


def quality_status(expected_records, available_records, precipitation_present=True):
    expected = int(expected_records or 0)
    available = int(available_records or 0)
    missing_rate = 1.0 if expected <= 0 else max(0.0, min(1.0, 1 - available / expected))
    if not precipitation_present:
        status = "PRECIPITATION_VARIABLE_MISSING"
    elif missing_rate <= MISSING_RATE_THRESHOLD:
        status = "QUALITY_ACCEPTABLE_FOR_REVIEW"
    else:
        status = "QUALITY_INCOMPLETE"
    return missing_rate, status


def readiness(source_public, window_defined, station_candidate, missing_rate, precipitation_present):
    ready = all([source_public, window_defined, station_candidate, precipitation_present,
                 float(missing_rate) <= MISSING_RATE_THRESHOLD])
    return "TEMPORAL_EVIDENCE_READY_FOR_REVIEW" if ready else "TEMPORAL_EVIDENCE_NOT_READY"


def run_build_hydromet_source_registry(args=None):
    stack = load_inputs()
    provenance = {r["source_id"]: r for r in stack["provenance"]}
    rows = []
    for number, plan in enumerate(stack["plan"], 1):
        rows.append(with_invariants({
            "source_registry_id": f"SRC_v2ax_{number:03d}", "source_id": plan["source_id"],
            "source_name": plan["source_id"], "official_url": plan["official_url"],
            "source_public": boolean(plan["source_id"] in provenance), "provenance_status": clean(
                provenance.get(plan["source_id"], {}).get("provenance_status")),
            "source_type": "DYNAMIC_MANUAL" if plan["source_id"] == "ANA_HIDROWEB" else "PUBLIC_TIMESERIES",
            "station_or_series_candidate": "true", "citation_required": "true",
        }))
    write_csv(dataset_path("v2ax_hydromet_source_registry.csv"), rows)
    return rows


def run_build_station_candidates(args=None):
    rows = []
    for number, source in enumerate(load_csv(dataset_path("v2ax_hydromet_source_registry.csv")), 1):
        rows.append(with_invariants({
            "station_candidate_id": f"STATION_v2ax_{number:03d}", "source_id": source["source_id"],
            "station_id": "", "station_name": "", "municipality": "", "state": "",
            "latitude": "", "longitude": "", "distance_to_patch_km": "",
            "variable": "precipitation", "station_or_series_candidate": "true",
            "candidate_status": "STATION_DISCOVERY_REQUIRED_OFFLINE",
            "nearby_station_proves_patch_event": "false",
        }))
    write_csv(dataset_path("v2ax_station_candidate_registry.csv"), rows)
    return rows


def run_build_event_temporal_windows(args=None):
    rows = []
    for number, window in enumerate(load_inputs()["windows"], 1):
        parsed = parse_event_window(window["event_id"])
        status = "TEMPORAL_WINDOW_DEFINED" if is_true(parsed["temporal_window_defined"]) else "TEMPORAL_WINDOW_INCOMPLETE"
        rows.append(with_invariants({
            "window_id": f"WINDOW_v2ax_{number:04d}", "assertion_id": window["assertion_id"],
            "candidate_id": window["candidate_id"], "event_id": window["event_id"], **parsed,
            "window_status": status,
        }))
    write_csv(dataset_path("v2ax_event_temporal_window_registry.csv"), rows)
    return rows


def run_acquire_timeseries_manifest(args=None):
    rows = []
    for number, source in enumerate(load_csv(dataset_path("v2ax_hydromet_source_registry.csv")), 1):
        attempt = fetch_light_series(source["official_url"], source["source_id"])
        rows.append(with_invariants({
            "manifest_id": f"MAN_v2ax_{number:03d}", "source_id": source["source_id"],
            "official_url": source["official_url"], "network_enabled": boolean(is_network_enabled()),
            **attempt, "derived_light_outputs_only": "true",
        }))
    write_csv(dataset_path("v2ax_timeseries_acquisition_manifest.csv"), rows)
    return rows


def run_audit_timeseries_quality(args=None):
    rows = []
    for number, station in enumerate(load_csv(dataset_path("v2ax_station_candidate_registry.csv")), 1):
        missing_rate, status = quality_status(0, 0, True)
        rows.append(with_invariants({
            "quality_id": f"QUALITY_v2ax_{number:03d}", "station_id": station["station_id"],
            "station_name": station["station_name"], "source_name": station["source_id"],
            "municipality": station["municipality"], "state": station["state"],
            "latitude": station["latitude"], "longitude": station["longitude"],
            "distance_to_patch_km": station["distance_to_patch_km"], "variable": station["variable"],
            "observed_start": "", "observed_end": "", "expected_records": "0", "available_records": "0",
            "missing_rate": f"{missing_rate:.3f}", "max_precip_1h": "", "precip_24h_max": "",
            "precip_window_total": "", "quality_status": status,
        }))
    write_csv(dataset_path("v2ax_timeseries_quality_audit.csv"), rows)
    return rows


def run_summarize_precipitation_events(args=None):
    rows = []
    for number, window in enumerate(load_csv(dataset_path("v2ax_event_temporal_window_registry.csv")), 1):
        rows.append(with_invariants({
            "summary_id": f"PRECIP_v2ax_{number:04d}", "assertion_id": window["assertion_id"],
            "event_id": window["event_id"], "window_start": window["window_start"],
            "event_date": window["event_date"], "window_end": window["window_end"],
            "precipitation_signal_present": "false", "max_precip_1h": "", "precip_24h_max": "",
            "precip_window_total": "", "summary_status": "NO_SERIES_AVAILABLE_OFFLINE",
            "absence_of_rain_creates_negative": "false",
        }))
    write_csv(dataset_path("v2ax_precipitation_event_summary.csv"), rows)
    return rows


def run_link_event_patch_temporal_evidence(args=None):
    windows = {r["assertion_id"]: r for r in load_csv(dataset_path("v2ax_event_temporal_window_registry.csv"))}
    rows = []
    for number, package in enumerate(load_inputs()["packages"], 1):
        window = windows.get(package["assertion_id"], {})
        rows.append(with_invariants({
            "linkage_id": f"LINK_v2ax_{number:04d}", "assertion_id": package["assertion_id"],
            "candidate_id": package["candidate_id"], "event_id": package["event_id"], "patch_id": package["patch_id"],
            "temporal_window_defined": clean(window.get("temporal_window_defined")) or "false",
            "station_or_series_candidate": "true", "precipitation_signal_present": "false",
            "geometry_observed": "false", "geometry_still_blocking_truth": "true",
            "linkage_status": "TEMPORAL_CANDIDATE_LINK_REVIEW_ONLY",
        }))
    write_csv(dataset_path("v2ax_event_patch_temporal_linkage.csv"), rows)
    return rows


def run_compute_temporal_readiness(args=None):
    rows = []
    for number, link in enumerate(load_csv(dataset_path("v2ax_event_patch_temporal_linkage.csv")), 1):
        status = readiness(True, is_true(link["temporal_window_defined"]), is_true(link["station_or_series_candidate"]),
                           1.0, False)
        rows.append(with_invariants({
            "readiness_id": f"READY_v2ax_{number:04d}", "assertion_id": link["assertion_id"],
            "candidate_id": link["candidate_id"], "source_public": "true",
            "temporal_window_defined": link["temporal_window_defined"],
            "station_or_series_candidate": link["station_or_series_candidate"], "missing_rate": "1.000",
            "precipitation_variables_present": "false", "temporal_readiness_status": status,
            "geometry_still_blocking_truth": "true",
        }))
    write_csv(dataset_path("v2ax_temporal_evidence_readiness.csv"), rows)
    return rows


def run_generate_observational_gap_report(args=None):
    rows = []
    for number, ready in enumerate(load_csv(dataset_path("v2ax_temporal_evidence_readiness.csv")), 1):
        rows.append(with_invariants({
            "gap_id": f"GAP_v2ax_{number:04d}", "assertion_id": ready["assertion_id"],
            "candidate_id": ready["candidate_id"], "temporal_readiness_status": ready["temporal_readiness_status"],
            "temporal_gap": "PUBLIC_TIMESERIES_NOT_ACQUIRED_OR_QUALITY_INCOMPLETE",
            "geometry_gap": "GEOMETRY_STILL_MISSING", "geometry_still_blocking_truth": "true",
            "next_action_rank_1": "ACQUIRE_OBSERVED_EVENT_GEOMETRY_FOR_TEMPORALLY_SUPPORTED_PACKAGES",
        }))
    write_csv(dataset_path("v2ax_observational_gap_report.csv"), rows)
    return rows


def scan_guardrails(path):
    violations = []
    for row in load_csv(path):
        for field, expected in INVARIANTS.items():
            if field in row and clean(row[field]).lower() != expected:
                violations.append(field)
        if is_true(row.get("precipitation_signal_present")) and is_true(row.get("can_create_patch_truth")):
            violations.append("precipitation_promoted_patch_truth")
        for value in row.values():
            if ABSOLUTE_PATH_RE.search(clean(value)):
                violations.append("absolute_path")
    return violations


def run_guardrail_regression(args=None):
    rows, failures = [], 0
    for name in sorted(os.listdir(DATASET_DIR)):
        if name.startswith("v2ax_") and name.endswith(".csv"):
            violations = scan_guardrails(dataset_path(name))
            failures += len(violations)
            rows.append({"regression_id": f"GR_v2ax_{len(rows):03d}", "artifact_path": rel_dataset(name),
                         "violation_count": str(len(violations)), "status": "PASS" if not violations else "FAIL"})
    cache_ok = read_text(ensure_cache_policy()) == "*\n!.gitignore\n"
    failures += int(not cache_ok)
    rows.append({"regression_id": f"GR_v2ax_{len(rows):03d}", "artifact_path": rel_doc("evidence_cache/.gitignore"),
                 "violation_count": "0" if cache_ok else "1", "status": "PASS" if cache_ok else "FAIL"})
    write_csv(dataset_path("v2ax_guardrail_regression.csv"), rows)
    if failures:
        raise ValueError(f"v2ax guardrail regression failed: {failures}")
    return rows


def _write_docs():
    ensure_cache_policy()
    write_markdown(doc_path("README.md"), [
        "# v2ax Hydrometeorological Temporal Evidence", "",
        "Public temporal evidence supports review only. Rainfall is not flood truth and time series are not geometry.",
    ])
    write_markdown(os.path.join(DOCS_DIR, "acquisition_logs", "README.md"),
                   ["# Acquisition logs", "", "Light public series only; raw files remain in ignored cache."])
    write_markdown(os.path.join(DOCS_DIR, "timeseries_summaries", "README.md"),
                   ["# Time-series summaries", "", "Only lightweight derived summaries and audits are versionable."])


ORCHESTRATION = [
    ("hydromet_source_registry", run_build_hydromet_source_registry, "v2ax_hydromet_source_registry.csv"),
    ("station_candidates", run_build_station_candidates, "v2ax_station_candidate_registry.csv"),
    ("event_temporal_windows", run_build_event_temporal_windows, "v2ax_event_temporal_window_registry.csv"),
    ("timeseries_acquisition_manifest", run_acquire_timeseries_manifest, "v2ax_timeseries_acquisition_manifest.csv"),
    ("timeseries_quality_audit", run_audit_timeseries_quality, "v2ax_timeseries_quality_audit.csv"),
    ("precipitation_event_summary", run_summarize_precipitation_events, "v2ax_precipitation_event_summary.csv"),
    ("event_patch_temporal_linkage", run_link_event_patch_temporal_evidence, "v2ax_event_patch_temporal_linkage.csv"),
    ("temporal_readiness", run_compute_temporal_readiness, "v2ax_temporal_evidence_readiness.csv"),
    ("observational_gap_report", run_generate_observational_gap_report, "v2ax_observational_gap_report.csv"),
    ("guardrail_regression", run_guardrail_regression, "v2ax_guardrail_regression.csv"),
]


def run_orchestrator(args=None):
    _write_docs()
    rows = []
    for order, (name, function, output) in enumerate(ORCHESTRATION, 1):
        try:
            function(args)
            status, notes = "OK", "Completed."
        except Exception as exc:
            status, notes = "FAIL", f"{type(exc).__name__}: {exc}"
        rows.append({"step_order": str(order), "step_name": name, "status": status,
                     "output": rel_dataset(output), "output_hash": sha256_file(dataset_path(output))[:16],
                     "notes": notes})
        if status == "FAIL":
            write_csv(dataset_path("v2ax_orchestrator_manifest.csv"), rows)
            raise ValueError(notes)
    write_csv(dataset_path("v2ax_orchestrator_manifest.csv"), rows)
    return rows

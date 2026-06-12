#!/usr/bin/env python3
"""v2av Official Source Terms Snapshot and Manual Acquisition Pack."""

import argparse
import csv
import hashlib
import os
import re
import urllib.request

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2av_official_source_terms_snapshot")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DOCS_DIR, "evidence_cache"))
NETWORK_ENV = "V2AV_NETWORK"
HTTP_TIMEOUT = 8
MAX_SNAPSHOT_BYTES = 256 * 1024
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
INPUTS = {
    "tasks": "v2au_resolution_task_registry.csv",
    "terms": "v2au_source_contact_and_terms_audit.csv",
    "licenses": "v2au_license_resolution_matrix.csv",
    "crs": "v2au_crs_resolution_matrix.csv",
    "temporal": "v2au_temporal_data_request_plan.csv",
    "instructions": "v2au_manual_download_instructions.csv",
    "endpoints": "v2au_source_endpoint_attempts.csv",
    "queue": "v2au_review_packet_priority_queue.csv",
}
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false",
    "can_train_model": "false", "quickview_can_promote": "false",
    "susceptibility_can_promote": "false", "dino_can_decide_event": "false",
    "terms_snapshot_is_not_legal_clearance": "true",
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
    return f"docs/protocolo_c/v2av_official_source_terms_snapshot/{name}"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not os.path.basename(path).startswith("v2av_"):
        raise ValueError(f"Refusing non-v2av output: {path}")
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


def load_v2au_inputs():
    stack = {}
    for key, name in INPUTS.items():
        path = dataset_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required v2au input missing: {path}")
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


def snapshot_page(url, source_id):
    ensure_cache_policy()
    if not is_network_enabled():
        return {"http_status": "", "snapshot_status": "NETWORK_DISABLED_DETERMINISTIC_RUN",
                "snapshot_path": "", "snapshot_sha256": "", "raw_data_versioned": "false"}
    request = urllib.request.Request(clean(url), headers={"User-Agent": "REV-P-v2av/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            payload = response.read(MAX_SNAPSHOT_BYTES + 1)
            status = str(getattr(response, "status", ""))
        if len(payload) > MAX_SNAPSHOT_BYTES:
            return {"http_status": status, "snapshot_status": "SNAPSHOT_TOO_LARGE_NOT_CACHED",
                    "snapshot_path": "", "snapshot_sha256": "", "raw_data_versioned": "false"}
        text = payload.decode("utf-8", errors="replace")
        name = re.sub(r"[^a-z0-9]+", "-", source_id.lower()).strip("-") + ".txt"
        path = cache_path(name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return {"http_status": status, "snapshot_status": "TEXT_SNAPSHOT_CACHED_FOR_REVIEW",
                "snapshot_path": rel_doc(f"evidence_cache/{name}"), "snapshot_sha256": sha256_file(path),
                "raw_data_versioned": "false"}
    except Exception as exc:
        return {"http_status": "", "snapshot_status": f"SNAPSHOT_FAILED_{type(exc).__name__.upper()}",
                "snapshot_path": "", "snapshot_sha256": "", "raw_data_versioned": "false"}


def classify_page(found=False, download=False, data_access=False):
    if data_access:
        return "DATA_ACCESS_CONFIRMED"
    if download:
        return "DOWNLOAD_PAGE_FOUND"
    if found:
        return "TERMS_PAGE_FOUND"
    return "TERMS_STILL_UNKNOWN"


def classify_license_candidate(explicit=False, ambiguous=False, reusable=True):
    if explicit and reusable:
        return "LICENSE_CANDIDATE_EXPLICIT"
    if explicit and not reusable:
        return "NOT_REUSABLE_FOR_PUBLIC_REPO"
    if ambiguous:
        return "REQUIRES_MANUAL_REVIEW"
    return "LICENSE_STILL_UNKNOWN"


def classify_crs_candidate(spatial=False, crs_explicit=False):
    if not spatial:
        return "NOT_SPATIAL"
    return "CRS_CANDIDATE_EXPLICIT" if crs_explicit else "CRS_STILL_UNKNOWN"


def source_role(source_id):
    if source_id in {"INMET", "CEMADEN", "ANA_HIDROWEB"}:
        return "OFFICIAL_OBSERVED_TIME_SERIES"
    if source_id == "SGB_CPRM":
        return "SUSCEPTIBILITY_OR_RISK_CONTEXT"
    if source_id == "INTERNATIONAL_CHARTER":
        return "QUICKVIEW_OR_PRODUCT"
    return "OFFICIAL_SPATIAL_PRODUCT"


def run_build_terms_snapshot_registry(args=None):
    rows = []
    for number, endpoint in enumerate(load_v2au_inputs()["endpoints"], 1):
        snap = snapshot_page(endpoint["target_url"], endpoint["source_id"])
        rows.append(with_invariants({
            "snapshot_id": f"SNAP_v2av_{number:03d}", "source_id": endpoint["source_id"],
            "official_url": endpoint["target_url"], **snap,
            "terms_classification": "TERMS_STILL_UNKNOWN",
            "license_candidate_status": "LICENSE_STILL_UNKNOWN",
        }))
    write_csv(dataset_path("v2av_source_terms_snapshot_registry.csv"), rows)
    return rows


def run_build_access_page_registry(args=None):
    rows = []
    for number, endpoint in enumerate(load_v2au_inputs()["endpoints"], 1):
        rows.append(with_invariants({
            "access_id": f"ACCESS_v2av_{number:03d}", "source_id": endpoint["source_id"],
            "official_url": endpoint["target_url"], "source_role": source_role(endpoint["source_id"]),
            "access_page_status": "REQUIRES_MANUAL_REVIEW",
            "download_public_is_public_license": "false", "official_source_is_observed_event": "false",
        }))
    write_csv(dataset_path("v2av_official_access_page_registry.csv"), rows)
    return rows


def run_build_manual_acquisition_packets(args=None):
    stack = load_v2au_inputs()
    instructions = {r["source_id"]: r for r in stack["instructions"]}
    rows = []
    for number, queue in enumerate(stack["queue"], 1):
        instruction = instructions.get(queue["source_id"], {})
        rows.append(with_invariants({
            "packet_id": f"PACK_v2av_{number:04d}", "queue_rank": queue["queue_rank"],
            "assertion_id": queue["assertion_id"], "source_id": queue["source_id"],
            "source_name": queue["source_name"], "priority_band": queue["priority_band"],
            "manual_steps_available": boolean(bool(instruction)), "packet_status": "REQUIRES_MANUAL_REVIEW",
            "allowed_use": "source_access_terms_and_temporal_evidence_review_only",
        }))
    write_csv(dataset_path("v2av_manual_acquisition_packet_index.csv"), rows)
    return rows


def run_build_license_resolution_candidates(args=None):
    rows = []
    for number, license_row in enumerate(load_v2au_inputs()["licenses"], 1):
        rows.append(with_invariants({
            "candidate_id": f"LICC_v2av_{number:04d}", "assertion_id": license_row["assertion_id"],
            "source_id": license_row["source_id"], "prior_license_status": license_row["license_resolution_status"],
            "license_candidate_status": "LICENSE_STILL_UNKNOWN", "explicit_text_found": "false",
            "institutional_page_traceable": "false", "legal_clearance_granted": "false",
        }))
    write_csv(dataset_path("v2av_license_resolution_candidates.csv"), rows)
    return rows


def run_build_temporal_observation_targets(args=None):
    rows = []
    for number, target in enumerate(load_v2au_inputs()["temporal"], 1):
        rows.append(with_invariants({
            "target_id": f"TEMP_v2av_{number:04d}", "assertion_id": target["assertion_id"],
            "candidate_id": target["candidate_id"], "target_sources": target["requested_sources"],
            "target_status": "OFFICIAL_OBSERVED_TIME_SERIES_REQUIRES_MANUAL_ACQUISITION",
            "can_improve_readiness": "true", "can_create_observed_geometry_alone": "false",
        }))
    write_csv(dataset_path("v2av_temporal_observation_targets.csv"), rows)
    return rows


def run_recheck_source_priority(args=None):
    rows = []
    for queue in load_v2au_inputs()["queue"]:
        rows.append(with_invariants({
            "priority_rank": queue["queue_rank"], "assertion_id": queue["assertion_id"],
            "source_id": queue["source_id"], "source_name": queue["source_name"],
            "priority_band": queue["priority_band"], "blocker_count": queue["blocker_count"],
            "terms_status": "TERMS_STILL_UNKNOWN", "license_status": "LICENSE_STILL_UNKNOWN",
            "priority_recheck_status": "HIGH_SOURCE_REQUIRES_MANUAL_TERMS_AND_ACCESS_REVIEW",
        }))
    write_csv(dataset_path("v2av_source_priority_recheck.csv"), rows)
    return rows


def run_generate_unresolved_terms_report(args=None):
    rows = []
    for number, snapshot in enumerate(load_csv(dataset_path("v2av_source_terms_snapshot_registry.csv")), 1):
        rows.append(with_invariants({
            "report_id": f"UNRES_v2av_{number:03d}", "source_id": snapshot["source_id"],
            "terms_status": snapshot["terms_classification"],
            "license_status": snapshot["license_candidate_status"],
            "recommended_action": "CONTACT_SOURCE_OWNER_FOR_LICENSE_CLARIFICATION",
            "resolution_status": "UNRESOLVED",
        }))
    write_csv(dataset_path("v2av_unresolved_terms_report.csv"), rows)
    return rows


def scan_guardrails(path):
    violations = []
    for row in load_csv(path):
        for field, expected in INVARIANTS.items():
            if field in row and clean(row[field]).lower() != expected:
                violations.append(field)
        if is_true(row.get("raw_data_versioned")) or is_true(row.get("legal_clearance_granted")):
            violations.append("raw_or_legal_promotion")
        for value in row.values():
            if ABSOLUTE_PATH_RE.search(clean(value)):
                violations.append("absolute_path")
    return violations


def run_guardrail_regression(args=None):
    rows, failures = [], 0
    for name in sorted(os.listdir(DATASET_DIR)):
        if not name.startswith("v2av_") or not name.endswith(".csv"):
            continue
        violations = scan_guardrails(dataset_path(name))
        failures += len(violations)
        rows.append({"regression_id": f"GR_v2av_{len(rows):03d}", "artifact_path": rel_dataset(name),
                     "violation_count": str(len(violations)), "status": "PASS" if not violations else "FAIL"})
    cache_ok = read_text(ensure_cache_policy()) == "*\n!.gitignore\n"
    failures += int(not cache_ok)
    rows.append({"regression_id": f"GR_v2av_{len(rows):03d}", "artifact_path": rel_doc("evidence_cache/.gitignore"),
                 "violation_count": "0" if cache_ok else "1", "status": "PASS" if cache_ok else "FAIL"})
    write_csv(dataset_path("v2av_guardrail_regression.csv"), rows)
    if failures:
        raise ValueError(f"v2av guardrail regression failed: {failures}")
    return rows


def _write_docs():
    ensure_cache_policy()
    write_markdown(doc_path("README.md"), [
        "# v2av Official Source Terms Snapshot and Manual Acquisition Pack", "",
        "Offline deterministic by default. Public download is not a public license.",
        "A terms snapshot is not legal clearance. No promotion or operational truth is created.",
    ])
    write_markdown(os.path.join(DOCS_DIR, "manual_acquisition_packets", "README.md"),
                   ["# Manual acquisition packets", "", "Human review instructions only."])
    write_markdown(os.path.join(DOCS_DIR, "source_snapshots", "README.md"),
                   ["# Source snapshots", "", "Snapshot indexes and metadata only; light text stays in ignored cache."])


ORCHESTRATION = [
    ("terms_snapshot_registry", run_build_terms_snapshot_registry, "v2av_source_terms_snapshot_registry.csv"),
    ("access_page_registry", run_build_access_page_registry, "v2av_official_access_page_registry.csv"),
    ("manual_acquisition_packets", run_build_manual_acquisition_packets, "v2av_manual_acquisition_packet_index.csv"),
    ("license_resolution_candidates", run_build_license_resolution_candidates, "v2av_license_resolution_candidates.csv"),
    ("temporal_observation_targets", run_build_temporal_observation_targets, "v2av_temporal_observation_targets.csv"),
    ("source_priority_recheck", run_recheck_source_priority, "v2av_source_priority_recheck.csv"),
    ("unresolved_terms_report", run_generate_unresolved_terms_report, "v2av_unresolved_terms_report.csv"),
    ("guardrail_regression", run_guardrail_regression, "v2av_guardrail_regression.csv"),
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
            write_csv(dataset_path("v2av_orchestrator_manifest.csv"), rows)
            raise ValueError(notes)
    write_csv(dataset_path("v2av_orchestrator_manifest.csv"), rows)
    return rows

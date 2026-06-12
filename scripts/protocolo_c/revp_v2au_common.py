#!/usr/bin/env python3
"""v2au Source Resolution and Evidence Acquisition Plan.

Transforms v2at blockers into auditable resolution work. It never creates
ground truth, patch truth, labels, negatives, training readiness, or promotion.
"""

import argparse
import csv
import hashlib
import os
import re
import urllib.request

PROTOCOL_VERSION = "v2au"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2au_source_resolution_plan")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DOCS_DIR, "evidence_cache"))
NETWORK_ENV = "V2AU_NETWORK"
HTTP_TIMEOUT = 8
MAX_CACHE_BYTES = 2 * 1024 * 1024
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")

INPUTS = {
    "facts": "v2at_fact_assertion_registry.csv",
    "gaps": "v2at_non_fact_gap_report.csv",
    "blockers": "v2at_promotion_blocker_audit.csv",
    "downloads": "v2at_download_target_manifest.csv",
}
INVARIANTS = {
    "can_create_ground_truth": "false",
    "can_create_patch_truth": "false",
    "can_create_label": "false",
    "can_create_negative": "false",
    "can_train_model": "false",
    "quickview_can_promote": "false",
    "susceptibility_can_promote": "false",
    "dino_can_decide_event": "false",
    "source_resolution_is_not_ground_truth": "true",
}
HYDROMET_SOURCES = "INMET|CEMADEN|ANA_HIDROWEB"


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
    return f"docs/protocolo_c/v2au_source_resolution_plan/{name}"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    if not os.path.basename(path).startswith("v2au_"):
        raise ValueError(f"Refusing non-v2au output: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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


def load_v2at_inputs():
    stack = {}
    for key, name in INPUTS.items():
        path = dataset_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required v2at input missing: {path}")
        stack[key] = load_csv(path)
    return stack


def ensure_cache_policy():
    os.makedirs(CACHE_DIR, exist_ok=True)
    marker = cache_path(".gitignore")
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("*\n!.gitignore\n")
    return marker


def validate_cache_policy():
    marker = ensure_cache_policy()
    return read_text(marker) == "*\n!.gitignore\n"


def is_network_enabled():
    return clean(os.environ.get(NETWORK_ENV)) == "1"


def attempt_endpoint(url, source_id):
    ensure_cache_policy()
    if not is_network_enabled():
        return {"attempt_status": "NETWORK_DISABLED_DETERMINISTIC_RUN", "cache_path": "",
                "cache_sha256": "", "raw_data_versioned": "false"}
    request = urllib.request.Request(clean(url), headers={"User-Agent": "REV-P-v2au/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            payload = response.read(MAX_CACHE_BYTES + 1)
        if len(payload) > MAX_CACHE_BYTES:
            return {"attempt_status": "PAYLOAD_TOO_LARGE_NOT_CACHED", "cache_path": "",
                    "cache_sha256": "", "raw_data_versioned": "false"}
        name = f"{re.sub(r'[^a-z0-9]+', '-', source_id.lower()).strip('-')}.cache"
        path = cache_path(name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return {"attempt_status": "CACHED_FOR_LOCAL_REVIEW", "cache_path": rel_doc(f"evidence_cache/{name}"),
                "cache_sha256": sha256_file(path), "raw_data_versioned": "false"}
    except Exception as exc:
        return {"attempt_status": f"ENDPOINT_FAILED_{type(exc).__name__.upper()}", "cache_path": "",
                "cache_sha256": "", "raw_data_versioned": "false"}


def classify_license(value):
    upper = clean(value).upper()
    if upper in {"NOT_APPLICABLE", "N/A"}:
        return "NOT_APPLICABLE"
    if any(token in upper for token in ("RESTRICTED", "PROPRIETARY", "NO_REDISTRIBUTION")):
        return "EXPLICIT_RESTRICTED"
    if any(token in upper for token in ("EXPLICIT_OPEN", "PUBLIC_OPEN", "CC-", "OPEN_LICENSE")):
        return "EXPLICIT_OPEN"
    return "UNKNOWN"


def classify_crs(value, evidence_role=""):
    upper = clean(value).upper()
    if clean(evidence_role).upper() in {"DOCUMENTARY_EVENT", "OBSERVED_MEASUREMENT"}:
        return "NOT_SPATIAL"
    if "EPSG:" in upper or upper.startswith("EPSG_"):
        return "EPSG_EXPLICIT"
    if upper and not any(token in upper for token in ("UNKNOWN", "PENDING", "NOT_DOCUMENTED", "NEEDS_")):
        return "CRS_DOCUMENTED"
    return "CRS_UNKNOWN"


def action_for_blocker(blocker):
    upper = clean(blocker).upper()
    if "LICENSE" in upper:
        return "MANUALLY_RESOLVE_LICENSE_AND_TERMS"
    if "CRS" in upper:
        return "MANUALLY_RESOLVE_CRS"
    if "TEMPORAL" in upper:
        return "DOWNLOAD_OFFICIAL_HYDROMETEOROLOGICAL_SERIES_FOR_REVIEW_PACKETS"
    if "GEOMETRY" in upper or "MEASUREMENT" in upper:
        return "ACQUIRE_EXPLICIT_GEOMETRY_OR_OBSERVED_MEASUREMENT"
    if "HAZARD" in upper:
        return "RESOLVE_HAZARD_TYPE_FROM_OFFICIAL_SOURCE"
    return "BUILD_MANUAL_REVIEW_PACKET"


def blocker_count(fact):
    checks = ["source_identified", "license_explicit", "crs_resolved", "observed_event",
              "temporal_compatible", "hazard_typed", "geometry_or_measurement_compatible",
              "human_review_complete", "independent_corroboration"]
    return sum(1 for field in checks if not is_true(fact.get(field)))


def priority_band(fact):
    count = blocker_count(fact)
    return "HIGH" if count <= 3 else "MEDIUM" if count <= 5 else "LOW"


def run_validate_v2at_inputs(args=None):
    stack = load_v2at_inputs()
    rows = []
    for key, name in INPUTS.items():
        rows.append(with_invariants({
            "input_id": f"INPUT_v2au_{len(rows)+1:02d}", "input_role": key,
            "input_path": rel_dataset(name), "row_count": str(len(stack[key])),
            "validation_status": "PRESENT_AND_READABLE",
        }))
    write_csv(dataset_path("v2au_input_validation.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_resolution_task_registry(args=None):
    stack = load_v2at_inputs()
    gap_by_assertion = {r["assertion_id"]: r for r in stack["gaps"]}
    rows = []
    for number, fact in enumerate(stack["facts"], 1):
        blocker = clean(fact.get("critical_blocker")) or clean(gap_by_assertion.get(fact["assertion_id"], {}).get("blocking_gap"))
        rows.append(with_invariants({
            "task_id": f"TASK_v2au_{number:04d}", "assertion_id": fact["assertion_id"],
            "candidate_id": fact["candidate_id"], "source_id": fact["source_id"],
            "source_name": fact["source_name"], "factual_blocker": blocker,
            "resolution_action": action_for_blocker(blocker), "blocker_count": str(blocker_count(fact)),
            "priority_band": priority_band(fact), "task_status": "OPEN_AUDITABLE_RESOLUTION_TASK",
        }))
    write_csv(dataset_path("v2au_resolution_task_registry.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_source_contact_and_terms_audit(args=None):
    stack = load_v2at_inputs()
    seen, rows = set(), []
    for fact in stack["facts"]:
        key = fact["source_id"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(with_invariants({
            "audit_id": f"TERMS_v2au_{len(rows)+1:04d}", "source_id": key,
            "source_name": fact["source_name"], "source_contact_status": "CONTACT_OR_TERMS_PAGE_NEEDS_MANUAL_RESOLUTION",
            "terms_status": "UNKNOWN", "license_status": "UNKNOWN",
            "required_human_action": "Locate official contact or terms page and record explicit reuse conditions.",
        }))
    write_csv(dataset_path("v2au_source_contact_and_terms_audit.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_license_resolution_matrix(args=None):
    rows = []
    for number, fact in enumerate(load_v2at_inputs()["facts"], 1):
        status = "EXPLICIT_OPEN" if is_true(fact.get("license_explicit")) else "UNKNOWN"
        rows.append(with_invariants({
            "license_id": f"LIC_v2au_{number:04d}", "assertion_id": fact["assertion_id"],
            "source_id": fact["source_id"], "license_resolution_status": classify_license(status),
            "reuse_allowed": boolean(status == "EXPLICIT_OPEN"), "manual_review_required": boolean(status != "EXPLICIT_OPEN"),
        }))
    write_csv(dataset_path("v2au_license_resolution_matrix.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_crs_resolution_matrix(args=None):
    rows = []
    for number, fact in enumerate(load_v2at_inputs()["facts"], 1):
        status = classify_crs("CRS_DOCUMENTED" if is_true(fact.get("crs_resolved")) else "CRS_UNKNOWN",
                              fact.get("evidence_role"))
        rows.append(with_invariants({
            "crs_id": f"CRS_v2au_{number:04d}", "assertion_id": fact["assertion_id"],
            "source_id": fact["source_id"], "evidence_role": fact["evidence_role"],
            "crs_resolution_status": status, "manual_review_required": boolean(status == "CRS_UNKNOWN"),
        }))
    write_csv(dataset_path("v2au_crs_resolution_matrix.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_temporal_data_request_plan(args=None):
    rows = []
    for number, fact in enumerate(load_v2at_inputs()["facts"], 1):
        rows.append(with_invariants({
            "request_id": f"TIME_v2au_{number:04d}", "assertion_id": fact["assertion_id"],
            "candidate_id": fact["candidate_id"], "requested_sources": HYDROMET_SOURCES,
            "request_scope": "Official hydrometeorological series covering the documented event window.",
            "temporal_series_required": "true", "request_status": "MANUAL_DOWNLOAD_OR_ENDPOINT_RESOLUTION_REQUIRED",
        }))
    write_csv(dataset_path("v2au_temporal_data_request_plan.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_manual_download_instructions(args=None):
    rows = []
    for number, target in enumerate(load_v2at_inputs()["downloads"], 1):
        rows.append(with_invariants({
            "instruction_id": f"MANUAL_v2au_{number:03d}", "source_id": target["source_id"],
            "target_url": target["target_url"],
            "step_1": "Open the official target URL and record access date and terms page.",
            "step_2": "Select the event period and smallest relevant official spatial scope.",
            "step_3": "Download only for local review cache and record checksum and metadata.",
            "step_4": "Do not version raw payload; update resolution matrices after human review.",
            "instruction_status": "REPRODUCIBLE_MANUAL_PLAN",
        }))
    write_csv(dataset_path("v2au_manual_download_instructions.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_source_endpoint_attempts(args=None):
    rows = []
    for number, target in enumerate(load_v2at_inputs()["downloads"], 1):
        rows.append(with_invariants({
            "attempt_id": f"ENDPOINT_v2au_{number:03d}", "source_id": target["source_id"],
            "target_url": target["target_url"], "network_enabled": boolean(is_network_enabled()),
            **attempt_endpoint(target["target_url"], target["source_id"]),
        }))
    write_csv(dataset_path("v2au_source_endpoint_attempts.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_review_packet_priority_queue(args=None):
    tasks = load_csv(dataset_path("v2au_resolution_task_registry.csv"))
    tasks.sort(key=lambda r: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}[r["priority_band"]],
                              int(r["blocker_count"]), r["source_name"], r["assertion_id"]))
    rows = [with_invariants({
        "queue_rank": str(rank), "task_id": task["task_id"], "assertion_id": task["assertion_id"],
        "source_id": task["source_id"], "source_name": task["source_name"],
        "priority_band": task["priority_band"], "blocker_count": task["blocker_count"],
        "review_packet_status": "READY_TO_ASSEMBLE_FOR_HUMAN_SOURCE_RESOLUTION",
    }) for rank, task in enumerate(tasks, 1)]
    write_csv(dataset_path("v2au_review_packet_priority_queue.csv"), list(rows[0].keys()), rows)
    return rows


def run_build_blocker_to_action_map(args=None):
    blockers = sorted({r["factual_blocker"] for r in load_csv(dataset_path("v2au_resolution_task_registry.csv"))})
    rows = [with_invariants({
        "map_id": f"MAP_v2au_{number:03d}", "factual_blocker": blocker,
        "resolution_action": action_for_blocker(blocker), "resolution_closes_fact_automatically": "false",
        "human_review_required": "true",
    }) for number, blocker in enumerate(blockers, 1)]
    write_csv(dataset_path("v2au_blocker_to_action_map.csv"), list(rows[0].keys()), rows)
    return rows


def run_next_action_ranker(args=None):
    licenses = load_csv(dataset_path("v2au_license_resolution_matrix.csv"))
    unknown = sum(1 for row in licenses if row["license_resolution_status"] == "UNKNOWN")
    actions = [
        ("MANUALLY_RESOLVE_LICENSE_AND_CRS_FOR_HIGH_PRIORITY_SOURCES", 100 if unknown else 70),
        ("DOWNLOAD_OFFICIAL_HYDROMETEOROLOGICAL_SERIES_FOR_REVIEW_PACKETS", 90),
        ("BUILD_SOURCE_RESOLUTION_REVIEW_PACKETS", 80),
        ("MAINTAIN_SOURCE_RESOLUTION_AS_NON_GROUND_TRUTH", 60),
    ]
    rows = [with_invariants({
        "rank": str(rank), "next_action": action, "score": str(score),
        "allowed_resolution_work": "true", "promotion_allowed": "false",
    }) for rank, (action, score) in enumerate(sorted(actions, key=lambda item: (-item[1], item[0])), 1)]
    write_csv(dataset_path("v2au_next_actions_registry.csv"), list(rows[0].keys()), rows)
    return rows


def _artifacts():
    if not os.path.isdir(DATASET_DIR):
        return []
    return [dataset_path(name) for name in sorted(os.listdir(DATASET_DIR))
            if name.startswith("v2au_") and name.endswith(".csv")]


def scan_guardrails(path):
    violations = []
    for row in load_csv(path):
        for field, expected in INVARIANTS.items():
            if field in row and clean(row[field]).lower() != expected:
                violations.append(field)
        if is_true(row.get("raw_data_versioned")) or is_true(row.get("promotion_allowed")):
            violations.append("raw_or_promotion")
        for value in row.values():
            if ABSOLUTE_PATH_RE.search(clean(value)):
                violations.append("absolute_path")
    return violations


def run_guardrail_regression(args=None):
    rows, failures = [], 0
    for path in _artifacts():
        violations = scan_guardrails(path)
        failures += len(violations)
        rows.append({"regression_id": f"GR_v2au_{len(rows):04d}", "artifact_path": rel_dataset(os.path.basename(path)),
                     "violation_count": str(len(violations)), "status": "PASS" if not violations else "FAIL"})
    cache_ok = validate_cache_policy()
    failures += int(not cache_ok)
    rows.append({"regression_id": f"GR_v2au_{len(rows):04d}", "artifact_path": rel_doc("evidence_cache/.gitignore"),
                 "violation_count": "0" if cache_ok else "1", "status": "PASS" if cache_ok else "FAIL"})
    write_csv(dataset_path("v2au_guardrail_regression.csv"), list(rows[0].keys()), rows)
    if failures:
        raise ValueError(f"v2au guardrail regression failed: {failures}")
    return rows


def _write_docs():
    ensure_cache_policy()
    write_markdown(doc_path("README.md"), [
        "# v2au Source Resolution and Evidence Acquisition Plan", "",
        "Offline deterministic by default. Source resolution is not ground truth.",
        "No labels, negatives, patch truth, training, promotion, or raw versioning.",
    ])
    write_markdown(os.path.join(DOCS_DIR, "manual_review_packets", "README.md"), [
        "# Manual review packets", "", "Packets organize human source resolution only."
    ])
    write_markdown(os.path.join(DOCS_DIR, "download_attempts", "README.md"), [
        "# Download attempts", "", "V2AU_NETWORK=1 is opt-in; payloads remain in the ignored cache."
    ])


ORCHESTRATION = [
    ("validate_v2at_inputs", run_validate_v2at_inputs, "v2au_input_validation.csv"),
    ("resolution_task_registry", run_build_resolution_task_registry, "v2au_resolution_task_registry.csv"),
    ("source_contact_and_terms_audit", run_build_source_contact_and_terms_audit, "v2au_source_contact_and_terms_audit.csv"),
    ("license_resolution_matrix", run_build_license_resolution_matrix, "v2au_license_resolution_matrix.csv"),
    ("crs_resolution_matrix", run_build_crs_resolution_matrix, "v2au_crs_resolution_matrix.csv"),
    ("temporal_data_request_plan", run_build_temporal_data_request_plan, "v2au_temporal_data_request_plan.csv"),
    ("manual_download_instructions", run_build_manual_download_instructions, "v2au_manual_download_instructions.csv"),
    ("source_endpoint_attempts", run_build_source_endpoint_attempts, "v2au_source_endpoint_attempts.csv"),
    ("review_packet_priority_queue", run_build_review_packet_priority_queue, "v2au_review_packet_priority_queue.csv"),
    ("blocker_to_action_map", run_build_blocker_to_action_map, "v2au_blocker_to_action_map.csv"),
    ("next_action_ranker", run_next_action_ranker, "v2au_next_actions_registry.csv"),
    ("guardrail_regression", run_guardrail_regression, "v2au_guardrail_regression.csv"),
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
            write_csv(dataset_path("v2au_orchestrator_manifest.csv"), list(rows[0].keys()), rows)
            raise ValueError(notes)
    write_csv(dataset_path("v2au_orchestrator_manifest.csv"), list(rows[0].keys()), rows)
    return rows


def run_all(args=None):
    return run_orchestrator(args)

#!/usr/bin/env python3
"""v2aw Public Data Provenance and Observational Acquisition."""

import argparse
import csv
import hashlib
import os
import re
import urllib.request

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2aw_public_data_observational_acquisition")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DOCS_DIR, "evidence_cache"))
NETWORK_ENV = "V2AW_NETWORK"
HTTP_TIMEOUT = 8
MAX_CACHE_BYTES = 512 * 1024
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
INPUTS = {
    "snapshots": "v2av_source_terms_snapshot_registry.csv",
    "manual_packets": "v2av_manual_acquisition_packet_index.csv",
    "tasks": "v2au_resolution_task_registry.csv",
    "facts": "v2at_fact_assertion_registry.csv",
}
GEOMETRY_INPUTS = ["v2as_observed_geometry_status.csv", "v2as_geojson_candidate_index.csv",
                   "v2as_geometry_candidate_classification.csv"]
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false",
    "can_train_model": "false", "license_blocker": "false",
    "provenance_required": "true", "source_publicity_is_not_ground_truth": "true",
    "temporal_observation_is_not_geometry": "true",
}
HYDROMET = {
    "INMET": "meteorological_time_series",
    "CEMADEN": "rainfall_hydrological_geotechnical_observation",
    "ANA_HIDROWEB": "hydrological_time_series",
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
    return f"docs/protocolo_c/v2aw_public_data_observational_acquisition/{name}"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not os.path.basename(path).startswith("v2aw_"):
        raise ValueError(f"Refusing non-v2aw output: {path}")
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
            raise FileNotFoundError(f"Required input missing: {path}")
        stack[key] = load_csv(path)
    stack["geometry"], stack["geometry_source"] = [], ""
    for name in GEOMETRY_INPUTS:
        path = dataset_path(name)
        if os.path.exists(path):
            stack["geometry"], stack["geometry_source"] = load_csv(path), rel_dataset(name)
            break
    return stack


def ensure_cache_policy():
    os.makedirs(CACHE_DIR, exist_ok=True)
    marker = cache_path(".gitignore")
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("*\n!.gitignore\n")
    return marker


def is_network_enabled():
    return clean(os.environ.get(NETWORK_ENV)) == "1"


def fetch_light_metadata(url, source_id):
    ensure_cache_policy()
    if not is_network_enabled():
        return {"acquisition_status": "NETWORK_DISABLED_DETERMINISTIC_RUN", "cache_path": "",
                "cache_sha256": "", "raw_data_versioned": "false"}
    request = urllib.request.Request(clean(url), headers={"User-Agent": "REV-P-v2aw/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            payload = response.read(MAX_CACHE_BYTES + 1)
        if len(payload) > MAX_CACHE_BYTES:
            return {"acquisition_status": "PAYLOAD_TOO_LARGE_NOT_CACHED", "cache_path": "",
                    "cache_sha256": "", "raw_data_versioned": "false"}
        name = re.sub(r"[^a-z0-9]+", "-", source_id.lower()).strip("-") + ".metadata"
        path = cache_path(name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return {"acquisition_status": "LIGHT_PUBLIC_METADATA_CACHED", "cache_path": rel_doc(f"evidence_cache/{name}"),
                "cache_sha256": sha256_file(path), "raw_data_versioned": "false"}
    except Exception as exc:
        return {"acquisition_status": f"ACQUISITION_FAILED_{type(exc).__name__.upper()}",
                "cache_path": "", "cache_sha256": "", "raw_data_versioned": "false"}


def reclassify_license(source_public=True, license_unknown=True, citation_complete=False):
    if not source_public:
        return "PUBLIC_SOURCE_PROVENANCE_PENDING"
    if citation_complete:
        return "PUBLIC_DATA_ACCEPTED_FOR_RESEARCH"
    if license_unknown:
        return "PUBLIC_SOURCE_PROVENANCE_PENDING"
    return "PUBLIC_DATA_CITATION_REQUIRED"


def observational_target_class(source_id, source_role=""):
    role = clean(source_role).upper()
    if source_id in HYDROMET or "OBSERVED_TIME_SERIES" in role:
        return "TEMPORAL_OBSERVATION_TARGET"
    if "SUSCEPTIBILITY" in role or "RISK" in role:
        return "CONTEXT_ONLY_SUSCEPTIBILITY"
    if "QUICKVIEW" in role:
        return "REVIEW_ONLY_QUICKVIEW"
    if "DINO" in role:
        return "REVIEW_ONLY_DINO_SIGNAL"
    return "PUBLIC_DATA_CITATION_REQUIRED"


def temporal_status(temporal_compatible=False, series_available=False):
    if temporal_compatible and series_available:
        return "TEMPORAL_OBSERVATION_READY"
    if temporal_compatible:
        return "TEMPORAL_OBSERVATION_TARGET"
    return "TEMPORAL_OBSERVATION_MISSING"


def package_status(fact):
    if not is_true(fact.get("temporal_compatible")):
        return "NOT_READY_FOR_REVIEW", "TEMPORAL_OBSERVATION_MISSING"
    if not is_true(fact.get("hazard_typed")):
        return "NOT_READY_FOR_REVIEW", "HAZARD_TYPING_REQUIRED"
    if not is_true(fact.get("geometry_or_measurement_compatible")):
        return "NOT_READY_FOR_REVIEW", "GEOMETRY_STILL_MISSING"
    return "EVENT_PATCH_PACKAGE_READY_FOR_REVIEW", ""


def run_reclassify_license_blockers(args=None):
    rows = []
    for number, fact in enumerate(load_inputs()["facts"], 1):
        rows.append(with_invariants({
            "reclassification_id": f"LIC_v2aw_{number:04d}", "assertion_id": fact["assertion_id"],
            "candidate_id": fact["candidate_id"], "source_id": fact["source_id"],
            "prior_classification": fact["fact_classification"], "prior_blocker": fact["critical_blocker"],
            "license_reclassification": reclassify_license(True, not is_true(fact.get("license_explicit")), False),
            "public_data_assumed": "true", "legal_use_assumed": "true", "source_citation_required": "true",
            "acquisition_allowed": "true", "scientific_gates_still_apply": "true",
        }))
    write_csv(dataset_path("v2aw_license_blocker_reclassification.csv"), rows)
    return rows


def run_build_public_source_provenance(args=None):
    rows = []
    for number, snapshot in enumerate(load_inputs()["snapshots"], 1):
        rows.append(with_invariants({
            "provenance_id": f"PROV_v2aw_{number:03d}", "source_id": snapshot["source_id"],
            "official_url": snapshot["official_url"], "institutional_source": "true",
            "public_data_assumed": "true", "legal_use_assumed": "true", "source_citation_required": "true",
            "access_date_required": "true", "provenance_status": "PUBLIC_SOURCE_PROVENANCE_PENDING",
        }))
    write_csv(dataset_path("v2aw_public_source_provenance_registry.csv"), rows)
    return rows


def run_build_observational_data_targets(args=None):
    stack = load_inputs()
    snapshots = {r["source_id"]: r for r in stack["snapshots"]}
    rows = []
    for number, source_id in enumerate(sorted(snapshots), 1):
        role = "OFFICIAL_OBSERVED_TIME_SERIES" if source_id in HYDROMET else (
            "SUSCEPTIBILITY_OR_RISK_CONTEXT" if source_id == "SGB_CPRM" else
            "QUICKVIEW_OR_PRODUCT" if source_id == "INTERNATIONAL_CHARTER" else "OFFICIAL_SPATIAL_PRODUCT")
        rows.append(with_invariants({
            "target_id": f"TARGET_v2aw_{number:03d}", "source_id": source_id,
            "official_url": snapshots[source_id]["official_url"], "source_role": role,
            "target_class": observational_target_class(source_id, role),
            "acquisition_priority": "HIGH" if source_id in HYDROMET else "MEDIUM",
            "public_data_assumed": "true", "legal_use_assumed": "true",
        }))
    write_csv(dataset_path("v2aw_observational_data_target_registry.csv"), rows)
    return rows


def run_build_hydrometeorological_acquisition_plan(args=None):
    snapshots = {r["source_id"]: r for r in load_inputs()["snapshots"]}
    rows = []
    for number, (source_id, observation_type) in enumerate(HYDROMET.items(), 1):
        snapshot = snapshots.get(source_id, {})
        attempt = fetch_light_metadata(snapshot.get("official_url"), source_id)
        rows.append(with_invariants({
            "plan_id": f"HYDRO_v2aw_{number:03d}", "source_id": source_id,
            "official_url": clean(snapshot.get("official_url")), "observation_type": observation_type,
            "network_enabled": boolean(is_network_enabled()), **attempt,
            "target_class": "TEMPORAL_OBSERVATION_TARGET", "source_citation_required": "true",
        }))
    write_csv(dataset_path("v2aw_hydrometeorological_acquisition_plan.csv"), rows)
    return rows


def run_build_temporal_window_candidates(args=None):
    rows = []
    for number, fact in enumerate(load_inputs()["facts"], 1):
        rows.append(with_invariants({
            "window_id": f"WINDOW_v2aw_{number:04d}", "assertion_id": fact["assertion_id"],
            "candidate_id": fact["candidate_id"], "event_id": fact["event_id"],
            "temporal_status": temporal_status(is_true(fact.get("temporal_compatible")), False),
            "hydrometeorological_sources": "|".join(HYDROMET), "human_validation_required": "true",
        }))
    write_csv(dataset_path("v2aw_temporal_window_candidate_registry.csv"), rows)
    return rows


def run_build_event_patch_observation_packages(args=None):
    manual = {r["assertion_id"]: r for r in load_inputs()["manual_packets"]}
    rows = []
    for number, fact in enumerate(load_inputs()["facts"], 1):
        status, blocker = package_status(fact)
        rows.append(with_invariants({
            "package_id": f"PKG_v2aw_{number:04d}", "assertion_id": fact["assertion_id"],
            "candidate_id": fact["candidate_id"], "event_id": fact["event_id"], "patch_id": fact["patch_id"],
            "manual_acquisition_packet_available": boolean(fact["assertion_id"] in manual),
            "temporal_status": temporal_status(is_true(fact.get("temporal_compatible")), False),
            "geometry_status": "GEOMETRY_COMPATIBLE" if is_true(fact.get("geometry_or_measurement_compatible")) else "GEOMETRY_STILL_MISSING",
            "hazard_status": "HAZARD_TYPED" if is_true(fact.get("hazard_typed")) else "HAZARD_TYPING_REQUIRED",
            "review_status": status, "blocking_reason": blocker,
        }))
    write_csv(dataset_path("v2aw_event_patch_observation_package_index.csv"), rows)
    return rows


def run_generate_observational_readiness_report(args=None):
    packages = load_csv(dataset_path("v2aw_event_patch_observation_package_index.csv"))
    ready = sum(1 for row in packages if row["review_status"] == "EVENT_PATCH_PACKAGE_READY_FOR_REVIEW")
    rows = [
        with_invariants({"report_id": "READ_v2aw_001", "metric": "event_patch_packages", "value": str(len(packages)),
                         "status": "RECORDED"}),
        with_invariants({"report_id": "READ_v2aw_002", "metric": "packages_ready_for_review", "value": str(ready),
                         "status": "RECORDED"}),
        with_invariants({"report_id": "READ_v2aw_003", "metric": "next_action_rank_1",
                         "value": "ACQUIRE_PUBLIC_HYDROMETEOROLOGICAL_TIME_SERIES_FOR_EVENT_PATCH_REVIEW",
                         "status": "SAFE_NEXT_ACTION"}),
    ]
    write_csv(dataset_path("v2aw_observational_readiness_report.csv"), rows)
    return rows


def scan_guardrails(path):
    violations = []
    for row in load_csv(path):
        for field, expected in INVARIANTS.items():
            if field in row and clean(row[field]).lower() != expected:
                violations.append(field)
        if is_true(row.get("raw_data_versioned")):
            violations.append("raw_data_versioned")
        for value in row.values():
            if ABSOLUTE_PATH_RE.search(clean(value)):
                violations.append("absolute_path")
    return violations


def run_guardrail_regression(args=None):
    rows, failures = [], 0
    for name in sorted(os.listdir(DATASET_DIR)):
        if name.startswith("v2aw_") and name.endswith(".csv"):
            violations = scan_guardrails(dataset_path(name))
            failures += len(violations)
            rows.append({"regression_id": f"GR_v2aw_{len(rows):03d}", "artifact_path": rel_dataset(name),
                         "violation_count": str(len(violations)), "status": "PASS" if not violations else "FAIL"})
    cache_ok = read_text(ensure_cache_policy()) == "*\n!.gitignore\n"
    failures += int(not cache_ok)
    rows.append({"regression_id": f"GR_v2aw_{len(rows):03d}", "artifact_path": rel_doc("evidence_cache/.gitignore"),
                 "violation_count": "0" if cache_ok else "1", "status": "PASS" if cache_ok else "FAIL"})
    write_csv(dataset_path("v2aw_guardrail_regression.csv"), rows)
    if failures:
        raise ValueError(f"v2aw guardrail regression failed: {failures}")
    return rows


def _write_docs():
    ensure_cache_policy()
    write_markdown(doc_path("README.md"), [
        "# v2aw Public Data Provenance and Observational Acquisition", "",
        "Public institutional data is accepted for research acquisition with provenance and citation.",
        "Source publicity and license reclassification do not create ground truth, labels, negatives, or training readiness.",
    ])
    write_markdown(os.path.join(DOCS_DIR, "acquisition_guides", "README.md"),
                   ["# Acquisition guides", "", "Acquire light public metadata and time series for review."])
    write_markdown(os.path.join(DOCS_DIR, "event_patch_packages", "README.md"),
                   ["# Event patch packages", "", "Packages remain review-only until scientific gates close."])


ORCHESTRATION = [
    ("license_blocker_reclassification", run_reclassify_license_blockers, "v2aw_license_blocker_reclassification.csv"),
    ("public_source_provenance", run_build_public_source_provenance, "v2aw_public_source_provenance_registry.csv"),
    ("observational_data_targets", run_build_observational_data_targets, "v2aw_observational_data_target_registry.csv"),
    ("hydrometeorological_acquisition_plan", run_build_hydrometeorological_acquisition_plan, "v2aw_hydrometeorological_acquisition_plan.csv"),
    ("temporal_window_candidates", run_build_temporal_window_candidates, "v2aw_temporal_window_candidate_registry.csv"),
    ("event_patch_observation_packages", run_build_event_patch_observation_packages, "v2aw_event_patch_observation_package_index.csv"),
    ("observational_readiness_report", run_generate_observational_readiness_report, "v2aw_observational_readiness_report.csv"),
    ("guardrail_regression", run_guardrail_regression, "v2aw_guardrail_regression.csv"),
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
            write_csv(dataset_path("v2aw_orchestrator_manifest.csv"), rows)
            raise ValueError(notes)
    write_csv(dataset_path("v2aw_orchestrator_manifest.csv"), rows)
    return rows

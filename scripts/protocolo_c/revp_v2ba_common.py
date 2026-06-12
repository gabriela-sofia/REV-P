#!/usr/bin/env python3
"""v2ba Official Geometry Search and Candidate Digitization Audit."""

import argparse
import csv
import hashlib
import json
import os
import re

STAGE = "v2ba"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2ba_official_geometry_search_and_digitization")
NETWORK_ENV = "V2BA_NETWORK"
INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false",
    "can_create_label": "false", "can_create_negative": "false", "can_train_model": "false",
    "candidate_geometry_is_not_ground_truth": "true", "candidate_digitization_is_not_label": "true",
    "official_source_is_not_automatic_truth": "true", "quickview_is_not_validated_product": "true",
    "susceptibility_is_not_observed_event": "true", "textual_location_is_not_geometry": "true",
    "rainfall_is_not_flood_extent": "true", "patch_boundary_is_not_event_geometry": "true",
    "raw_data_versioned": "false",
}
INPUTS = {
    "packets": "v2az_assisted_review_packet_index.csv",
    "manual": "v2az_manual_review_table.csv",
    "findings": "v2az_geometry_evidence_findings.csv",
    "targets": "v2az_geometry_source_search_targets.csv",
    "null_manifest": "v2az_null_geometry_review_geojson_manifest.csv",
    "readiness_report": "v2az_assisted_review_readiness_report.csv",
    "metrics": "v2ay_window_precipitation_metrics.csv",
    "temporal": "v2ay_event_patch_temporal_readiness_update.csv",
    "payload": "v2as_geometry_payload_detection.csv",
    "classification": "v2as_geometry_candidate_classification.csv",
    "sources": "v2ar_official_geometry_source_registry.csv",
}
OUTPUTS = [
    "v2ba_review_ready_packet_selection.csv", "v2ba_geometry_search_plan.csv",
    "v2ba_official_geometry_source_probe.csv", "v2ba_geometry_evidence_classification.csv",
    "v2ba_candidate_digitization_registry.csv", "v2ba_candidate_geojson_manifest.csv",
    "v2ba_candidate_geometry_validation.csv", "v2ba_geometry_uncertainty_scores.csv",
    "v2ba_human_adjudication_queue.csv", "v2ba_geometry_audit_report.csv",
    "v2ba_guardrail_regression.csv",
]


def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", clean(value).lower()).strip("-")


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(*parts):
    return os.path.join(DOCS_DIR, *parts)


def with_invariants(row):
    return {**row, **INVARIANTS}


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing empty output: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def by(rows, key):
    return {row.get(key, ""): row for row in rows}


def load_inputs():
    return {key: load_csv(dataset_path(name)) for key, name in INPUTS.items()}


def supported_packets():
    return load_inputs()["packets"]


def source_classification(source_type="", source_name="", has_geometry=False, found_map=False, quickview=False, susceptibility=False):
    kind, name = clean(source_type).upper(), clean(source_name).upper()
    if has_geometry: return "OFFICIAL_OBSERVED_GEOMETRY"
    if quickview or "IMAGE OF THE DAY" in name: return "QUICKVIEW_ONLY"
    if susceptibility or "SGB" in name or "CPRM" in name or "SUSCET" in name: return "SUSCEPTIBILITY_CONTEXT_ONLY"
    if found_map: return "OFFICIAL_MAP_PRODUCT_REQUIRES_VALIDATION"
    if kind in {"ACADEMIC_PAPER", "COPERNICUS_PRODUCT", "VHR_REVIEW"}: return "REVIEW_ONLY_VISUAL_SUPPORT"
    if kind in {"TECHNICAL_REPORT", "OFFICIAL_MUNICIPAL"}: return "TEXTUAL_LOCATION_ONLY"
    if kind == "OFFICIAL_VECTOR": return "CANDIDATE_GEOMETRY_FROM_REVIEWABLE_EVIDENCE"
    return "CONTEXT_ONLY"


def candidate_allowed(evidence_class):
    return evidence_class in {"OFFICIAL_OBSERVED_GEOMETRY", "OFFICIAL_MAP_PRODUCT_REQUIRES_VALIDATION", "CANDIDATE_GEOMETRY_FROM_REVIEWABLE_EVIDENCE"}


def digitization_method(evidence_class, source_type=""):
    if evidence_class == "OFFICIAL_OBSERVED_GEOMETRY": return "OFFICIAL_VECTOR"
    if evidence_class == "OFFICIAL_MAP_PRODUCT_REQUIRES_VALIDATION": return "OFFICIAL_RASTER_TRACE"
    if evidence_class == "CANDIDATE_GEOMETRY_FROM_REVIEWABLE_EVIDENCE": return "REVIEWABLE_VISUAL_TRACE"
    if evidence_class in {"TEXTUAL_LOCATION_ONLY", "REVIEW_ONLY_VISUAL_SUPPORT"}: return "MANUAL_DIGITIZATION_REQUIRED"
    return "NONE"


def flatten_coordinates(value):
    if not isinstance(value, list): return []
    if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]): return [value[:2]]
    result = []
    for item in value: result.extend(flatten_coordinates(item))
    return result


def validate_geometry_payload(geometry, crs=""):
    if not geometry:
        return {"geometry_present": "false", "geometry_valid": "false", "geometry_type": "", "crs_present": "false",
                "coordinates_within_brazil": "false", "area_reasonable": "false", "validation_status": "NULL_GEOMETRY_VALID_FOR_BLOCKED_CANDIDATE",
                "fail_reason": "NO_TRACEABLE_GEOMETRY"}
    coordinates = flatten_coordinates(geometry.get("coordinates"))
    valid_type = geometry.get("type") in {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}
    brazil = bool(coordinates) and all(-74 <= point[0] <= -34 and -34 <= point[1] <= 6 for point in coordinates)
    valid = valid_type and brazil and bool(crs)
    reasons = []
    if not valid_type: reasons.append("INVALID_GEOMETRY_TYPE")
    if not crs: reasons.append("CRS_MISSING")
    if not brazil: reasons.append("COORDINATES_OUTSIDE_BRAZIL_OR_MISSING")
    return {"geometry_present": "true", "geometry_valid": str(valid).lower(), "geometry_type": geometry.get("type", ""),
            "crs_present": str(bool(crs)).lower(), "coordinates_within_brazil": str(brazil).lower(),
            "area_reasonable": "REQUIRES_HUMAN_REVIEW", "validation_status": "CANDIDATE_REQUIRES_HUMAN_VALIDATION" if valid else "VALIDATION_FAILED",
            "fail_reason": "|".join(reasons)}


def run_select_review_ready_packets(args=None):
    data = load_inputs()
    temporal = by(data["temporal"], "assertion_id")
    rows = []
    for packet in data["packets"]:
        rows.append(with_invariants({"selection_id": f"SEL_v2ba_{len(rows)+1:04d}", "review_packet_id": packet["review_packet_id"],
            "event_patch_package_id": packet["event_patch_package_id"], "candidate_id": packet["candidate_id"], "region": packet["region"],
            "city": packet["city"], "event_date": packet["event_date"], "temporal_support_status": temporal.get(packet["event_patch_package_id"], {}).get("temporal_readiness_status", ""),
            "selection_status": "REVIEW_READY_SELECTED", "recife_gap_excluded": "false"}))
    write_csv(dataset_path(OUTPUTS[0]), rows)
    return rows


def run_build_geometry_search_plan(args=None):
    data = load_inputs()
    packets = by(data["packets"], "candidate_id")
    rows = []
    for source in data["sources"]:
        packet = packets.get(source["candidate_id"])
        if not packet: continue
        rows.append(with_invariants({"search_task_id": f"SEARCH_v2ba_{len(rows)+1:04d}", "review_packet_id": packet["review_packet_id"],
            "event_patch_package_id": packet["event_patch_package_id"], "candidate_id": packet["candidate_id"], "region": packet["region"],
            "city": packet["city"], "patch_id": packet["patch_id"], "event_date": packet["event_date"],
            "temporal_support_status": "TEMPORAL_EVIDENCE_READY_FOR_REVIEW", "source_target": source["source_name"],
            "source_type": source["source_type"], "search_url_or_reference": source["source_url_or_document"],
            "expected_geometry_type": source["expected_geometry_type"], "search_priority": "HIGH" if source["source_role"] == "primary" else "MEDIUM",
            "search_reason": "Search official or reviewable geometry without automatic promotion.", "can_promote_directly": "false"}))
    write_csv(dataset_path(OUTPUTS[1]), rows)
    for packet in data["packets"]:
        tasks = [row for row in rows if row["review_packet_id"] == packet["review_packet_id"]]
        lines = "\n".join(f"- {row['search_priority']}: {row['source_target']} - {row['search_url_or_reference']}" for row in tasks)
        write_text(doc_path("search_plans", f"{slug(packet['candidate_id'])}.md"),
                   f"# Geometry search plan: {packet['candidate_id']}\n\nEu/equipe devo abrir e revisar:\n{lines}\n\nNenhuma fonte promove truth diretamente.\n")
    return rows


def probe_flags(task):
    evidence = source_classification(task["source_type"], task["source_target"])
    return {
        "found_geometry_payload": "false", "found_map_product": "false",
        "found_textual_location": str(evidence == "TEXTUAL_LOCATION_ONLY").lower(),
        "found_quickview": str(evidence == "QUICKVIEW_ONLY").lower(),
        "found_susceptibility_context": str(evidence == "SUSCEPTIBILITY_CONTEXT_ONLY").lower(),
        "found_official_product": str(task["source_type"] in {"TECHNICAL_REPORT", "OFFICIAL_MUNICIPAL", "COPERNICUS_PRODUCT"}).lower(),
    }


def run_probe_official_geometry_sources(args=None):
    rows = []
    network = os.environ.get(NETWORK_ENV) == "1"
    for task in load_csv(dataset_path(OUTPUTS[1])):
        flags = probe_flags(task)
        status = "NETWORK_METADATA_NOT_IMPLEMENTED_REQUIRES_MANUAL_REVIEW" if network else "NETWORK_DISABLED_DETERMINISTIC_RUN"
        rows.append(with_invariants({"probe_id": f"PROBE_v2ba_{len(rows)+1:04d}", "search_task_id": task["search_task_id"],
            "review_packet_id": task["review_packet_id"], "event_patch_package_id": task["event_patch_package_id"],
            "source_target": task["source_target"], "source_url": task["search_url_or_reference"],
            "probe_mode": "NETWORK_METADATA" if network else "OFFLINE_DETERMINISTIC", "status": status, "http_status": "",
            **flags, "raw_payload_cached": "false", "probe_note": "Eu/equipe registrei a referencia; payload explicito nao foi observado no estado local."}))
    write_csv(dataset_path(OUTPUTS[2]), rows)
    for review_packet_id in sorted({row["review_packet_id"] for row in rows}):
        matches = [row for row in rows if row["review_packet_id"] == review_packet_id]
        lines = "\n".join(f"- {row['source_target']}: {row['status']}; geometry payload={row['found_geometry_payload']}" for row in matches)
        write_text(doc_path("source_probe_summaries", f"{slug(review_packet_id)}.md"),
                   f"# Source probe summary: {review_packet_id}\n\nEu/equipe registrei {len(matches)} sondagens offline deterministicas.\n\n{lines}\n\nNenhum payload bruto foi versionado.\n")
    return rows


def run_classify_geometry_evidence(args=None):
    tasks = by(load_csv(dataset_path(OUTPUTS[1])), "search_task_id")
    rows = []
    for probe in load_csv(dataset_path(OUTPUTS[2])):
        task = tasks[probe["search_task_id"]]
        evidence = source_classification(task["source_type"], task["source_target"], is_true(probe["found_geometry_payload"]),
                                         is_true(probe["found_map_product"]), is_true(probe["found_quickview"]), is_true(probe["found_susceptibility_context"]))
        allowed = candidate_allowed(evidence)
        rows.append(with_invariants({"classification_id": f"CLASS_v2ba_{len(rows)+1:04d}", "review_packet_id": probe["review_packet_id"],
            "event_patch_package_id": probe["event_patch_package_id"], "search_task_id": probe["search_task_id"], "evidence_class": evidence,
            "source_name": task["source_target"], "source_reference": task["search_url_or_reference"], "geometry_candidate_allowed": str(allowed).lower(),
            "manual_digitization_allowed": str(evidence in {"OFFICIAL_MAP_PRODUCT_REQUIRES_VALIDATION", "CANDIDATE_GEOMETRY_FROM_REVIEWABLE_EVIDENCE"}).lower(),
            "automatic_digitization_allowed": "false", "requires_human_validation": "true", "requires_adjudication": "true",
            "reason": "No explicit traceable geometry in deterministic local evidence; textual/visual/contextual evidence cannot become geometry automatically."}))
    write_csv(dataset_path(OUTPUTS[3]), rows)
    return rows


def run_build_candidate_digitization_registry(args=None):
    data = load_inputs()
    classes = load_csv(dataset_path(OUTPUTS[3]))
    rows = []
    for packet in data["packets"]:
        matches = [row for row in classes if row["review_packet_id"] == packet["review_packet_id"] and is_true(row["geometry_candidate_allowed"])]
        best = matches[0] if matches else {}
        evidence = best.get("evidence_class", "NO_GEOMETRY_FOUND")
        created = bool(matches)
        rows.append(with_invariants({"candidate_geometry_id": f"CG_v2ba_{packet['review_packet_id'].split('_')[-1]}",
            "review_packet_id": packet["review_packet_id"], "event_patch_package_id": packet["event_patch_package_id"],
            "candidate_id": packet["candidate_id"], "candidate_source_type": best.get("evidence_class", ""),
            "candidate_source_reference": best.get("source_reference", ""), "digitization_method": digitization_method(evidence),
            "geometry_type": "", "crs": "", "spatial_precision": "UNKNOWN", "temporal_precision": "EVENT_WINDOW",
            "hazard_type_candidate": packet["hazard_type_candidate"], "uncertainty_level": "UNKNOWN" if created else "VERY_HIGH",
            "candidate_status": "CANDIDATE_REQUIRES_HUMAN_VALIDATION" if created else "NOT_CREATED",
            "blocker_reason": "" if created else "NO_TRACEABLE_GEOMETRY; TEXTUAL_OR_VISUAL_CONTEXT_CANNOT_BE_AUTO_DIGITIZED"}))
    write_csv(dataset_path(OUTPUTS[4]), rows)
    return rows


def run_generate_candidate_geojsons(args=None):
    packets = by(load_inputs()["packets"], "review_packet_id")
    rows = []
    for candidate in load_csv(dataset_path(OUTPUTS[4])):
        packet = packets[candidate["review_packet_id"]]
        name = f"{slug(packet['candidate_id'])}.geojson"
        path = doc_path("candidate_geojsons", name)
        payload = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": None, "properties": {
            "candidate_geometry_id": candidate["candidate_geometry_id"], "review_packet_id": candidate["review_packet_id"],
            "event_patch_package_id": candidate["event_patch_package_id"], "candidate_status": candidate["candidate_status"],
            "blocker_reason": candidate["blocker_reason"], "geometry_null_allowed": True, "validation_required": True,
            "can_create_ground_truth": False, "can_create_label": False}}]}
        write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        rows.append(with_invariants({"candidate_geometry_id": candidate["candidate_geometry_id"], "review_packet_id": candidate["review_packet_id"],
            "event_patch_package_id": candidate["event_patch_package_id"], "geojson_path": f"docs/protocolo_c/v2ba_official_geometry_search_and_digitization/candidate_geojsons/{name}",
            "geojson_sha256": sha256(path), "geometry_present": "false", "geometry_null_allowed": "true",
            "candidate_status": candidate["candidate_status"], "blocker_reason": candidate["blocker_reason"]}))
    write_csv(dataset_path(OUTPUTS[5]), rows)
    return rows


def run_validate_candidate_geometries(args=None):
    rows = []
    for manifest in load_csv(dataset_path(OUTPUTS[5])):
        result = validate_geometry_payload(None)
        rows.append(with_invariants({"candidate_geometry_id": manifest["candidate_geometry_id"], **result,
            "crs_present": "false", "coordinates_within_expected_region": "false", "source_traceable": "false",
            "validation_required": "true", "can_promote": "false"}))
    write_csv(dataset_path(OUTPUTS[6]), rows)
    return rows


def uncertainty_level(candidate_status, hazard_type):
    if candidate_status != "CANDIDATE_REQUIRES_HUMAN_VALIDATION": return "VERY_HIGH"
    if clean(hazard_type).upper() in {"MIXED", "HAZARD_TYPED_BUT_REQUIRES_HUMAN_SEPARATION"}: return "HIGH"
    return "MODERATE"


def run_compute_geometry_uncertainty(args=None):
    rows = []
    for candidate in load_csv(dataset_path(OUTPUTS[4])):
        overall = uncertainty_level(candidate["candidate_status"], candidate["hazard_type_candidate"])
        rows.append(with_invariants({"candidate_geometry_id": candidate["candidate_geometry_id"], "review_packet_id": candidate["review_packet_id"],
            "spatial_uncertainty": "VERY_HIGH", "temporal_uncertainty": "LOW", "source_uncertainty": "HIGH",
            "hazard_uncertainty": "HIGH", "digitization_uncertainty": "VERY_HIGH", "overall_uncertainty": overall,
            "uncertainty_reason": "No explicit geometry or CRS; hazard and digitization require human review.",
            "reviewer_warning": "Do not treat candidate or null geometry as event truth."}))
    write_csv(dataset_path(OUTPUTS[7]), rows)
    return rows


def run_build_human_adjudication_queue(args=None):
    rows = []
    for candidate in load_csv(dataset_path(OUTPUTS[4])):
        rows.append(with_invariants({"adjudication_id": f"ADJ_v2ba_{len(rows)+1:04d}", "review_packet_id": candidate["review_packet_id"],
            "event_patch_package_id": candidate["event_patch_package_id"], "package_status": "REVIEW_READY_TEMPORALLY_SUPPORTED",
            "geometry_candidate_status": candidate["candidate_status"], "reason_for_adjudication": candidate["blocker_reason"] or "CANDIDATE_REQUIRES_HUMAN_VALIDATION",
            "required_human_action": "Open official/reviewable sources and either digitize traceably or keep geometry missing.",
            "suggested_sources_to_open": candidate["candidate_source_reference"] or "v2ba_geometry_search_plan.csv",
            "decision_options": "ACCEPT_AS_CANDIDATE_FOR_NEXT_REVIEW|REJECT_GEOMETRY_CANDIDATE|REQUEST_MORE_EVIDENCE|MARK_HAZARD_AMBIGUOUS|KEEP_GEOMETRY_MISSING",
            "current_truth_status": "NOT_GROUND_TRUTH"}))
    write_csv(dataset_path(OUTPUTS[8]), rows)
    for row in rows:
        write_text(doc_path("human_adjudication_packets", f"{slug(row['review_packet_id'])}.md"),
                   f"# Human adjudication: {row['review_packet_id']}\n\nEu/equipe devo revisar as fontes sugeridas e manter `NOT_GROUND_TRUTH`.\n\n- Status: {row['geometry_candidate_status']}\n- Razao: {row['reason_for_adjudication']}\n- Opcoes: {row['decision_options']}\n")
    return rows


def run_generate_geometry_audit_report(args=None):
    packets = load_inputs()["packets"]
    candidates = load_csv(dataset_path(OUTPUTS[4]))
    probes = load_csv(dataset_path(OUTPUTS[2]))
    created = sum(row["candidate_status"] == "CANDIDATE_REQUIRES_HUMAN_VALIDATION" for row in candidates)
    nulls = len(candidates) - created
    next_action = "HUMAN_ADJUDICATE_CANDIDATE_GEOMETRIES" if created else "MANUAL_DIGITIZATION_FROM_REVIEWABLE_EVIDENCE_OR_SECONDARY_SOURCE_SEARCH"
    values = [
        ("review_ready_packets", len(packets)), ("recife_gap_packets", 3), ("search_tasks", len(probes)),
        ("official_geometries_found", 0), ("candidate_geometries_created", created), ("null_geojsons", nulls),
        ("human_adjudication_packets", len(candidates)), ("ground_truth_created", 0), ("labels_created", 0),
        ("negatives_created", 0), ("training_runs", 0), ("next_action_rank_1", next_action),
        ("recife_next_action", "RESOLVE_RECIFE_TEMPORAL_GAP_WITH_CEMADEN_OR_SECONDARY_STATIONS"),
    ]
    rows = [with_invariants({"report_id": f"REPORT_v2ba_{number:03d}", "metric": metric, "value": str(value),
                             "status": "SAFE_NEXT_ACTION" if "action" in metric else "RECORDED"}) for number, (metric, value) in enumerate(values, 1)]
    write_csv(dataset_path(OUTPUTS[9]), rows)
    write_text(doc_path("README.md"), f"""# v2ba Official Geometry Search and Candidate Digitization Audit

Eu/equipe analisei {len(packets)} pacotes temporalmente suportados e mantive 3 pacotes Recife fora da busca principal.

- Tarefas/fontes propostas e auditadas offline: {len(probes)}
- Geometrias oficiais encontradas: 0
- Geometrias candidatas criadas: {created}
- GeoJSONs com `geometry: null`: {nulls}
- Pacotes para adjudicacao humana: {len(candidates)}
- Ground truth, labels, negativos e treino: 0
- Proxima acao: `{next_action}`

Geometria candidata nao e ground truth. Fonte textual, quickview, suscetibilidade, chuva e patch nao viram geometria de evento.
""")
    return rows


def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label", "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    for number, name in enumerate(OUTPUTS[:10], 1):
        violations = sum(row.get(field, "").lower() == "true" for row in load_csv(dataset_path(name)) for field in forbidden)
        rows.append({"regression_id": f"GR_v2ba_{number:03d}", "artifact_path": f"datasets/protocolo_c/{name}", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    marker = doc_path("evidence_cache", ".gitignore")
    violations = 0 if os.path.exists(marker) and open(marker, encoding="utf-8").read() == "*\n!.gitignore\n" else 1
    rows.append({"regression_id": "GR_v2ba_011", "artifact_path": "docs/protocolo_c/v2ba_official_geometry_search_and_digitization/evidence_cache/.gitignore", "violation_count": str(violations), "status": "PASS" if not violations else "FAIL"})
    if any(row["status"] != "PASS" for row in rows): raise ValueError("v2ba guardrail regression failed")
    write_csv(dataset_path(OUTPUTS[10]), rows)
    return rows


STEPS = [
    ("select_review_ready_packets", run_select_review_ready_packets, OUTPUTS[0]),
    ("build_geometry_search_plan", run_build_geometry_search_plan, OUTPUTS[1]),
    ("probe_official_geometry_sources", run_probe_official_geometry_sources, OUTPUTS[2]),
    ("classify_geometry_evidence", run_classify_geometry_evidence, OUTPUTS[3]),
    ("build_candidate_digitization_registry", run_build_candidate_digitization_registry, OUTPUTS[4]),
    ("generate_candidate_geojsons", run_generate_candidate_geojsons, OUTPUTS[5]),
    ("validate_candidate_geometries", run_validate_candidate_geometries, OUTPUTS[6]),
    ("compute_geometry_uncertainty", run_compute_geometry_uncertainty, OUTPUTS[7]),
    ("build_human_adjudication_queue", run_build_human_adjudication_queue, OUTPUTS[8]),
    ("generate_geometry_audit_report", run_generate_geometry_audit_report, OUTPUTS[9]),
    ("guardrail_regression", run_guardrail_regression, OUTPUTS[10]),
]


def ensure_structure():
    for folder in ("search_plans", "source_probe_summaries", "candidate_geojsons", "human_adjudication_packets", "evidence_cache"):
        os.makedirs(doc_path(folder), exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure()
    manifest = []
    for number, (name, function, output) in enumerate(STEPS, 1):
        function(args)
        path = dataset_path(output)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK", "output": f"datasets/protocolo_c/{output}", "output_hash": sha256(path)[:16], "notes": "Completed."})
    write_csv(dataset_path("v2ba_orchestrator_manifest.csv"), manifest)
    return manifest

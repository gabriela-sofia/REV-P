#!/usr/bin/env python3
"""v2az Turning Point Replay Orchestrator + Real Geometry Intake Adapter.

Validates v2ax manual intake, promotes only valid real geometries into formal
feeds, and computes TP0-TP4 readiness. Replay subprocesses remain fail-closed
and use an isolated workspace, never canonical prior-stage outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

try:
    import v2ax_recife_geometry_intake_pack_engine as v2ax
except ModuleNotFoundError:
    from scripts import v2ax_recife_geometry_intake_pack_engine as v2ax


STAGE = "v2az_turning_point_replay_orchestrator"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_NAME = "v2az_turning_point_replay_orchestrator_config.json"
MANUAL_SUBDIR = os.path.join("manual_intake", "recife_p1")
PATCH_INTAKE = "recife_p1_patch_geometry_intake.csv"
EVENT_INTAKE = "recife_p1_event_geometry_intake.csv"

INPUTS = {
    "packages": "v2at_event_patch_package_registry.csv",
    "patches": os.path.join(MANUAL_SUBDIR, PATCH_INTAKE),
    "events": os.path.join(MANUAL_SUBDIR, EVENT_INTAKE),
    "v2ax_manifest": "v2ax_recife_manual_intake_manifest.csv",
    "v2ay_contract": "v2ay_minimum_real_geometry_contract.csv",
    "v2ay_gates": "v2ay_turning_point_readiness_gate.csv",
    "overlays": "v2au_patch_event_overlay_registry.csv",
}

OUTPUTS = {
    "snapshot": "v2az_manual_intake_validation_snapshot.csv",
    "aw_patch": "v2az_feed_v2aw_patch_sources.csv",
    "aw_event": "v2az_feed_v2aw_event_sources.csv",
    "av_patch": "v2az_feed_v2av_patch_geometry_sources.csv",
    "au_geom": "v2az_feed_v2au_geometry_sources.csv",
    "promotion": "v2az_input_promotion_audit.csv",
    "plan": "v2az_replay_execution_plan.csv",
    "log": "v2az_replay_execution_log.csv",
    "progress": "v2az_turning_point_progress_registry.csv",
    "certificate": "v2az_replay_readiness_certificate.csv",
}

SNAPSHOT_COLUMNS = [
    "snapshot_id", "target_type", "target_id", "patch_id", "event_id", "package_id",
    "region", "city", "source_file", "source_type", "geometry_present", "geometry_format",
    "geometry_format_valid", "geometry_path_exists", "crs", "crs_present", "crs_accepted",
    "provenance_present", "license_present", "review_status", "review_status_valid", "is_point",
    "is_polygon_or_bbox", "can_feed_v2aw", "can_feed_v2av", "can_feed_v2au",
    "blocking_reason", "recommended_fix", "notes",
]
PROMOTION_COLUMNS = [
    "promotion_audit_id", "source_row_id", "target_stage", "target_file", "target_type",
    "target_id", "promotion_attempted", "promotion_allowed", "promotion_status",
    "required_conditions", "observed_conditions", "blocking_reason", "notes",
]
PLAN_COLUMNS = [
    "replay_step_id", "step_order", "stage", "command", "mode", "will_run", "precondition",
    "precondition_met", "expected_output", "success_condition", "blocking_reason", "notes",
]
LOG_COLUMNS = [
    "execution_id", "step_order", "stage", "command", "mode", "executed", "exit_code",
    "status", "stdout_summary", "stderr_summary", "outputs_checked", "blocking_reason", "notes",
]
PROGRESS_COLUMNS = [
    "tp_id", "turning_point_level", "required_condition", "observed_value", "gate_passed",
    "unlocked", "remaining_blocker", "next_action", "notes",
]
CERT_COLUMNS = [
    "certificate_id", "mode", "manual_patch_rows", "manual_event_rows", "valid_patch_boundaries",
    "valid_event_polygons", "valid_patch_event_pairs", "feed_v2aw_rows", "feed_v2av_rows",
    "feed_v2au_rows", "can_replay_v2aw", "can_replay_v2av", "can_replay_v2au",
    "can_attempt_tp4", "turning_point_level", "turning_point_ready", "blocking_reason", "notes",
]

HANDOFF_PATCH_COLUMNS = v2ax.PATCH_COLUMNS + ["selected_package_id", "selection_reason", "evidence_score"]
HANDOFF_EVENT_COLUMNS = v2ax.EVENT_COLUMNS + ["selected_package_id", "selection_reason", "evidence_score"]
PAIR_COLUMNS = [
    "pairing_id", "package_id", "patch_id", "event_id", "region", "city", "hazard_type",
    "evidence_score", "allowed_use", "selection_reason", "patch_handoff_file", "event_handoff_file",
    "pair_ready", "blocking_reason", "notes",
]

REPORT_REL = os.path.join("execution_reports", "v2az_turning_point_replay_orchestrator_report.md")
SUMMARY_REL = os.path.join("execution_reports", "v2az_turning_point_replay_orchestrator_summary.json")
LOG_REL = os.path.join("logs_summary", "v2az_turning_point_replay_orchestrator.txt")

DEFAULT_CONFIG = {
    "offline_mode": True, "default_mode": "dry_run", "allow_subprocess_replay": True,
    "run_v2aw_on_replay": True, "run_v2av_on_replay": True, "run_v2au_on_replay": True,
    "priority_region": "Recife", "priority_event_id": "REC_2022_05_24_30",
    "minimum_tp1_patch_boundaries": 1, "minimum_tp2_event_polygons": 1,
    "minimum_tp4_confirmed_overlays": 1,
    "accepted_crs": ["EPSG:4326", "EPSG:3857", "EPSG:31982", "EPSG:31983"],
    "accepted_patch_geometry_formats": ["bbox", "wkt", "geojson_inline", "geojson_file"],
    "accepted_event_geometry_formats": ["wkt", "geojson_inline", "geojson_file"],
    "allow_point_as_patch_boundary": False, "allow_point_as_event_polygon": False,
    "allow_auto_geometry_generation": False, "can_create_operational_labels": False,
    "can_train_model": False,
}


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_dirs():
    return (
        os.environ.get("DATASET_DIR") or project_path("datasets"),
        os.environ.get("OUTPUT_DIR") or project_path("outputs_public"),
        os.environ.get("CONFIG_DIR") or project_path("configs"),
        os.environ.get("DOCS_DIR") or project_path("docs"),
    )


def clean(value):
    return "" if value is None else str(value).strip()


def b(value):
    return "true" if bool(value) else "false"


def stable_id(prefix, *parts):
    raw = "|".join(clean(part) for part in parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def source_hash(row):
    fields = ("source_type", "geometry_value", "geometry_path", "crs", "provenance_type",
              "provenance_note", "license_status", "review_status")
    return hashlib.sha256("|".join(clean(row.get(field)) for field in fields).encode("utf-8")).hexdigest()[:16]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{col: row.get(col, "") for col in columns} for row in rows])


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def load_config(config_dir):
    config = dict(DEFAULT_CONFIG)
    path = os.path.join(config_dir, CONFIG_NAME)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            config.update(json.load(handle))
    config["allowed_patch_geometry_formats"] = config["accepted_patch_geometry_formats"]
    config["allowed_event_geometry_formats"] = config["accepted_event_geometry_formats"]
    return config


def load_inputs(dataset_dir):
    return {key: load_csv(os.path.join(dataset_dir, rel)) for key, rel in INPUTS.items()}


def package_maps(packages):
    by_patch, by_event = {}, {}
    for package in packages:
        by_patch.setdefault(clean(package.get("patch_id")), package)
        by_event.setdefault(clean(package.get("event_id")), package)
    return by_patch, by_event


def path_exists(row, dataset_dir):
    path = clean(row.get("geometry_path"))
    if not path:
        return False
    candidates = [path, os.path.join(dataset_dir, path), os.path.join(PROJECT_ROOT, path)]
    return any(os.path.exists(candidate) for candidate in candidates)


def validate_rows(inputs, config, dataset_dir):
    by_patch, by_event = package_maps(inputs["packages"])
    rows = []
    for target_type, source_rows in (("patch", inputs["patches"]), ("event", inputs["events"])):
        for source in source_rows:
            target_id = clean(source.get("patch_id")) if target_type == "patch" else clean(source.get("event_id"))
            package = by_patch.get(target_id, {}) if target_type == "patch" else by_event.get(target_id, {})
            result = v2ax.validate_intake(target_type, source, clean(package.get("package_id")), config, dataset_dir)
            rows.append({
                "snapshot_id": stable_id("V2AZ_SNAP_", target_type, target_id),
                "target_type": target_type, "target_id": target_id,
                "patch_id": target_id if target_type == "patch" else "",
                "event_id": target_id if target_type == "event" else "",
                "package_id": clean(package.get("package_id")), "region": clean(source.get("region")),
                "city": clean(source.get("city")), "source_file": PATCH_INTAKE if target_type == "patch" else EVENT_INTAKE,
                "source_type": clean(source.get("source_type")), "geometry_present": result["geometry_present"],
                "geometry_format": result["geometry_format"],
                "geometry_format_valid": result["geometry_format_valid"],
                "geometry_path_exists": b(path_exists(source, dataset_dir)), "crs": clean(source.get("crs")),
                "crs_present": result["crs_present"], "crs_accepted": result["crs_accepted"],
                "provenance_present": result["provenance_present"], "license_present": result["license_present"],
                "review_status": clean(source.get("review_status")),
                "review_status_valid": result["review_status_valid"], "is_point": result["is_point"],
                "is_polygon_or_bbox": result["is_polygon_or_bbox"], "can_feed_v2aw": result["can_feed_v2aw"],
                "can_feed_v2av": result["can_feed_v2av"], "can_feed_v2au": result["can_feed_v2au"],
                "blocking_reason": result["blocking_reason"], "recommended_fix": result["recommended_fix"],
                "notes": f"source_hash={source_hash(source)}; fail-closed manual intake snapshot.",
            })
    return sorted(rows, key=lambda row: (row["target_type"], row["target_id"]))


def build_feeds(inputs, snapshot):
    valid = {(row["target_type"], row["target_id"]): row for row in snapshot}
    aw_patch, aw_event, av_patch, au_geom = [], [], [], []
    for row in inputs["patches"]:
        patch_id = clean(row.get("patch_id"))
        if valid[("patch", patch_id)]["can_feed_v2av"] != "true":
            continue
        digest = source_hash(row)
        aw_patch.append({
            "geometry_source_id": clean(row.get("intake_id")), "linked_patch_id": patch_id,
            "region": row.get("region"), "city": row.get("city"), "priority_rank": row.get("priority_rank"),
            "source_type": row.get("source_type"), "geometry_value": row.get("geometry_value"),
            "geometry_path": row.get("geometry_path"), "crs": row.get("crs"),
            "provenance_type": row.get("provenance_type"), "provenance_note": row.get("provenance_note"),
            "digitized_by": row.get("digitized_by"), "digitized_at": row.get("digitized_at"),
            "source_document": row.get("source_document"), "source_document_page": row.get("source_document_page"),
            "source_confidence": row.get("source_confidence"), "license_status": row.get("license_status"),
            "review_status": row.get("review_status"), "notes": f"v2az source_hash={digest}; validated real source.",
        })
        av_patch.append({
            "patch_id": patch_id, "region": row.get("region"), "city": row.get("city"),
            "source_file": f"datasets/manual_intake/recife_p1/{PATCH_INTAKE}",
            "source_field": "geometry_value|geometry_path", "source_type": row.get("source_type"),
            "geometry_value": row.get("geometry_value"), "geometry_path": row.get("geometry_path"),
            "crs": row.get("crs"), "center_lat": "", "center_lon": "", "size_meters": "",
            "source_confidence": clean(row.get("source_confidence")) or f"VALIDATED_{digest}",
        })
        au_geom.append({
            "geometry_role": "patch_boundary", "linked_event_id": "", "linked_patch_id": patch_id,
            "source_id": clean(row.get("intake_id")), "source_name": clean(row.get("source_document")),
            "geometry_type": "polygon", "geometry_format": row.get("source_type"),
            "geometry_value": row.get("geometry_value"), "geometry_path": row.get("geometry_path"),
            "crs": row.get("crs"), "latitude": "", "longitude": "",
        })
    for row in inputs["events"]:
        event_id = clean(row.get("event_id"))
        if valid[("event", event_id)]["can_feed_v2au"] != "true":
            continue
        digest = source_hash(row)
        aw_event.append({
            "event_geometry_source_id": clean(row.get("event_intake_id")), "linked_event_id": event_id,
            "region": row.get("region"), "city": row.get("city"), "hazard_type": row.get("hazard_type"),
            "source_type": row.get("source_type"), "geometry_value": row.get("geometry_value"),
            "geometry_path": row.get("geometry_path"), "crs": row.get("crs"),
            "event_geometry_role": row.get("event_geometry_role"), "source_id": row.get("source_id"),
            "source_name": row.get("source_name"), "provenance_type": row.get("provenance_type"),
            "provenance_note": row.get("provenance_note"), "digitized_by": row.get("digitized_by"),
            "digitized_at": row.get("digitized_at"), "source_document": row.get("source_document"),
            "source_document_page": row.get("source_document_page"), "source_confidence": row.get("source_confidence"),
            "license_status": row.get("license_status"), "review_status": row.get("review_status"),
            "notes": f"v2az source_hash={digest}; validated real source.",
        })
        au_geom.append({
            "geometry_role": "event_observed_geometry", "linked_event_id": event_id, "linked_patch_id": "",
            "source_id": clean(row.get("source_id")) or clean(row.get("event_intake_id")),
            "source_name": clean(row.get("source_name")) or clean(row.get("source_document")),
            "geometry_type": "polygon", "geometry_format": row.get("source_type"),
            "geometry_value": row.get("geometry_value"), "geometry_path": row.get("geometry_path"),
            "crs": row.get("crs"), "latitude": "", "longitude": "",
        })
    return aw_patch, aw_event, av_patch, au_geom


def valid_pairs(inputs, snapshot):
    patch_valid = {row["target_id"] for row in snapshot if row["target_type"] == "patch"
                   and row["can_feed_v2av"] == "true"}
    event_valid = {row["target_id"] for row in snapshot if row["target_type"] == "event"
                   and row["can_feed_v2au"] == "true"}
    return [package for package in inputs["packages"]
            if clean(package.get("patch_id")) in patch_valid and clean(package.get("event_id")) in event_valid]


def overlay_count(inputs):
    return sum(clean(row.get("has_patch_overlay")).lower() == "true" for row in inputs["overlays"])


def turning_level(patch_count, event_count, pair_count, overlays):
    if overlays:
        return "TP4_ONE_OVERLAY_CONFIRMED_REQUIRES_HUMAN_REVIEW"
    if pair_count:
        return "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY"
    if event_count:
        return "TP2_ONE_EVENT_POLYGON_VALIDATED"
    if patch_count:
        return "TP1_ONE_PATCH_BOUNDARY_VALIDATED"
    return "TP0_DOCUMENTED_ABSENCE"


def build_progress(patch_count, event_count, pair_count, overlays):
    checks = [
        ("TP0_DOCUMENTED_ABSENCE", True, "Current absence/readiness is documented", ""),
        ("TP1_ONE_PATCH_BOUNDARY_VALIDATED", patch_count >= 1, f"valid_patch_boundaries={patch_count}",
         "NO_VALID_PATCH_BOUNDARY"),
        ("TP2_ONE_EVENT_POLYGON_VALIDATED", event_count >= 1, f"valid_event_polygons={event_count}",
         "NO_VALID_EVENT_POLYGON"),
        ("TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY", pair_count >= 1, f"valid_patch_event_pairs={pair_count}",
         "NO_VALID_PATCH_EVENT_PAIR"),
        ("TP4_ONE_OVERLAY_CONFIRMED_REQUIRES_HUMAN_REVIEW", overlays >= 1, f"confirmed_overlays={overlays}",
         "NO_CONFIRMED_OVERLAY"),
    ]
    return [{
        "tp_id": stable_id("V2AZ_TP_", level), "turning_point_level": level,
        "required_condition": condition, "observed_value": observed, "gate_passed": b(passed),
        "unlocked": b(passed), "remaining_blocker": "" if passed else blocker,
        "next_action": "Preserve documented state" if level.startswith("TP0") else
        "Acquire/validate real geometry and run controlled replay",
        "notes": "No turning point creates an operational label, final ground truth, or model.",
    } for level, passed, observed, blocker in checks for condition in [level.replace("_", " ").lower()]]


def build_promotion(snapshot):
    rows = []
    targets = {
        "patch": [("v2aw", OUTPUTS["aw_patch"]), ("v2av", OUTPUTS["av_patch"]), ("v2au", OUTPUTS["au_geom"])],
        "event": [("v2aw", OUTPUTS["aw_event"]), ("v2au", OUTPUTS["au_geom"])],
    }
    for source in snapshot:
        for stage, target_file in targets[source["target_type"]]:
            allowed = source["can_feed_v2av"] == "true" if stage == "v2av" else source["can_feed_v2au"] == "true" \
                if stage == "v2au" else source["can_feed_v2aw"] == "true"
            rows.append({
                "promotion_audit_id": stable_id("V2AZ_PROM_", source["snapshot_id"], stage),
                "source_row_id": source["snapshot_id"], "target_stage": stage, "target_file": target_file,
                "target_type": source["target_type"], "target_id": source["target_id"],
                "promotion_attempted": b(allowed), "promotion_allowed": b(allowed),
                "promotion_status": "PROMOTED_TO_DERIVED_FEED" if allowed else "BLOCKED_NOT_PROMOTED",
                "required_conditions": "valid polygon/bbox|accepted CRS|provenance|license|valid review status",
                "observed_conditions": f"geometry={source['geometry_present']};crs={source['crs_accepted']};"
                f"provenance={source['provenance_present']};license={source['license_present']};"
                f"review={source['review_status_valid']}",
                "blocking_reason": "" if allowed else source["blocking_reason"],
                "notes": "Only derived feed is written; canonical prior-stage input is not overwritten.",
            })
    return rows


def choose_candidate(packages):
    candidates = [row for row in packages if clean(row.get("region")) == "Recife"
                  and clean(row.get("event_id")) == "REC_2022_05_24_30"
                  and clean(row.get("allowed_use")) == "candidate_reference"]
    return sorted(candidates, key=lambda row: (-float(clean(row.get("evidence_score")) or 0),
                                               clean(row.get("patch_id")),
                                               clean(row.get("package_id"))))[0]


def build_handoff(inputs):
    candidate = choose_candidate(inputs["packages"])
    patch = next((row for row in inputs["patches"]
                  if clean(row.get("patch_id")) == clean(candidate.get("patch_id"))), inputs["patches"][0])
    event = next((row for row in inputs["events"]
                  if clean(row.get("event_id")) == clean(candidate.get("event_id"))), inputs["events"][0])
    paired = (clean(patch.get("patch_id")) == clean(candidate.get("patch_id"))
              and clean(event.get("event_id")) == clean(candidate.get("event_id")))
    reason = "candidate_reference; Recife real event; highest evidence_score; stable patch_id tie-break"
    patch_row = dict(patch)
    patch_row.update({"selected_package_id": candidate["package_id"], "selection_reason": reason,
                      "evidence_score": candidate.get("evidence_score", "")})
    event_row = dict(event)
    event_row.update({"selected_package_id": candidate["package_id"], "selection_reason": reason,
                      "evidence_score": candidate.get("evidence_score", "")})
    pairing = [{
        "pairing_id": stable_id("V2AZ_PAIR_", candidate["package_id"]), "package_id": candidate["package_id"],
        "patch_id": candidate["patch_id"], "event_id": candidate["event_id"], "region": candidate["region"],
        "city": candidate["city"], "hazard_type": candidate["hazard_type"],
        "evidence_score": candidate.get("evidence_score", ""), "allowed_use": candidate["allowed_use"],
        "selection_reason": reason,
        "patch_handoff_file": f"datasets/manual_intake/recife_p1/minimal_turning_point_candidate_patch.csv",
        "event_handoff_file": f"datasets/manual_intake/recife_p1/minimal_turning_point_candidate_event.csv",
        "pair_ready": "false", "blocking_reason": "REAL_PATCH_AND_EVENT_GEOMETRY_REQUIRED"
        if paired else "MANUAL_INTAKE_NOT_PAIRED_TO_SELECTED_PACKAGE",
        "notes": "Handoff contains no invented geometry.",
    }]
    return [patch_row], [event_row], pairing


def build_plan(mode, patch_count, event_count, pair_count):
    replay = mode == "replay"
    any_real_source = patch_count + event_count > 0
    specs = [
        ("1", "v2ax_validate_manual_intake", "python scripts/run_v2ax_recife_geometry_intake_pack.py",
         any_real_source, "at least one valid real source exists", "v2ax validation/feeds"),
        ("2", "v2aw_geometry_source_intake", "python scripts/run_v2aw_geometry_source_intake.py",
         patch_count + event_count > 0, "at least one valid real source", "v2aw validation/readiness"),
        ("3", "v2av_patch_boundary_geometry_builder", "python scripts/run_v2av_patch_boundary_geometry_builder.py",
         patch_count > 0, "at least one valid patch boundary", "v2av boundary registry"),
        ("4", "v2au_patch_event_overlay_geometry", "python scripts/run_v2au_patch_event_overlay_geometry.py",
         pair_count > 0, "at least one valid linked patch-event pair", "v2au overlay registry"),
        ("5", "v2ay_turning_point_recheck",
         "python scripts/run_v2ay_event_scope_reconciliation_turning_point.py",
         pair_count > 0, "controlled replay completed", "v2ay turning point recheck"),
    ]
    return [{
        "replay_step_id": stable_id("V2AZ_STEP_", order), "step_order": order, "stage": stage,
        "command": command, "mode": mode, "will_run": b(replay and precondition),
        "precondition": precondition_text, "precondition_met": b(precondition),
        "expected_output": expected, "success_condition": "exit 0 and expected derived output checked",
        "blocking_reason": "" if replay and precondition else
        "DRY_RUN_NO_SUBPROCESS" if not replay else "PRECONDITION_NOT_MET",
        "notes": "Replay runs only in isolated workspace; canonical previous outputs are not overwritten.",
    } for order, stage, command, precondition, precondition_text, expected in specs]


def replay_workspace(dataset_dir, output_dir, feeds):
    root = os.path.join(output_dir, "v2az_replay_workspace")
    replay_data, replay_output = os.path.join(root, "datasets"), os.path.join(root, "outputs_public")
    os.makedirs(replay_data, exist_ok=True)
    os.makedirs(replay_output, exist_ok=True)
    required = [
        "v2at_event_patch_package_registry.csv", "v2av_patch_boundary_recovery_queue.csv",
        "ground_reference_event_registry.csv",
    ]
    for name in required:
        source = os.path.join(dataset_dir, name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(replay_data, name))
    write_csv(os.path.join(replay_data, "v2aw_patch_geometry_sources.csv"), v2ax.V2AW_PATCH_COLUMNS, feeds[0])
    write_csv(os.path.join(replay_data, "v2aw_event_geometry_sources.csv"), v2ax.V2AW_EVENT_COLUMNS, feeds[1])
    write_csv(os.path.join(replay_data, "v2av_patch_geometry_sources.csv"), v2ax.V2AV_PATCH_COLUMNS, feeds[2])
    write_csv(os.path.join(replay_data, "v2au_geometry_sources.csv"), v2ax.V2AU_GEOM_COLUMNS, feeds[3])
    return replay_data, replay_output


def execute_plan(plan, dataset_dir, output_dir, config_dir, feeds):
    logs = []
    replay_data = replay_output = None
    if any(row["will_run"] == "true" for row in plan):
        replay_data, replay_output = replay_workspace(dataset_dir, output_dir, feeds)
    for row in plan:
        executed = row["will_run"] == "true"
        code, stdout, stderr = "", "", ""
        status = "NOT_EXECUTED_DRY_RUN" if row["mode"] == "dry_run" else "BLOCKED_PRECONDITION"
        if executed:
            env = dict(os.environ)
            env.update({"DATASET_DIR": replay_data, "OUTPUT_DIR": replay_output, "CONFIG_DIR": config_dir})
            proc = subprocess.run(row["command"].split(), cwd=PROJECT_ROOT, env=env, capture_output=True,
                                  text=True, timeout=120)
            code, stdout, stderr = str(proc.returncode), proc.stdout[-500:], proc.stderr[-500:]
            status = "EXECUTED_OK" if proc.returncode == 0 else "EXECUTED_FAILED"
        logs.append({
            "execution_id": stable_id("V2AZ_EXEC_", row["step_order"], row["mode"]),
            "step_order": row["step_order"], "stage": row["stage"], "command": row["command"],
            "mode": row["mode"], "executed": b(executed), "exit_code": code, "status": status,
            "stdout_summary": stdout, "stderr_summary": stderr,
            "outputs_checked": replay_output or "", "blocking_reason": row["blocking_reason"],
            "notes": "Canonical prior-stage outputs remain untouched.",
        })
    return logs


def build_certificate(mode, inputs, feeds, snapshot, pairs):
    patch_count = sum(row["target_type"] == "patch" and row["can_feed_v2av"] == "true" for row in snapshot)
    event_count = sum(row["target_type"] == "event" and row["can_feed_v2au"] == "true" for row in snapshot)
    overlays = overlay_count(inputs)
    level = turning_level(patch_count, event_count, len(pairs), overlays)
    return [{
        "certificate_id": stable_id("V2AZ_CERT_", mode, level), "mode": mode,
        "manual_patch_rows": str(len(inputs["patches"])), "manual_event_rows": str(len(inputs["events"])),
        "valid_patch_boundaries": str(patch_count), "valid_event_polygons": str(event_count),
        "valid_patch_event_pairs": str(len(pairs)), "feed_v2aw_rows": str(len(feeds[0]) + len(feeds[1])),
        "feed_v2av_rows": str(len(feeds[2])), "feed_v2au_rows": str(len(feeds[3])),
        "can_replay_v2aw": b(patch_count + event_count > 0), "can_replay_v2av": b(patch_count > 0),
        "can_replay_v2au": b(len(pairs) > 0), "can_attempt_tp4": b(len(pairs) > 0),
        "turning_point_level": level, "turning_point_ready": b(level != "TP0_DOCUMENTED_ABSENCE"),
        "blocking_reason": "" if level != "TP0_DOCUMENTED_ABSENCE" else "WAITING_FOR_REAL_GEOMETRY",
        "notes": "Replay readiness is fail-closed and creates no label/model/ground truth.",
    }]


def schema(name, columns):
    props = {col: {"type": "string"} for col in columns}
    for col in columns:
        if col.startswith("can_") or col in ("geometry_present", "geometry_format_valid", "geometry_path_exists",
                                             "crs_present", "crs_accepted", "provenance_present", "license_present",
                                             "review_status_valid", "is_point", "is_polygon_or_bbox",
                                             "promotion_attempted", "promotion_allowed", "will_run",
                                             "precondition_met", "executed", "gate_passed", "unlocked",
                                             "turning_point_ready"):
            props[col]["enum"] = ["true", "false"]
    return {"$schema": "http://json-schema.org/draft-07/schema#", "title": name,
            "description": "v2az fail-closed replay adapter; no invented geometry, label, model, final ground truth or automatic C4.",
            "type": "object", "required": columns, "additionalProperties": False, "properties": props}


def write_schemas(dataset_dir):
    specs = {
        OUTPUTS["snapshot"]: SNAPSHOT_COLUMNS, OUTPUTS["promotion"]: PROMOTION_COLUMNS,
        OUTPUTS["plan"]: PLAN_COLUMNS, OUTPUTS["log"]: LOG_COLUMNS, OUTPUTS["progress"]: PROGRESS_COLUMNS,
        OUTPUTS["certificate"]: CERT_COLUMNS, OUTPUTS["aw_patch"]: v2ax.V2AW_PATCH_COLUMNS,
        OUTPUTS["aw_event"]: v2ax.V2AW_EVENT_COLUMNS, OUTPUTS["av_patch"]: v2ax.V2AV_PATCH_COLUMNS,
        OUTPUTS["au_geom"]: v2ax.V2AU_GEOM_COLUMNS,
    }
    for filename, columns in specs.items():
        name = os.path.splitext(filename)[0] + ".schema.json"
        write_text(os.path.join(dataset_dir, "schemas", name), json.dumps(schema(name, columns), indent=2) + "\n")


def write_docs(docs_dir, candidate):
    patch_id, event_id = candidate["patch_id"], candidate["event_id"]
    docs = {
        "v2az_turning_point_replay_orchestrator.md": """# v2az - Turning Point Replay Orchestrator

`dry_run` valida intake, gera feeds e plano, mas nao executa subprocessos. `replay` so executa
quando ha precondicoes reais e usa workspace isolado. Feeds contem somente geometrias validas.
Ausencia vira blocker. A cadeia controlada e v2ax -> v2aw -> v2av -> v2au -> v2ay. Nada cria label.
""",
        "v2az_minimal_real_geometry_handoff.md": f"""# v2az - Handoff minimo de geometria real

Preencha primeiro o patch `{patch_id}` e o evento `{event_id}` selecionados de forma auditavel.
Campos minimos: `source_type`, `geometry_value` ou `geometry_path`, `crs`, proveniencia, licenca e
review_status valido. Patch aceita bbox/WKT/GeoJSON; evento aceita WKT/GeoJSON polygon. Rode dry_run,
leia blockers e corrija antes de replay. Nao use ponto/centroide.
""",
        "v2az_how_to_reach_tp4.md": """# v2az - Como chegar ao TP4

1. Preencha boundary real do patch.
2. Preencha poligono observado real do evento.
3. Rode `python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run`.
4. Se feeds validos aparecerem, rode com `--mode replay`.
5. Revise o overlay v2au.
6. Aceite no maximo `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`.
7. Nunca treine modelo nesta etapa.
""",
    }
    for name, text in docs.items():
        write_text(os.path.join(docs_dir, name), text)


def build_summary(mode, inputs, snapshot, feeds, pairs):
    patch_count = sum(row["target_type"] == "patch" and row["can_feed_v2av"] == "true" for row in snapshot)
    event_count = sum(row["target_type"] == "event" and row["can_feed_v2au"] == "true" for row in snapshot)
    overlays = overlay_count(inputs)
    level = turning_level(patch_count, event_count, len(pairs), overlays)
    return {
        "stage": STAGE, "status": "OK_WITH_EXPECTED_BLOCKERS", "mode": mode,
        "manual_patch_rows": len(inputs["patches"]), "manual_event_rows": len(inputs["events"]),
        "valid_patch_boundaries": patch_count, "valid_event_polygons": event_count,
        "valid_patch_event_pairs": len(pairs), "feed_v2aw_rows": len(feeds[0]) + len(feeds[1]),
        "feed_v2av_rows": len(feeds[2]), "feed_v2au_rows": len(feeds[3]),
        "turning_point_level": level, "turning_point_ready": level != "TP0_DOCUMENTED_ABSENCE",
        "can_attempt_replay": len(pairs) > 0, "can_attempt_tp4": len(pairs) > 0,
        "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "TURNING_POINT_REPLAY_READY_WAITING_FOR_REAL_GEOMETRY_NOT_FOR_TRAINING",
    }


def report(summary, candidate):
    return f"""# v2az - Turning Point Replay Orchestrator

Mode: **{summary['mode']}**. Manual rows: **{summary['manual_patch_rows']} patch + {summary['manual_event_rows']} event**.
Valid geometries/pairs: **{summary['valid_patch_boundaries']} / {summary['valid_event_polygons']} / {summary['valid_patch_event_pairs']}**.
Feeds v2aw/v2av/v2au: **{summary['feed_v2aw_rows']} / {summary['feed_v2av_rows']} / {summary['feed_v2au_rows']}**.
Turning point: **{summary['turning_point_level']}**.

Minimal handoff: package `{candidate['package_id']}`, patch `{candidate['patch_id']}`, event `{candidate['event_id']}`.
No subprocess runs in dry_run. Replay remains isolated and conditional.
Nenhum modelo, label, treino supervisionado, ground truth final, geometria inventada ou C4 automatico foi criado.
"""


def log_text(summary):
    return (
        f"[v2az] mode={summary['mode']} manual_patch={summary['manual_patch_rows']} manual_event={summary['manual_event_rows']}\n"
        f"[v2az] valid_patch={summary['valid_patch_boundaries']} valid_event={summary['valid_event_polygons']} pairs={summary['valid_patch_event_pairs']}\n"
        f"[v2az] feeds_v2aw={summary['feed_v2aw_rows']} feeds_v2av={summary['feed_v2av_rows']} feeds_v2au={summary['feed_v2au_rows']}\n"
        f"[v2az] turning_point={summary['turning_point_level']} replay={summary['can_attempt_replay']} tp4={summary['can_attempt_tp4']}\n"
        "[v2az] can_train_model=False can_create_operational_labels=False\n"
        f"[v2az] status={summary['status']}\n"
    )


def run(mode=None, dataset_dir=None, output_dir=None, config_dir=None, docs_dir=None):
    env_dataset, env_output, env_config, env_docs = resolve_dirs()
    dataset_dir, output_dir = dataset_dir or env_dataset, output_dir or env_output
    config_dir, docs_dir = config_dir or env_config, docs_dir or env_docs
    config = load_config(config_dir)
    mode = mode or config["default_mode"]
    if mode not in ("dry_run", "replay"):
        return 2, None
    inputs = load_inputs(dataset_dir)
    snapshot = validate_rows(inputs, config, dataset_dir)
    feeds = build_feeds(inputs, snapshot)
    pairs = valid_pairs(inputs, snapshot)
    promotion = build_promotion(snapshot)
    patch_count = len(feeds[2])
    event_count = len(feeds[1])
    plan = build_plan(mode, patch_count, event_count, len(pairs))
    execution_log = execute_plan(plan, dataset_dir, output_dir, config_dir, feeds)
    progress = build_progress(patch_count, event_count, len(pairs), overlay_count(inputs))
    certificate = build_certificate(mode, inputs, feeds, snapshot, pairs)
    handoff_patch, handoff_event, pairing = build_handoff(inputs)
    candidate = pairing[0]

    write_csv(os.path.join(dataset_dir, OUTPUTS["snapshot"]), SNAPSHOT_COLUMNS, snapshot)
    write_csv(os.path.join(dataset_dir, OUTPUTS["aw_patch"]), v2ax.V2AW_PATCH_COLUMNS, feeds[0])
    write_csv(os.path.join(dataset_dir, OUTPUTS["aw_event"]), v2ax.V2AW_EVENT_COLUMNS, feeds[1])
    write_csv(os.path.join(dataset_dir, OUTPUTS["av_patch"]), v2ax.V2AV_PATCH_COLUMNS, feeds[2])
    write_csv(os.path.join(dataset_dir, OUTPUTS["au_geom"]), v2ax.V2AU_GEOM_COLUMNS, feeds[3])
    write_csv(os.path.join(dataset_dir, OUTPUTS["promotion"]), PROMOTION_COLUMNS, promotion)
    write_csv(os.path.join(dataset_dir, OUTPUTS["plan"]), PLAN_COLUMNS, plan)
    write_csv(os.path.join(dataset_dir, OUTPUTS["log"]), LOG_COLUMNS, execution_log)
    write_csv(os.path.join(dataset_dir, OUTPUTS["progress"]), PROGRESS_COLUMNS, progress)
    write_csv(os.path.join(dataset_dir, OUTPUTS["certificate"]), CERT_COLUMNS, certificate)
    manual_dir = os.path.join(dataset_dir, MANUAL_SUBDIR)
    write_csv(os.path.join(manual_dir, "minimal_turning_point_candidate_patch.csv"),
              HANDOFF_PATCH_COLUMNS, handoff_patch)
    write_csv(os.path.join(manual_dir, "minimal_turning_point_candidate_event.csv"),
              HANDOFF_EVENT_COLUMNS, handoff_event)
    write_csv(os.path.join(manual_dir, "minimal_turning_point_pairing.csv"), PAIR_COLUMNS, pairing)
    write_schemas(dataset_dir)
    write_docs(docs_dir, candidate)
    summary = build_summary(mode, inputs, snapshot, feeds, pairs)
    write_text(os.path.join(output_dir, SUMMARY_REL), json.dumps(summary, indent=2) + "\n")
    write_text(os.path.join(output_dir, REPORT_REL), report(summary, candidate))
    write_text(os.path.join(output_dir, LOG_REL), log_text(summary))
    sys.stdout.write(log_text(summary))
    return 0, summary


if __name__ == "__main__":
    raise SystemExit(run()[0])

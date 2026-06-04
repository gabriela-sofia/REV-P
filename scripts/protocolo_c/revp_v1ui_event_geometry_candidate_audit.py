#!/usr/bin/env python3
"""
v1ui — Event Geometry Candidate Audit

Applies 14-gate evaluation to geometry candidates.
G12-G14 always FAIL — max status is READY_FOR_SUPERVISOR_REVIEW.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1ui"

CANDIDATE_COLUMNS = [
    "candidate_audit_id", "event_id", "extraction_id", "source_id",
    "G01_official_public_source", "G02_artifact_traceable",
    "G03_license_public_access", "G04_event_date_available",
    "G05_event_date_compatible", "G06_hazard_type_available",
    "G07_phenomenon_separated", "G08_locality_or_geometry",
    "G09_geometry_available", "G10_crs_available",
    "G11_geometry_quality_sufficient",
    "G12_supervisor_review_pending", "G13_patch_overlay_not_executed",
    "G14_label_forbidden",
    "gates_passed", "gates_failed", "max_status",
    "can_create_ground_reference", "can_create_training_label",
    "notes",
]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_gates(ext, event):
    gates = {}
    is_candidate = ext.get("can_be_observed_geometry_candidate") == "true"
    has_geom = ext.get("has_geometry") == "true"
    has_date = ext.get("has_date_field") == "true"
    has_hazard = ext.get("has_hazard_field") == "true"
    has_locality = ext.get("has_locality_field") == "true"
    has_coords = ext.get("has_coordinate_fields") == "true"
    has_crs = bool(ext.get("crs"))
    is_observed = ext.get("is_observed_occurrence") == "true"
    hazard_scope = event.get("hazard_scope", "")

    gates["G01"] = "PASS" if is_candidate else "FAIL"
    gates["G02"] = "PASS" if ext.get("artifact_id") else "FAIL"
    gates["G03"] = "PASS"
    gates["G04"] = "PASS" if has_date else "FAIL"
    gates["G05"] = "PASS" if has_date else "FAIL"
    gates["G06"] = "PASS" if has_hazard else "FAIL"
    gates["G07"] = "PASS" if hazard_scope != "mixed" or has_hazard else "FAIL"
    if hazard_scope == "mixed":
        gates["G07"] = "FAIL"
    gates["G08"] = "PASS" if (has_locality or has_geom or has_coords) else "FAIL"
    gates["G09"] = "PASS" if (has_geom or has_coords) else "FAIL"
    gates["G10"] = "PASS" if has_crs else "NEEDS_REVIEW"
    gates["G11"] = "PASS" if is_observed else "NEEDS_REVIEW"
    gates["G12"] = "FAIL"
    gates["G13"] = "FAIL"
    gates["G14"] = "FAIL"

    passed = sum(1 for v in gates.values() if v == "PASS")
    failed = sum(1 for v in gates.values() if v == "FAIL")

    g1_g11_pass = all(gates[f"G{i:02d}"] == "PASS" for i in range(1, 12))
    if g1_g11_pass:
        max_status = "OBSERVED_GEOMETRY_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW"
    elif is_candidate:
        max_status = "CANDIDATE_WITH_BLOCKERS"
    else:
        max_status = "NOT_A_GEOMETRY_CANDIDATE"

    return gates, passed, failed, max_status


def main():
    parser = argparse.ArgumentParser(description="v1ui — Event Geometry Candidate Audit")
    parser.add_argument("--extractions",
                        default="datasets/protocolo_c/v1ui_observed_geometry_extraction_registry.csv")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ui_event_geometry_candidate_registry.csv")
    args = parser.parse_args()

    extractions = load_csv(args.extractions)
    events_by_id = {e["event_id"]: e for e in load_csv(args.events)}

    rows = []
    seq = 0
    for ext in extractions:
        event_id = ext.get("event_id", "")
        event = events_by_id.get(event_id, {})
        gates, passed, failed, max_status = evaluate_gates(ext, event)

        row = {
            "candidate_audit_id": f"GAUD_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id,
            "extraction_id": ext.get("extraction_id", ""),
            "source_id": ext.get("source_id", ""),
        }
        for i in range(1, 15):
            gname = f"G{i:02d}"
            col_names = [
                "official_public_source", "artifact_traceable",
                "license_public_access", "event_date_available",
                "event_date_compatible", "hazard_type_available",
                "phenomenon_separated", "locality_or_geometry",
                "geometry_available", "crs_available",
                "geometry_quality_sufficient", "supervisor_review_pending",
                "patch_overlay_not_executed", "label_forbidden",
            ]
            row[f"{gname}_{col_names[i-1]}"] = gates[gname]

        row["gates_passed"] = str(passed)
        row["gates_failed"] = str(failed)
        row["max_status"] = max_status
        row["can_create_ground_reference"] = "false"
        row["can_create_training_label"] = "false"
        row["notes"] = ""
        rows.append(row)
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(1 for r in rows if "READY_FOR_SUPERVISOR_REVIEW" in r["max_status"])
    print(f"[Event Geometry Candidate Audit v1ui] {len(rows)} evaluated | ready_for_review={ready}")
    print(f"  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

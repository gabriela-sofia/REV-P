#!/usr/bin/env python3
"""
v1ui — Supervisor Review Prequeue

Assembles review queue from public discovery candidates.
Never auto-approves. can_create_ground_reference always false.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1ui"

PREQUEUE_COLUMNS = [
    "prequeue_id", "event_id", "candidate_id", "source_id",
    "review_priority", "review_status", "evidence_summary",
    "blocking_gates", "reviewer_task", "decision_options",
    "can_be_reviewed_now",
    "can_create_ground_reference", "can_create_training_label", "notes",
]

DECISION_OPTIONS = [
    "REJECT_CONTEXT_ONLY", "REQUEST_MORE_INFO",
    "ACCEPT_AS_OBSERVED_GEOMETRY_CANDIDATE",
    "NEEDS_CRS_FIX", "NEEDS_PHENOMENON_SEPARATION",
    "NEEDS_DATE_CONFIRMATION", "NEEDS_LICENSE_REVIEW",
    "DO_NOT_PROMOTE",
]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="v1ui — Supervisor Review Prequeue")
    parser.add_argument("--candidates",
                        default="datasets/protocolo_c/v1ui_event_geometry_candidate_registry.csv")
    parser.add_argument("--extractions",
                        default="datasets/protocolo_c/v1ui_observed_geometry_extraction_registry.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ui_supervisor_review_prequeue.csv")
    args = parser.parse_args()

    candidates = load_csv(args.candidates)
    ext_by_id = {e["extraction_id"]: e for e in load_csv(args.extractions)}

    rows = []
    seq = 0
    for cand in candidates:
        ext_id = cand.get("extraction_id", "")
        ext = ext_by_id.get(ext_id, {})
        max_status = cand.get("max_status", "")
        is_ready = "READY_FOR_SUPERVISOR_REVIEW" in max_status
        is_candidate = max_status in (
            "OBSERVED_GEOMETRY_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW",
            "CANDIDATE_WITH_BLOCKERS",
        )

        blocking = []
        for i in range(1, 15):
            col_names = [
                "official_public_source", "artifact_traceable",
                "license_public_access", "event_date_available",
                "event_date_compatible", "hazard_type_available",
                "phenomenon_separated", "locality_or_geometry",
                "geometry_available", "crs_available",
                "geometry_quality_sufficient", "supervisor_review_pending",
                "patch_overlay_not_executed", "label_forbidden",
            ]
            gate_col = f"G{i:02d}_{col_names[i-1]}"
            if cand.get(gate_col) == "FAIL" and i <= 11:
                blocking.append(f"G{i:02d}")

        evidence = []
        gclass = ext.get("geometry_candidate_class", "")
        if gclass:
            evidence.append(f"class={gclass}")
        if ext.get("has_geometry") == "true":
            evidence.append("has_geometry")
        if ext.get("has_date_field") == "true":
            evidence.append("has_date")
        if ext.get("has_hazard_field") == "true":
            evidence.append("has_hazard")

        if is_ready:
            priority = "1"
            status = "READY_FOR_REVIEW"
            task = "Review geometry candidate and decide disposition"
        elif is_candidate:
            priority = "2"
            status = "BLOCKED_PENDING_GATES"
            task = f"Resolve gates: {', '.join(blocking[:5])}"
        else:
            priority = "9"
            status = "NOT_REVIEWABLE"
            task = "Not a geometry candidate"

        rows.append({
            "prequeue_id": f"PRQ_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": cand.get("event_id", ""),
            "candidate_id": cand.get("candidate_audit_id", ""),
            "source_id": cand.get("source_id", ""),
            "review_priority": priority,
            "review_status": status,
            "evidence_summary": "; ".join(evidence),
            "blocking_gates": "|".join(blocking),
            "reviewer_task": task,
            "decision_options": "|".join(DECISION_OPTIONS),
            "can_be_reviewed_now": str(is_ready).lower(),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "",
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREQUEUE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(1 for r in rows if r["review_status"] == "READY_FOR_REVIEW")
    print(f"[Supervisor Review Prequeue v1ui] {len(rows)} entries | ready={ready}")
    print(f"  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

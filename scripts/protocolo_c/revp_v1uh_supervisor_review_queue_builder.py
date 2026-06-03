#!/usr/bin/env python3
"""
v1uh — Supervisor Review Queue Builder

Assembles a queue of candidates ready (or blocked) for human supervisor
review. Never auto-approves. can_create_ground_reference always false.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1uh"

QUEUE_COLUMNS = [
    "queue_id", "event_id", "candidate_id", "institution",
    "review_priority", "review_status", "reviewer_task",
    "evidence_summary", "blocking_gates",
    "required_inputs", "decision_options",
    "can_be_reviewed_now",
    "can_create_ground_reference", "can_create_training_label", "notes",
]

DECISION_OPTIONS = [
    "REJECT_CONTEXT_ONLY",
    "REQUEST_MORE_INFO",
    "ACCEPT_AS_OBSERVED_GEOMETRY_CANDIDATE",
    "NEEDS_CRS_FIX",
    "NEEDS_PHENOMENON_SEPARATION",
    "NEEDS_DATE_CONFIRMATION",
    "NEEDS_LICENSE_REVIEW",
    "DO_NOT_PROMOTE",
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_queue(candidates: list[dict], crs_audits: dict,
                phenom_gates: dict) -> list[dict]:
    rows = []
    seq = 0

    for cand in candidates:
        cand_id = cand.get("candidate_id", "")
        crs = crs_audits.get(cand_id, {})
        gate = phenom_gates.get(cand_id, {})

        candidate_class = cand.get("candidate_class", "")
        can_be_candidate = cand.get("can_be_ground_reference_candidate") == "true"

        blockers = []
        crs_blocking = crs.get("blocking") == "true"
        if crs_blocking:
            blockers.append(f"crs:{crs.get('required_action', 'unknown')}")

        temporal_blocked = gate.get("temporal_gate_status") == "BLOCKED"
        phenom_blocked = gate.get("phenomenon_gate_status") == "BLOCKED"
        if temporal_blocked:
            blockers.append("temporal_gate_blocked")
        if phenom_blocked:
            blockers.append("phenomenon_gate_blocked")

        if not can_be_candidate:
            blockers.append("not_geometry_candidate")

        can_review = can_be_candidate and not crs_blocking
        if candidate_class in ("DOCUMENT_ONLY", "STATIC_MAP_ONLY",
                               "MODELED_SUSCEPTIBILITY_CONTEXT",
                               "INSUFFICIENT_METADATA"):
            can_review = False

        if can_review and not temporal_blocked and not phenom_blocked:
            review_status = "READY_FOR_REVIEW"
            priority = "1"
        elif can_review:
            review_status = "BLOCKED_PENDING_GATES"
            priority = "2"
        else:
            review_status = "NOT_REVIEWABLE"
            priority = "9"

        evidence = []
        if candidate_class:
            evidence.append(f"class={candidate_class}")
        if crs.get("crs_value"):
            evidence.append(f"crs={crs['crs_value']}")
        if gate.get("event_date_status"):
            evidence.append(f"date={gate['event_date_status']}")

        task = "Review geometry candidate and decide disposition"
        if blockers:
            task = f"Resolve blockers: {'; '.join(blockers[:3])}"

        rows.append({
            "queue_id": f"SRVQ_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": cand.get("event_id", ""),
            "candidate_id": cand_id,
            "institution": cand.get("institution", ""),
            "review_priority": priority,
            "review_status": review_status,
            "reviewer_task": task,
            "evidence_summary": "; ".join(evidence),
            "blocking_gates": "|".join(blockers) if blockers else "",
            "required_inputs": cand.get("required_next_action", ""),
            "decision_options": "|".join(DECISION_OPTIONS),
            "can_be_reviewed_now": str(can_review and review_status == "READY_FOR_REVIEW").lower(),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "",
        })
        seq += 1

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="v1uh — Supervisor Review Queue Builder")
    parser.add_argument("--candidates",
                        default="datasets/protocolo_c/v1uh_observed_geometry_candidate_registry.csv")
    parser.add_argument("--crs-audit",
                        default="datasets/protocolo_c/v1uh_crs_geometry_quality_audit.csv")
    parser.add_argument("--phenom-gates",
                        default="datasets/protocolo_c/v1uh_phenomenon_temporal_gate_audit.csv")
    parser.add_argument("--out",
                        default="datasets/protocolo_c/v1uh_supervisor_review_queue.csv")
    args = parser.parse_args()

    candidates = load_csv(args.candidates)
    crs_by_cand = {r["candidate_id"]: r for r in load_csv(args.crs_audit)}
    phenom_by_cand = {r["candidate_id"]: r for r in load_csv(args.phenom_gates)}

    rows = build_queue(candidates, crs_by_cand, phenom_by_cand)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(1 for r in rows if r["review_status"] == "READY_FOR_REVIEW")
    blocked = sum(1 for r in rows if r["review_status"] == "BLOCKED_PENDING_GATES")
    not_rev = sum(1 for r in rows if r["review_status"] == "NOT_REVIEWABLE")
    print(f"[Supervisor Review Queue v1uh] {len(rows)} entries")
    print(f"  READY: {ready} | BLOCKED: {blocked} | NOT_REVIEWABLE: {not_rev}")
    print(f"  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

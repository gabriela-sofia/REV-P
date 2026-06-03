#!/usr/bin/env python3
"""
v1uh — Phenomenon and Temporal Gate Audit

Audits whether geometry candidates have compatible event dates
and hazard types. PET_2022 mixed events require phenomenon separation.
can_create_ground_reference is always false.
"""

import argparse
import csv
import os
from datetime import datetime

PROTOCOL_VERSION = "v1uh"

GATE_COLUMNS = [
    "gate_id", "candidate_id", "event_id", "asset_id",
    "event_date_status", "event_date_match",
    "hazard_status", "phenomenon_status", "phenomenon_separated",
    "temporal_gate_status", "phenomenon_gate_status",
    "can_advance_to_supervisor_review",
    "can_create_ground_reference", "can_create_training_label",
    "blocking_reason", "required_action", "notes",
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_date(s: str):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def check_temporal(candidate: dict, event: dict, mappings: list[dict]) -> tuple[str, str]:
    has_date_field = candidate.get("has_event_date_field") == "true"
    if not has_date_field:
        date_mappings = [m for m in mappings
                         if m.get("canonical_field") == "event_date"
                         and m.get("mapping_status") == "MAPPED"]
        if date_mappings:
            has_date_field = True

    if not has_date_field:
        return "MISSING", "No event date field detected"

    start = parse_date(event.get("start_date", ""))
    if not start:
        return "EVENT_DATE_UNKNOWN", "Cannot verify — event date not parsed"

    return "PRESENT_NEEDS_VERIFICATION", "Date field exists but values need human verification"


def check_phenomenon(candidate: dict, event: dict,
                     mappings: list[dict]) -> tuple[str, str, str]:
    hazard_scope = event.get("hazard_scope", "")
    has_hazard = candidate.get("has_hazard_field") == "true"
    if not has_hazard:
        hazard_mappings = [m for m in mappings
                           if m.get("canonical_field") in ("hazard_type", "phenomenon")
                           and m.get("mapping_status") == "MAPPED"]
        if hazard_mappings:
            has_hazard = True

    if not has_hazard:
        return "MISSING", "BLOCKED", "No hazard/phenomenon field"

    if hazard_scope == "mixed":
        return "PRESENT", "BLOCKED_PHENOMENON_SEPARATION_REQUIRED", \
            "Mixed event — separation needed"

    return "PRESENT", "PASS", ""


def audit_gate(candidate: dict, event: dict,
               mappings: list[dict]) -> dict:
    temporal_status, temporal_match = check_temporal(candidate, event, mappings)
    hazard_status, phenom_status, phenom_note = check_phenomenon(
        candidate, event, mappings)

    temporal_gate = "PASS" if temporal_status.startswith("PRESENT") else "BLOCKED"
    phenomenon_gate = "PASS" if phenom_status == "PASS" else "BLOCKED"

    can_advance = (temporal_gate == "PASS" and phenomenon_gate == "PASS")

    blockers = []
    if temporal_gate == "BLOCKED":
        blockers.append("no_event_date" if temporal_status == "MISSING"
                        else "event_date_unverified")
    if phenomenon_gate == "BLOCKED":
        blockers.append("phenomenon_separation_required"
                        if "separation" in phenom_status.lower()
                        else "no_hazard_field")

    actions = []
    if "no_event_date" in blockers:
        actions.append("Obtain event date from source")
    if "phenomenon_separation_required" in blockers:
        actions.append("Request data with flood/landslide separated")
    if "no_hazard_field" in blockers:
        actions.append("Identify hazard type field in data")

    return {
        "event_date_status": temporal_status,
        "event_date_match": temporal_match,
        "hazard_status": hazard_status,
        "phenomenon_status": phenom_status,
        "phenomenon_separated": str(phenom_status == "PASS").lower(),
        "temporal_gate_status": temporal_gate,
        "phenomenon_gate_status": phenomenon_gate,
        "can_advance_to_supervisor_review": str(can_advance).lower(),
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
        "blocking_reason": "|".join(blockers),
        "required_action": "; ".join(actions),
        "notes": phenom_note,
    }


def main():
    parser = argparse.ArgumentParser(
        description="v1uh — Phenomenon and Temporal Gate Audit")
    parser.add_argument("--candidates",
                        default="datasets/protocolo_c/v1uh_observed_geometry_candidate_registry.csv")
    parser.add_argument("--events",
                        default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--mappings",
                        default="datasets/protocolo_c/v1uh_event_field_mapping_registry.csv")
    parser.add_argument("--out",
                        default="datasets/protocolo_c/v1uh_phenomenon_temporal_gate_audit.csv")
    args = parser.parse_args()

    candidates = load_csv(args.candidates)
    events_by_id = {e["event_id"]: e for e in load_csv(args.events)}
    mappings = load_csv(args.mappings)

    mappings_by_cand = {}
    for m in mappings:
        mappings_by_cand.setdefault(m.get("candidate_id", ""), []).append(m)

    rows = []
    seq = 0
    for cand in candidates:
        event_id = cand.get("event_id", "")
        event = events_by_id.get(event_id, {})
        cand_mappings = mappings_by_cand.get(cand.get("candidate_id", ""), [])
        result = audit_gate(cand, event, cand_mappings)
        rows.append({
            "gate_id": f"PGATE_{PROTOCOL_VERSION}_{seq:04d}",
            "candidate_id": cand.get("candidate_id", ""),
            "event_id": event_id,
            "asset_id": cand.get("asset_id", ""),
            **result,
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    can_adv = sum(1 for r in rows
                  if r["can_advance_to_supervisor_review"] == "true")
    print(f"[Phenomenon Temporal Gate Audit v1uh] {len(rows)} audited")
    print(f"  Can advance to supervisor review: {can_adv}")
    print(f"  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
v1ug — Supervisor Review Checklist Generator

Instantiates the supervisor review checklist per event.
All items default to NOT_EVALUATED. No review is executed —
this is a template for future human supervisor review.
"""

import argparse
import csv
import os

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1ug"

COLUMNS = [
    "checklist_entry_id", "event_id", "item_id", "item_text",
    "reviewer_role", "blocking_if_missing", "decision_options",
    "current_decision", "decision_rationale", "supervisor_review_completed",
]

GUARDRAILS = {
    "supervisor_review_completed": False,
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
}


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_yaml(path: str) -> dict:
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def generate_checklists(events: list[dict], policy: dict) -> list[dict]:
    items = policy.get("checklist_items", [])
    rows = []
    seq = 0
    for event in events:
        event_id = event["event_id"]
        for item in items:
            default = item.get("default_response", "NOT_EVALUATED")
            rows.append({
                "checklist_entry_id": f"CK_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id,
                "item_id": item["item_id"],
                "item_text": item["item"],
                "reviewer_role": item.get("reviewer_role", "supervisor"),
                "blocking_if_missing": str(item.get("blocking_if_missing", True)).lower(),
                "decision_options": "|".join(item.get("decision_options", [])),
                "current_decision": default,
                "decision_rationale": "",
                "supervisor_review_completed": "false",
            })
            seq += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description="v1ug — Supervisor Review Checklist Generator")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--policy", default="configs/protocolo_c/v1ug_supervisor_review_policy.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ug_supervisor_review_checklist.csv")
    args = parser.parse_args()

    events = load_csv(args.events)
    policy = load_yaml(args.policy)

    rows = generate_checklists(events, policy)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_events = len({r["event_id"] for r in rows})
    n_items = len(policy.get("checklist_items", []))
    not_eval = sum(1 for r in rows if r["current_decision"] == "NOT_EVALUATED")
    print(f"[Supervisor Review Checklist v1ug] {len(rows)} entries ({n_events} events x {n_items} items)")
    print(f"  NOT_EVALUATED: {not_eval} | supervisor_review_completed=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
v1ug — Formal Request Finalizer

Reads the institution targets config and the gap matrix to produce
a concrete formal request queue: which institution to contact,
for which event, requesting which data, resolving which gates.
No request is sent — output is a registry for human action.
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
    "request_id", "event_id", "institution_id", "institution_name",
    "priority", "gates_to_resolve", "requested_data_summary",
    "requested_formats", "contact_url", "sensitive_data_policy",
    "request_status", "human_action_required",
]

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
    "formal_request_only": True,
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


def build_request_queue(events: list[dict], targets_cfg: dict,
                        gap_rows: list[dict]) -> list[dict]:
    institutions = targets_cfg.get("institutions", [])
    event_ids = {e["event_id"] for e in events}

    gap_by_event = {}
    for g in gap_rows:
        gap_by_event.setdefault(g["event_id"], []).append(g)

    rows = []
    seq = 0
    for inst in institutions:
        inst_id = inst["institution_id"]
        applicable = inst.get("applicable_events", [])
        for ev_id in applicable:
            if ev_id not in event_ids:
                continue
            gates = inst.get("gates_to_resolve", [])
            data_items = []
            for item in inst.get("requested_data", []):
                if isinstance(item, dict):
                    data_items.extend(f"{k}: {v}" for k, v in item.items())
                else:
                    data_items.append(str(item))
            rows.append({
                "request_id": f"REQ_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": ev_id,
                "institution_id": inst_id,
                "institution_name": inst.get("name", ""),
                "priority": str(inst.get("priority", 9)),
                "gates_to_resolve": "|".join(gates),
                "requested_data_summary": "; ".join(data_items[:3])
                    + (f" (+{len(data_items)-3} more)" if len(data_items) > 3 else ""),
                "requested_formats": "|".join(inst.get("requested_formats", [])),
                "contact_url": inst.get("contact_url", ""),
                "sensitive_data_policy": inst.get("sensitive_data_policy", ""),
                "request_status": "PENDING_HUMAN_ACTION",
                "human_action_required": "true",
            })
            seq += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description="v1ug — Formal Request Finalizer")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--targets", default="configs/protocolo_c/v1ug_formal_request_targets.yaml")
    parser.add_argument("--gap-matrix", default="datasets/protocolo_c/v1ug_event_gap_matrix.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ug_formal_request_queue.csv")
    args = parser.parse_args()

    events = load_csv(args.events)
    targets_cfg = load_yaml(args.targets)
    gap_rows = load_csv(args.gap_matrix)

    rows = build_request_queue(events, targets_cfg, gap_rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Formal Request Finalizer v1ug] {len(rows)} requests queued")
    for r in rows:
        print(f"  {r['request_id']}: {r['event_id']} -> {r['institution_id']} [{r['request_status']}]")
    print(f"\n  formal_request_only=true | human_action_required=true (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

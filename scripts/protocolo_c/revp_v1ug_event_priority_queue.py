#!/usr/bin/env python3
"""
v1ug — Event Priority Queue

Ranks events by actionability: which events have the fewest
remaining blockers and where formal requests would have most impact.
No promotion, no ground reference, no label.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1ug"

COLUMNS = [
    "rank", "event_id", "region", "city", "review_package_status",
    "blocking_dimensions_count", "fail_gap_count", "pass_gap_count",
    "formal_request_count", "priority_score", "recommended_next_step",
    "can_create_ground_reference", "can_create_training_label",
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_priority(event: dict, readiness: dict, gaps: list[dict],
                     requests: list[dict]) -> dict:
    event_id = event["event_id"]
    ev_gaps = [g for g in gaps if g["event_id"] == event_id]
    ev_requests = [r for r in requests if r["event_id"] == event_id]

    fail_count = sum(1 for g in ev_gaps if g["current_status"] == "FAIL")
    pass_count = sum(1 for g in ev_gaps if g["current_status"] == "PASS")
    blocking = int(readiness.get("blocking_dimensions_count", "99"))

    base_priority = int(event.get("priority", "5"))
    score = (12 - fail_count) * 10 + pass_count * 5 - blocking * 15 + (10 - base_priority) * 3

    overall = readiness.get("overall_readiness", "NOT_READY_FOR_GROUND_REFERENCE")
    if overall == "WAITING_OBSERVED_GEOMETRY":
        next_step = "Enviar pedido formal para geometria observada (Defesa Civil/SGB/COMPDEC)"
    elif overall == "WAITING_PHENOMENON_SEPARATION":
        next_step = "Solicitar dados com separação inundação/deslizamento (SGB_CPRM/DRM_RJ)"
    elif overall == "READY_FOR_FORMAL_REQUEST":
        next_step = "Enviar pedidos formais conforme fila de requisições"
    else:
        next_step = "Adquirir evidência faltante antes de próximo passo"

    return {
        "event_id": event_id,
        "region": event.get("region", ""),
        "city": event.get("city", ""),
        "review_package_status": readiness.get("overall_readiness", ""),
        "blocking_dimensions_count": str(blocking),
        "fail_gap_count": str(fail_count),
        "pass_gap_count": str(pass_count),
        "formal_request_count": str(len(ev_requests)),
        "priority_score": str(score),
        "recommended_next_step": next_step,
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
    }


def main():
    parser = argparse.ArgumentParser(description="v1ug — Event Priority Queue")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--readiness", default="datasets/protocolo_c/v1ug_ground_reference_readiness_matrix.csv")
    parser.add_argument("--gap-matrix", default="datasets/protocolo_c/v1ug_event_gap_matrix.csv")
    parser.add_argument("--requests", default="datasets/protocolo_c/v1ug_formal_request_queue.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ug_event_priority_queue.csv")
    args = parser.parse_args()

    events = load_csv(args.events)
    readiness_rows = {r["event_id"]: r for r in load_csv(args.readiness)}
    gap_rows = load_csv(args.gap_matrix)
    requests = load_csv(args.requests)

    scored = []
    for event in events:
        rd = readiness_rows.get(event["event_id"], {})
        row = compute_priority(event, rd, gap_rows, requests)
        scored.append(row)

    scored.sort(key=lambda r: -int(r["priority_score"]))
    for rank, row in enumerate(scored, 1):
        row["rank"] = str(rank)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(scored)

    print(f"[Event Priority Queue v1ug] {len(scored)} events ranked")
    for r in scored:
        print(f"  #{r['rank']}: {r['event_id']} score={r['priority_score']} -> {r['recommended_next_step'][:60]}...")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
v1ug — Ground Reference Readiness Matrix

Evaluates each event against the readiness dimensions defined
in v1ug_ground_reference_readiness_policy.yaml.
No event can reach READY_FOR_GROUND_REFERENCE in this stage.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1ug"

DIMENSIONS = [
    "temporal_readiness",
    "hydromet_readiness",
    "phenomenon_readiness",
    "locality_readiness",
    "geometry_readiness",
    "overlay_readiness",
    "supervisor_review_readiness",
    "label_readiness",
]

COLUMNS = [
    "event_id", "overall_readiness",
] + [f"{d}_status" for d in DIMENSIONS] + [
    "missing_dimensions", "blocking_dimensions_count",
    "can_create_ground_reference", "can_create_training_label",
    "next_required_action",
]

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
    "can_reopen_protocol_b": False,
}


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_readiness(event: dict, gap_rows: list[dict],
                       _package: dict) -> dict:
    event_id = event["event_id"]
    gaps = {g["gap_name"]: g["current_status"] for g in gap_rows
            if g["event_id"] == event_id}
    has_date = bool(event.get("start_date"))
    has_city = bool(event.get("city"))

    dim_status = {}

    dim_status["temporal_readiness"] = (
        "PASS" if has_date and gaps.get("event_date_confirmed") == "PASS"
        else "FAIL"
    )

    hydromet_gap = gaps.get("hydromet_temporal_anchor", "FAIL")
    dim_status["hydromet_readiness"] = (
        "PASS" if hydromet_gap == "PASS"
        else "NEEDS_REVIEW" if hydromet_gap == "NEEDS_REVIEW"
        else "FAIL"
    )

    phenom = gaps.get("phenomenon_separated", "FAIL")
    dim_status["phenomenon_readiness"] = (
        "PASS" if phenom == "PASS"
        else "FAIL"
    )

    dim_status["locality_readiness"] = (
        "PASS" if has_city and gaps.get("locality_confirmed") == "PASS"
        else "FAIL"
    )

    dim_status["geometry_readiness"] = "FAIL"
    dim_status["overlay_readiness"] = "FAIL"
    dim_status["supervisor_review_readiness"] = "FAIL"
    dim_status["label_readiness"] = "FAIL"

    required_for_gr = [
        "temporal_readiness", "phenomenon_readiness", "locality_readiness",
        "geometry_readiness", "overlay_readiness", "supervisor_review_readiness",
    ]
    missing = [d for d in required_for_gr if dim_status.get(d) != "PASS"]
    blocking_count = len(missing)

    overall = "NOT_READY_FOR_GROUND_REFERENCE"
    if blocking_count == 0:
        overall = "NOT_READY_FOR_GROUND_REFERENCE"

    if dim_status["temporal_readiness"] == "PASS" and dim_status["phenomenon_readiness"] == "PASS" \
       and dim_status["locality_readiness"] == "PASS":
        if dim_status["geometry_readiness"] == "FAIL":
            overall = "WAITING_OBSERVED_GEOMETRY"
    elif dim_status["temporal_readiness"] == "PASS" and dim_status["phenomenon_readiness"] == "FAIL":
        overall = "WAITING_PHENOMENON_SEPARATION"
    elif dim_status["temporal_readiness"] == "PASS":
        overall = "READY_FOR_FORMAL_REQUEST"

    if missing:
        next_action = f"Resolve: {missing[0]}"
    else:
        next_action = "All dimensions PASS — but ground reference still blocked in v1ug"

    row = {"event_id": event_id, "overall_readiness": overall}
    for d in DIMENSIONS:
        row[f"{d}_status"] = dim_status.get(d, "FAIL")
    row["missing_dimensions"] = "|".join(missing) if missing else "NONE"
    row["blocking_dimensions_count"] = str(blocking_count)
    row["can_create_ground_reference"] = "false"
    row["can_create_training_label"] = "false"
    row["next_required_action"] = next_action
    return row


def main():
    parser = argparse.ArgumentParser(description="v1ug — Ground Reference Readiness Matrix")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--gap-matrix", default="datasets/protocolo_c/v1ug_event_gap_matrix.csv")
    parser.add_argument("--packages", default="datasets/protocolo_c/v1ug_event_review_package_registry.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ug_ground_reference_readiness_matrix.csv")
    args = parser.parse_args()

    events = load_csv(args.events)
    gap_rows = load_csv(args.gap_matrix)
    packages = {r["event_id"]: r for r in load_csv(args.packages)}

    rows = []
    for event in events:
        pkg = packages.get(event["event_id"], {})
        row = evaluate_readiness(event, gap_rows, pkg)
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Ground Reference Readiness Matrix v1ug] {len(rows)} events evaluated")
    for r in rows:
        print(f"  {r['event_id']}: {r['overall_readiness']} (blocking={r['blocking_dimensions_count']})")
    print(f"\n  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

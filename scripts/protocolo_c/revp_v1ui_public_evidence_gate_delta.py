#!/usr/bin/env python3
"""
v1ui — Public Evidence Gate Delta

Compares v1uh (no formal responses) -> v1ui (public discovery).
Registers gains and remaining blockers per event.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1ui"

DELTA_COLUMNS = [
    "delta_id", "event_id", "dimension", "v1uh_status", "v1ui_status",
    "delta_type", "evidence_source", "notes",
]

DIMENSIONS = [
    "public_artifact_found", "geometry_found", "crs_found",
    "date_field_found", "hazard_field_found", "locality_found",
    "event_specificity_improved", "supervisor_review_possible",
    "ground_reference_possible", "label_possible",
]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="v1ui — Public Evidence Gate Delta")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--extractions",
                        default="datasets/protocolo_c/v1ui_observed_geometry_extraction_registry.csv")
    parser.add_argument("--candidates",
                        default="datasets/protocolo_c/v1ui_event_geometry_candidate_registry.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ui_public_evidence_gate_delta.csv")
    args = parser.parse_args()

    events = load_csv(args.events)
    extractions = load_csv(args.extractions)
    candidates = load_csv(args.candidates)

    ext_by_event = {}
    for e in extractions:
        ext_by_event.setdefault(e.get("event_id", ""), []).append(e)
    cand_by_event = {}
    for c in candidates:
        cand_by_event.setdefault(c.get("event_id", ""), []).append(c)

    rows = []
    seq = 0

    for event in events:
        eid = event["event_id"]
        ev_ext = ext_by_event.get(eid, [])
        ev_cand = cand_by_event.get(eid, [])

        has_artifact = len(ev_ext) > 0
        has_geom = any(e.get("has_geometry") == "true" or
                       e.get("has_coordinate_fields") == "true" for e in ev_ext)
        has_crs = any(bool(e.get("crs")) for e in ev_ext)
        has_date = any(e.get("has_date_field") == "true" for e in ev_ext)
        has_hazard = any(e.get("has_hazard_field") == "true" for e in ev_ext)
        has_locality = any(e.get("has_locality_field") == "true" for e in ev_ext)
        is_specific = any(e.get("is_event_specific") == "true" for e in ev_ext)
        ready_review = any("READY_FOR_SUPERVISOR_REVIEW" in c.get("max_status", "")
                          for c in ev_cand)

        dim_vals = {
            "public_artifact_found": ("NO", "YES" if has_artifact else "NO"),
            "geometry_found": ("NO", "YES" if has_geom else "NO"),
            "crs_found": ("NO", "YES" if has_crs else "NO"),
            "date_field_found": ("NO", "YES" if has_date else "NO"),
            "hazard_field_found": ("NO", "YES" if has_hazard else "NO"),
            "locality_found": ("NO", "YES" if has_locality else "NO"),
            "event_specificity_improved": ("NO", "YES" if is_specific else "NO"),
            "supervisor_review_possible": ("NO", "YES" if ready_review else "NO"),
            "ground_reference_possible": ("NO", "NO"),
            "label_possible": ("NO", "NO"),
        }

        for dim in DIMENSIONS:
            v1uh_s, v1ui_s = dim_vals[dim]
            delta = "GAIN" if v1uh_s == "NO" and v1ui_s == "YES" else \
                    "NO_CHANGE" if v1uh_s == v1ui_s else "REGRESSION"
            rows.append({
                "delta_id": f"DELTA_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": eid, "dimension": dim,
                "v1uh_status": v1uh_s, "v1ui_status": v1ui_s,
                "delta_type": delta, "evidence_source": "public_discovery",
                "notes": "",
            })
            seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DELTA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    gains = sum(1 for r in rows if r["delta_type"] == "GAIN")
    print(f"[Public Evidence Gate Delta v1ui] {len(rows)} deltas | gains={gains}")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

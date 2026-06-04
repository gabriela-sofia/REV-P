#!/usr/bin/env python3
"""
v1ue — Event Evidence Scorecard

Scores evidence dimensions per event WITHOUT automatic promotion.
High score only sets next action — never creates label or ground truth.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1ue"

SCORECARD_COLUMNS = [
    "scorecard_id", "event_id", "region", "city", "hazard_scope",
    "temporal_evidence_score", "hydrometeorological_score",
    "phenomenon_typing_score", "locality_score", "geometry_score",
    "source_authority_score", "independence_score", "review_readiness_score",
    "aggregate_score", "classification", "can_create_ground_reference",
    "can_create_training_label", "ground_truth_operational",
    "supervisor_review_completed", "blocking_summary",
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify(scores: dict, hazard_scope: str, has_formal_block: bool) -> tuple[str, str]:
    t = scores["temporal_evidence_score"]
    h = scores["hydrometeorological_score"]
    p = scores["phenomenon_typing_score"]
    loc = scores["locality_score"]
    g = scores["geometry_score"]
    auth = scores["source_authority_score"]

    blocks = []

    # Phenomenon separation block for mixed events
    if hazard_scope == "mixed" and p < 0.5:
        blocks.append("PHENOMENON_SEPARATION_REQUIRED")

    if g == 0:
        blocks.append("GEOMETRY_MISSING")

    if has_formal_block:
        blocks.append("FORMAL_REQUEST_REQUIRED")

    # Classification — never promotes ground truth
    if g > 0 and p >= 0.5 and auth >= 0.5:
        classification = "READY_FOR_HUMAN_REVIEW"
    elif hazard_scope == "mixed" and p < 0.5:
        classification = "BLOCKED_PHENOMENON_SEPARATION_REQUIRED"
    elif g == 0 and has_formal_block:
        classification = "BLOCKED_FORMAL_REQUEST_REQUIRED"
    elif g == 0 and t >= 0.3 and (h > 0 or loc > 0):
        if t >= 0.5 and p >= 0.3 and loc > 0:
            classification = "OBSERVATIONAL_CANDIDATE_MODERATE"
        else:
            classification = "OBSERVATIONAL_CANDIDATE_WEAK"
    elif t >= 0.3 and g == 0 and p < 0.3:
        classification = "TEMPORAL_ANCHOR_ONLY"
    elif g == 0:
        classification = "BLOCKED_GEOMETRY_MISSING"
    else:
        classification = "CONTEXT_ONLY"

    return classification, "|".join(blocks) if blocks else "none"


def score_event(
    event: dict,
    windows: list[dict],
    stations: list[dict],
    observations: list[dict],
    resolutions: list[dict],
) -> dict:
    event_id = event["event_id"]
    hazard_scope = event.get("hazard_scope", "unknown")

    ev_windows = [w for w in windows if w["event_id"] == event_id]
    ev_stations = [s for s in stations if s["event_id"] == event_id]
    ev_obs = [o for o in observations if o["event_id"] == event_id]
    ev_res = [r for r in resolutions if r["event_id"] == event_id]

    # Generic portal homepages must NOT contribute real locality/hazard evidence.
    # Their nav/footer terms (e.g. "rua", "avenida") are false-positive signals.
    def is_substantive(o: dict) -> bool:
        return o.get("event_specificity") != "GENERIC_PORTAL_HOMEPAGE"

    substantive_obs = [o for o in ev_obs if is_substantive(o)]

    # temporal_evidence_score: windows + stations that can anchor temporal
    temporal = 0.0
    if ev_windows:
        temporal += 0.3
    if any(s.get("can_anchor_temporal_evidence") == "true" for s in ev_stations):
        temporal += 0.3
    if any(r.get("is_year_specific") == "true" for r in ev_res):
        temporal += 0.2
    temporal = min(temporal, 1.0)

    # hydrometeorological_score: only substantive (non-portal) assets count
    hydromet = 0.0
    if any(o.get("evidence_role") == "temporal_anchor" for o in substantive_obs):
        hydromet += 0.3
    if any(o.get("observed_variable") == "precipitation" for o in substantive_obs):
        hydromet += 0.2
    hydromet = min(hydromet, 1.0)

    # phenomenon_typing_score: single hazard easier to type
    phenomenon = 0.0
    if hazard_scope in ("flood", "inundation", "urban_flooding"):
        phenomenon = 0.5
    elif hazard_scope == "mixed":
        phenomenon = 0.1  # mixed needs separation
    if any(o.get("hazard_terms_found") for o in substantive_obs):
        phenomenon = min(phenomenon + 0.2, 1.0)

    # locality_score: locality terms only count from substantive (non-portal) assets
    locality = 0.0
    if any(o.get("locality_terms_found") for o in substantive_obs):
        locality = 0.3

    # geometry_score: observational geometry available
    geometry = 0.0
    if any(o.get("geometry_metadata_available") == "true" for o in ev_obs):
        geometry = 0.5

    # source_authority_score: official sources present
    authority = 0.0
    if any(s.get("is_official") == "true" for s in ev_stations):
        authority += 0.4
    if ev_res:
        authority += 0.2
    authority = min(authority, 1.0)

    # independence_score: distinct source types
    source_types = set(o.get("source_id") for o in ev_obs) | set(s.get("source_id") for s in ev_stations)
    independence = min(len(source_types) * 0.2, 1.0)

    # review_readiness_score: combination
    review_readiness = round((temporal + authority + independence) / 3, 2)

    scores = {
        "temporal_evidence_score": round(temporal, 2),
        "hydrometeorological_score": round(hydromet, 2),
        "phenomenon_typing_score": round(phenomenon, 2),
        "locality_score": round(locality, 2),
        "geometry_score": round(geometry, 2),
        "source_authority_score": round(authority, 2),
        "independence_score": round(independence, 2),
        "review_readiness_score": review_readiness,
    }

    has_formal_block = any(
        r.get("source_id") in ("SGB_CPRM_CARTOGRAFIA", "ANA_HIDROWEB")
        and r.get("resolution_status") not in ("EVENT_SPECIFIC_RESOLVED",)
        for r in ev_res
    )

    classification, blocking = classify(scores, hazard_scope, has_formal_block)

    aggregate = round(sum(scores.values()) / len(scores), 2)

    return {
        "scorecard_id": f"SCD_{PROTOCOL_VERSION}_{event_id}",
        "event_id": event_id,
        "region": event.get("region", ""),
        "city": event.get("city", ""),
        "hazard_scope": hazard_scope,
        **scores,
        "aggregate_score": aggregate,
        "classification": classification,
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
        "ground_truth_operational": "false",
        "supervisor_review_completed": "false",
        "blocking_summary": blocking,
    }


def main():
    parser = argparse.ArgumentParser(description="v1ue — Event Evidence Scorecard")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--windows", default="datasets/protocolo_c/v1ue_event_temporal_window_registry.csv")
    parser.add_argument("--stations", default="datasets/protocolo_c/v1ue_station_candidate_registry.csv")
    parser.add_argument("--observations", default="datasets/protocolo_c/v1ue_observation_series_registry.csv")
    parser.add_argument("--resolutions", default="datasets/protocolo_c/v1ue_official_dataset_resolution_registry.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ue_event_evidence_scorecard.csv")
    args = parser.parse_args()

    events = load_csv(args.events)
    windows = load_csv(args.windows)
    stations = load_csv(args.stations)
    observations = load_csv(args.observations)
    resolutions = load_csv(args.resolutions)

    rows = []
    for event in events:
        rows.append(score_event(event, windows, stations, observations, resolutions))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Event Evidence Scorecard v1ue] {len(rows)} events scored")
    for r in rows:
        print(f"  {r['event_id']}: {r['classification']} (agg={r['aggregate_score']})")
    print(f"\n  can_create_ground_reference=false (all)")
    print(f"  ground_truth_operational=false (all)")
    print(f"\nScorecard: {args.out}")


if __name__ == "__main__":
    main()

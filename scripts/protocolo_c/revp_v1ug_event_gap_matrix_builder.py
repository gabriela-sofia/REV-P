#!/usr/bin/env python3
"""
v1ug — Event Gap Matrix Builder

Creates a matrix of evidence gaps per event × gap dimension.
training_label_allowed is always FAIL.
observed_geometry_available is FAIL for all events (none yet acquired).
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1ug"

GAP_COLUMNS = [
    "gap_id", "event_id", "gap_name", "current_status", "evidence_support",
    "blocking_severity", "required_action", "target_institution",
    "can_be_resolved_by_programming", "can_be_resolved_by_formal_request",
    "can_be_resolved_by_human_review", "notes",
]

GAP_DEFINITIONS = [
    ("event_date_confirmed",       "LOW",      "Confirm event dates from official registry",              "",                         "true",  "false", "false"),
    ("official_source_traceable",  "LOW",      "Verify all sources have official citation",               "",                         "true",  "false", "false"),
    ("hydromet_temporal_anchor",   "MEDIUM",   "Acquire official hydromet series for event window",       "CEMADEN|ANA_HIDROWEB",     "true",  "true",  "false"),
    ("official_station_coordinates","MEDIUM",  "Resolve official station coordinates from catalog",       "INMET_BDMEP|ANA_HIDROWEB", "true",  "false", "false"),
    ("phenomenon_type_confirmed",  "HIGH",     "Identify and document hazard type from official report",  "SGB_CPRM|DEFESA_CIVIL",    "false", "true",  "true"),
    ("phenomenon_separated",       "CRITICAL", "Separate flood from landslide in evidence",               "SGB_CPRM|DRM_RJ_NADE",     "false", "true",  "true"),
    ("locality_confirmed",         "HIGH",     "Identify affected localities (bairro/rua/coordinate)",    "DEFESA_CIVIL|COMPDEC",     "false", "true",  "true"),
    ("observed_geometry_available","CRITICAL", "Obtain observed geometry (polygon/point of occurrence)",  "DEFESA_CIVIL|SGB_CPRM|COMPDEC", "false", "true", "false"),
    ("geometry_crs_available",     "HIGH",     "Obtain CRS for observed geometry",                        "DEFESA_CIVIL|SGB_CPRM",    "false", "true",  "true"),
    ("patch_overlay_possible",     "CRITICAL", "Execute patch-evidence overlay (future step)",            "",                         "true",  "false", "false"),
    ("supervisor_review_possible", "HIGH",     "Schedule human supervisor review",                        "",                         "false", "false", "true"),
    ("training_label_allowed",     "CRITICAL", "Training label creation — always FAIL this stage",        "",                         "false", "false", "false"),
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_gap(gap_name: str, event: dict, hydromet: dict, v1ue: dict) -> tuple[str, str]:
    hazard = event.get("hazard_scope", "")
    h_level = hydromet.get("hydromet_evidence_level", "")
    has_series = hydromet.get("has_official_station_series") == "true"
    has_coord = hydromet.get("has_station_coordinates") == "true"
    has_precip = hydromet.get("has_precipitation_during_event") == "true"

    g = gap_name
    if g == "event_date_confirmed":
        return ("PASS" if event.get("start_date") else "FAIL",
                f"date={event.get('start_date', '')}")
    if g == "official_source_traceable":
        return "PASS", "Source registry v1uc/v1ud established"
    if g == "hydromet_temporal_anchor":
        if has_series and has_precip:
            return "PASS", f"INMET series extracted; hydromet_level={h_level}"
        if has_series and not has_precip:
            return "NEEDS_REVIEW", f"Series extracted but insufficient coverage: {h_level}"
        return "FAIL", "No official hydromet series extracted"
    if g == "official_station_coordinates":
        return ("PASS" if has_coord else "NEEDS_REVIEW",
                "FROM_OFFICIAL_CATALOG via INMET API" if has_coord else "MISSING")
    if g == "phenomenon_type_confirmed":
        if hazard in ("flood", "inundation", "urban_flooding"):
            return "PASS", f"Single hazard: {hazard}"
        return ("NEEDS_REVIEW" if hazard != "mixed" else "FAIL",
                f"hazard_scope={hazard}")
    if g == "phenomenon_separated":
        if hazard == "mixed":
            return "FAIL", "Evento misto — separação inundação/deslizamento necessária"
        if hazard in ("flood", "inundation", "urban_flooding"):
            return "PASS", "Single phenomenon — no separation needed"
        return "NEEDS_REVIEW", f"hazard_scope={hazard}"
    if g == "locality_confirmed":
        return ("PASS" if event.get("city") else "FAIL",
                f"city={event.get('city', '')}")
    if g == "observed_geometry_available":
        return "FAIL", "No observed geometry acquired (requires formal request)"
    if g == "geometry_crs_available":
        return "NOT_APPLICABLE", "Geometry not yet available"
    if g == "patch_overlay_possible":
        return "FAIL", "No geometry available — overlay not executable"
    if g == "supervisor_review_possible":
        return "FAIL", "Supervisor review not yet scheduled"
    if g == "training_label_allowed":
        return "FAIL", "Training label always FAIL: no ground reference, no label"
    return "NEEDS_REVIEW", "Not evaluated"


def main():
    parser = argparse.ArgumentParser(description="v1ug — Event Gap Matrix Builder")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--hydromet-scorecard", default="datasets/protocolo_c/v1uf_event_hydromet_scorecard.csv")
    parser.add_argument("--v1ue-scorecard", default="datasets/protocolo_c/v1ue_event_evidence_scorecard.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ug_event_gap_matrix.csv")
    args = parser.parse_args()

    events = load_csv(args.events)
    hydromet = {r["event_id"]: r for r in load_csv(args.hydromet_scorecard)}
    v1ue = {r["event_id"]: r for r in load_csv(args.v1ue_scorecard)}

    rows = []
    seq = 0
    for event in events:
        event_id = event["event_id"]
        hsc = hydromet.get(event_id, {})
        vsc = v1ue.get(event_id, {})

        for gap_name, severity, action, institution, prog, req, human in GAP_DEFINITIONS:
            status, support = evaluate_gap(gap_name, event, hsc, vsc)
            rows.append({
                "gap_id": f"GAP_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id,
                "gap_name": gap_name,
                "current_status": status,
                "evidence_support": support,
                "blocking_severity": severity,
                "required_action": action,
                "target_institution": institution,
                "can_be_resolved_by_programming": prog,
                "can_be_resolved_by_formal_request": req,
                "can_be_resolved_by_human_review": human,
                "notes": "",
            })
            seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GAP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    fail = sum(1 for r in rows if r["current_status"] == "FAIL")
    pass_ = sum(1 for r in rows if r["current_status"] == "PASS")
    review = sum(1 for r in rows if r["current_status"] in ("NEEDS_REVIEW", "NOT_APPLICABLE"))
    print(f"[Event Gap Matrix v1ug] {len(rows)} gaps | FAIL:{fail} PASS:{pass_} REVIEW:{review}")
    print(f"  training_label_allowed=FAIL (enforced in all events)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()

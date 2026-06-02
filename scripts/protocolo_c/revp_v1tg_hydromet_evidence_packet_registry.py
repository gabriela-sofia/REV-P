"""REV-P v1tg — Hydromet evidence packet registry.

One packet per event_candidate_id. Consolidates nearest station, rainfall
context, and support level. Never validates events.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tg_v1tm_hydromet_review_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_extended, scan_guardrails_extended,
    parse_float_safe, hash_short,
    classify_station_coverage, classify_precipitation_context,
    classify_hydromet_support_level,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_PKT  = _p("REVP_V1TG_OUT_PKT",  DATASETS / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv")
OUT_SUM  = _p("REVP_V1TG_OUT_SUM",  DATASETS / "protocol_c_hydromet_evidence_packet_summary_v1tg.csv")
SCHEMA_P = _p("REVP_V1TG_SCHEMA_P", SCHEMAS  / "protocol_c_hydromet_evidence_packet_registry_v1tg_schema.csv")
SCHEMA_S = _p("REVP_V1TG_SCHEMA_S", SCHEMAS  / "protocol_c_hydromet_evidence_packet_summary_v1tg_schema.csv")
DOC      = _p("REVP_V1TG_DOC",      DOCS     / "revp_v1tg_hydromet_evidence_packet_registry.md")

PKT_FIELDS = [
    "hydromet_packet_id", "event_candidate_id", "region", "event_window",
    "nearest_station_code", "nearest_station_name", "nearest_station_distance_km",
    "rain_1d", "rain_3d", "rain_7d", "max_1d_in_window",
    "station_coverage_status", "precipitation_context_status",
    "hydromet_support_level", "evidence_role",
    "review_only", "hydromet_validates_event", "hydromet_is_negative_evidence",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    bridge     = read_csv_safe(DATASETS / "protocol_c_hydromet_event_evidence_bridge_v1td.csv")
    station_reg= read_csv_safe(DATASETS / "protocol_c_inmet_canonical_station_registry_v1ta.csv")
    windows    = read_csv_safe(DATASETS / "protocol_c_event_date_windows_v1ss.csv")
    features   = read_csv_safe(DATASETS / "protocol_c_rolling_rainfall_context_features_v1su.csv")

    station_meta: dict[str, dict] = {r["station_code"]: r for r in station_reg}
    windows_by_candidate: dict[str, dict] = {}
    for w in windows:
        if not w.get("blocked_reason"):
            windows_by_candidate[w.get("event_candidate_id", "")] = w

    # Best feature per event window (highest rain_7d)
    best_feat: dict[str, dict] = {}
    for f in features:
        if f.get("feature_status") != "ROLLING_CONTEXT_REVIEW_ONLY":
            continue
        wid = f.get("event_window_id", "")
        cur = best_feat.get(wid)
        if cur is None or parse_float_safe(f.get("rain_7d","0")) > parse_float_safe(cur.get("rain_7d","0")):
            best_feat[wid] = f

    rows: list[dict[str, Any]] = []

    for b in bridge:
        cid    = b.get("event_candidate_id", "")
        region = b.get("region", "")
        code   = b.get("nearest_station_code", "")
        dist   = parse_float_safe(b.get("nearest_station_distance_km", ""), -1.0)
        r1d    = b.get("rain_1d", "")
        r3d    = b.get("rain_3d", "")
        r7d    = b.get("rain_7d", "")
        max1d  = b.get("max_1d_in_window", "")
        ev_win = b.get("event_window", "")

        st  = station_meta.get(code, {})
        win = windows_by_candidate.get(cid)
        wid = win.get("event_window_id", "") if win else ""
        feat= best_feat.get(wid) if wid else None

        # Use feature values when richer
        if feat:
            r1d   = r1d   or feat.get("rain_1d", "")
            r3d   = r3d   or feat.get("rain_3d", "")
            r7d   = r7d   or feat.get("rain_7d", "")
            max1d = max1d or feat.get("max_1d_in_window", "")

        r7d_f = parse_float_safe(r7d, -1.0)
        r1d_f = parse_float_safe(r1d, -1.0)

        row: dict[str, Any] = {
            "hydromet_packet_id":              f"V1TG_PKT_{hash_short(cid, 10)}",
            "event_candidate_id":              cid,
            "region":                          region,
            "event_window":                    ev_win,
            "nearest_station_code":            code,
            "nearest_station_name":            st.get("station_name", ""),
            "nearest_station_distance_km":     b.get("nearest_station_distance_km", ""),
            "rain_1d":                         r1d,
            "rain_3d":                         r3d,
            "rain_7d":                         r7d,
            "max_1d_in_window":                max1d,
            "station_coverage_status":         classify_station_coverage(dist),
            "precipitation_context_status":    classify_precipitation_context(r7d_f, r1d_f),
            "hydromet_support_level":          classify_hydromet_support_level(
                                                  dist, r7d_f, win is not None),
            "evidence_role":                   "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY",
            "notes":                           "",
        }
        row.update(guardrail_row_extended())
        rows.append(row)

    if not rows:
        rows = [{
            "hydromet_packet_id": "FAIL_CLOSED_NO_BRIDGE",
            "event_candidate_id": "", "region": "", "event_window": "",
            "nearest_station_code": "", "nearest_station_name": "",
            "nearest_station_distance_km": "", "rain_1d": "", "rain_3d": "",
            "rain_7d": "", "max_1d_in_window": "",
            "station_coverage_status":      "NO_STATION",
            "precipitation_context_status": "PRECIPITATION_DATA_MISSING",
            "hydromet_support_level":       "HYDROMET_CONTEXT_WAITING_EVENT_WINDOW",
            "evidence_role": "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY",
            "notes": "no bridge input", **guardrail_row_extended(),
        }]

    violations = scan_guardrails_extended(rows, "v1tg_packets")
    if violations:
        raise ValueError(f"Guardrail violations in v1tg: {violations[:3]}")

    write_csv_with_header(OUT_PKT, rows, PKT_FIELDS)
    write_schema(SCHEMA_P, PKT_FIELDS, "v1tg_packet")

    available = sum(1 for r in rows if r["hydromet_support_level"] == "HYDROMET_CONTEXT_AVAILABLE")
    summary = [
        {"stat_key": "total_packets",             "stat_value": str(len(rows))},
        {"stat_key": "context_available",         "stat_value": str(available)},
        {"stat_key": "validates_event",           "stat_value": "false"},
        {"stat_key": "creates_negative",          "stat_value": "false"},
        {"stat_key": "stage",                     "stat_value": "v1tg"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tg_summary")

    write_doc(DOC, "v1tg — Hydromet Evidence Packet Registry", [
        "## Objetivo",
        "Um pacote de evidência hidromet por event_candidate_id. Consolida "
        "estação mais próxima, acumulados e classificação contextual.",
        f"## Resultado\nPacotes: {len(rows)}. Contexto disponível: {available}.",
        "## Limitação",
        "Pacotes hidromet são contexto, não validação. rain_7d alto não confirma evento.",
    ])
    print(f"[v1tg] packets={len(rows)} available={available}")
    return {"packets": len(rows), "available": available}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tg hydromet packet registry").parse_args()
    run()

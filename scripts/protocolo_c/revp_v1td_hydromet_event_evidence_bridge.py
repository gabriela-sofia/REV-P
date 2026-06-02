"""REV-P v1td — Hydromet event evidence bridge to review packets.

For each Protocol C event window (v1ss), assembles a single hydromet
evidence packet: nearest station, rolling rainfall from canonical index (v1tc),
crosswalk entry (v1sv). Review-only; never validates event automatically.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
from collections import defaultdict

from revp_v1ta_v1tf_inmet_canonical_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row, hash_short, parse_date_safe, rolling_window_summary,
    parse_decimal_comma_float,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_BRG  = _p("REVP_V1TD_OUT_BRG",  DATASETS / "protocol_c_hydromet_event_evidence_bridge_v1td.csv")
OUT_SUM  = _p("REVP_V1TD_OUT_SUM",  DATASETS / "protocol_c_hydromet_event_evidence_bridge_summary_v1td.csv")
SCHEMA_B = _p("REVP_V1TD_SCHEMA_B", SCHEMAS  / "protocol_c_hydromet_event_evidence_bridge_v1td_schema.csv")
SCHEMA_S = _p("REVP_V1TD_SCHEMA_S", SCHEMAS  / "protocol_c_hydromet_event_evidence_bridge_summary_v1td_schema.csv")
DOC      = _p("REVP_V1TD_DOC",      DOCS     / "revp_v1td_hydromet_event_evidence_bridge.md")

BRG_FIELDS = [
    "hydromet_evidence_id", "event_candidate_id", "region", "event_window",
    "nearest_station_code", "nearest_station_distance_km",
    "rain_1d", "rain_3d", "rain_7d", "max_1d_in_window",
    "evidence_role", "supports_manual_review", "does_not_validate_event",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative",
    "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    wins      = read_csv_safe(DATASETS / "protocol_c_event_date_windows_v1ss.csv")
    feat_rows = read_csv_safe(DATASETS / "protocol_c_rolling_rainfall_context_features_v1su.csv")
    station_rows = read_csv_safe(DATASETS / "protocol_c_inmet_canonical_station_registry_v1ta.csv")

    # Station distance lookup
    station_dist: dict[str, str] = {r["station_code"]: r.get("nearest_region_distance_km", "")
                                      for r in station_rows}
    nearest_by_region: dict[str, str] = {}
    for r in station_rows:
        reg = r.get("nearest_region", "")
        if r.get("within_100km") == "true":
            # Keep the closest
            existing_dist = parse_decimal_comma_float(
                station_dist.get(nearest_by_region.get(reg, ""), "9999"), 9999.0)
            this_dist = parse_decimal_comma_float(r.get("nearest_region_distance_km", "9999"), 9999.0)
            if this_dist < existing_dist:
                nearest_by_region[reg] = r["station_code"]

    # Features by window
    feat_by_win: dict[str, dict[str, str]] = {}
    for r in feat_rows:
        if r.get("feature_status") == "ROLLING_CONTEXT_REVIEW_ONLY":
            wid = r.get("event_window_id", "")
            # Keep the one with the highest rain_7d (most informative)
            existing = feat_by_win.get(wid)
            if existing is None:
                feat_by_win[wid] = r
            else:
                if parse_decimal_comma_float(r.get("rain_7d", "0"), 0.0) > \
                   parse_decimal_comma_float(existing.get("rain_7d", "0"), 0.0):
                    feat_by_win[wid] = r

    # Also read canonical precip for max_1d_in_window fallback
    precip_rows = read_csv_safe(DATASETS / "protocol_c_inmet_canonical_precipitation_index_v1tc.csv")
    # Build {region: {station_code: {date: mm}}}
    precip_by_st: dict[str, dict[str, float]] = defaultdict(dict)
    for r in precip_rows:
        if r.get("provenance_status") == "OFFICIAL_INMET_CANONICAL_REVIEW_ONLY":
            code = r.get("station_code", "")
            d = r.get("date", "")
            mm = parse_decimal_comma_float(r.get("precipitation_mm", "0"), 0.0)
            if code and d:
                precip_by_st[code][d] = mm

    rows: list[dict[str, Any]] = []
    for w in wins:
        if w.get("blocked_reason"):
            continue
        wid = w["event_window_id"]
        region = w.get("region", "")
        ev_d = parse_date_safe(w.get("parsed_date", ""))

        feat = feat_by_win.get(wid)
        nearest_code = nearest_by_region.get(region, "")
        nearest_dist = station_dist.get(nearest_code, "")

        if feat:
            rain_1d = feat.get("rain_1d", "")
            rain_3d = feat.get("rain_3d", "")
            rain_7d = feat.get("rain_7d", "")
            max_1d  = feat.get("max_1d_in_window", "")
            used_code = feat.get("station_code", nearest_code)
            used_dist = station_dist.get(used_code, nearest_dist)
        elif ev_d and nearest_code:
            # Compute from canonical precip index
            daily = precip_by_st.get(nearest_code, {})
            roll = rolling_window_summary(daily, ev_d, windows=(1, 3, 7))
            rain_1d = str(roll.get("rain_1d", ""))
            rain_3d = str(roll.get("rain_3d", ""))
            rain_7d = str(roll.get("rain_7d", ""))
            max_1d  = str(max(daily.values(), default=0.0))
            used_code = nearest_code
            used_dist = nearest_dist
        else:
            rain_1d = rain_3d = rain_7d = max_1d = ""
            used_code = nearest_code
            used_dist = nearest_dist

        window_str = f"{w.get('window_start','')} to {w.get('window_end','')}"
        row: dict[str, Any] = {
            "hydromet_evidence_id":          f"V1TD_{hash_short(wid, 10)}",
            "event_candidate_id":            w.get("event_candidate_id", ""),
            "region":                        region,
            "event_window":                  window_str,
            "nearest_station_code":          used_code,
            "nearest_station_distance_km":   used_dist,
            "rain_1d":                       rain_1d,
            "rain_3d":                       rain_3d,
            "rain_7d":                       rain_7d,
            "max_1d_in_window":              max_1d,
            "evidence_role":                 "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY",
            "supports_manual_review":        "true",
            "does_not_validate_event":       "true",
            "notes":                         "",
        }
        row.update(guardrail_row())
        rows.append(row)

    if not rows:
        rows = [{
            "hydromet_evidence_id": "FAIL_CLOSED_NO_WINDOWS",
            "event_candidate_id": "", "region": "", "event_window": "",
            "nearest_station_code": "", "nearest_station_distance_km": "",
            "rain_1d": "", "rain_3d": "", "rain_7d": "", "max_1d_in_window": "",
            "evidence_role": "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY",
            "supports_manual_review": "false", "does_not_validate_event": "true",
            "notes": "no event windows", **guardrail_row(),
        }]

    write_csv_with_header(OUT_BRG, rows, BRG_FIELDS)
    write_schema(SCHEMA_B, BRG_FIELDS, "v1td_bridge")

    with_rain = sum(1 for r in rows if r.get("rain_7d"))
    summary = [
        {"stat_key": "bridge_rows",              "stat_value": str(len(rows))},
        {"stat_key": "rows_with_rain_data",      "stat_value": str(with_rain)},
        {"stat_key": "validates_event",          "stat_value": "false"},
        {"stat_key": "creates_negative",         "stat_value": "false"},
        {"stat_key": "all_manual_review",        "stat_value": "true"},
        {"stat_key": "stage",                    "stat_value": "v1td"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1td_summary")

    write_doc(DOC, "v1td — Hydromet Event Evidence Bridge", [
        "## Objetivo",
        "Ponte entre contexto hidromet (v1sr-v1su) e revisão de eventos A/B. "
        "Para cada janela de evento: estação mais próxima, acumulados 1d/3d/7d, "
        "status de revisão manual.",
        f"## Resultado\nBridge rows: {len(rows)}. Com dados de chuva: {with_rain}.",
        "## Limitação",
        "Acumulados de precipitação não validam evento. A ponte suporta revisão "
        "humana mas não substitui evidência observacional independente.",
    ])
    print(f"[v1td] bridge={len(rows)} with_rain={with_rain}")
    return {"bridge": len(rows), "with_rain": with_rain}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1td hydromet evidence bridge").parse_args()
    run()
